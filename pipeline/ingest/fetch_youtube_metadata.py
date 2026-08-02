import argparse
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
import yt_dlp
from dotenv import load_dotenv
from imageio_ffmpeg import get_ffmpeg_exe

from pipeline.support.json_io import write_json
from pipeline.support.paths import (
    init_dir,
    youtube_api_infos_path,
    youtube_comments_path,
)


API = "https://www.googleapis.com/youtube/v3"
CHANNEL = "https://www.youtube.com/@IONIS-STM/videos"
DOWNLOAD_DIR = Path("downloads/youtube")
BIN_DIR = Path("downloads/bin")
DEFAULT_YOUTUBE_API_SLEEP_SECONDS = 0.5
DEFAULT_YOUTUBE_API_RETRIES = 3
DEFAULT_YOUTUBE_API_RETRY_DELAY_SECONDS = 1.0
# Keep the downloaded MP4 broadly compatible with Windows media players.
# Generic bestaudio often resolves to Opus/WebM, which is not reliably
# supported by the default Windows player.
DEFAULT_YTDLP_FORMAT_360P = "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best"
DEFAULT_YTDLP_MERGE_FORMAT = "mp4"
YOUTUBE_API_INFOS_SUFFIX = ".youtube_api_infos.json"
YOUTUBE_COMMENTS_SUFFIX = ".youtube_comments.json"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def youtube(endpoint, ignore_403=False, **params):
    url = f"{API}/{endpoint}"
    query = {**params, "key": os.environ["YOUTUBE_API_KEY"]}
    print(f"{url}?{urlencode({**query, 'key': '***'})}")
    response = None
    for attempt in range(1, DEFAULT_YOUTUBE_API_RETRIES + 1):
        time.sleep(DEFAULT_YOUTUBE_API_SLEEP_SECONDS)
        try:
            response = requests.get(url, params=query, timeout=30)
        except requests.RequestException:
            if attempt == DEFAULT_YOUTUBE_API_RETRIES:
                raise
            delay = DEFAULT_YOUTUBE_API_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            print(f"[retry] {endpoint}: erreur reseau, nouvelle tentative dans {delay:.0f}s")
            time.sleep(delay)
            continue
        if response.status_code != 429 and response.status_code < 500:
            break
        if attempt == DEFAULT_YOUTUBE_API_RETRIES:
            break
        delay = DEFAULT_YOUTUBE_API_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
        print(
            f"[retry] {endpoint}: HTTP {response.status_code}, "
            f"nouvelle tentative dans {delay:.0f}s"
        )
        time.sleep(delay)

    if response is None:
        raise RuntimeError(f"Aucune reponse YouTube pour {endpoint}.")
    if ignore_403 and response.status_code == 403:
        error = response.json().get("error", {})
        reason = error.get("errors", [{}])[0].get("reason")
        message = error.get("message", "")
        video_id = params.get("videoId", "")
        if endpoint == "commentThreads":
            print(f"commentThreads erreur 403 pour {video_id}: {reason} - {message}")
        if reason == "commentsDisabled":
            return {"items": []}

    response.raise_for_status()
    return response.json()


def get_channel_id(channel):
    channel = channel.strip().rstrip("/")
    if channel.endswith("/videos"):
        channel = channel.removesuffix("/videos")
    if channel.startswith("UC"):
        return channel
    if "/channel/" in channel:
        return channel.split("/channel/", 1)[1].split("/", 1)[0]

    handle = channel.split("/")[-1].removeprefix("@")
    return youtube("channels", part="id", forHandle=handle)["items"][0]["id"]


def uploads_playlist_id(channel):
    channel_id = get_channel_id(channel)
    data = youtube("channels", part="contentDetails", id=channel_id)
    return data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def parse_datetime(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def parse_duration(duration):
    total = 0
    number = ""
    for char in duration.removeprefix("P").removeprefix("T"):
        if char == "T":
            continue
        if char.isdigit():
            number += char
        elif number:
            total += int(number) * {"D": 86400, "H": 3600, "M": 60, "S": 1}[char]
            number = ""
    return total


def extract_youtube_video_id(value):
    raw_value = value.strip()
    parsed = urlparse(raw_value if "://" in raw_value else f"https://{raw_value}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = parts[1] if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"} else ""
    else:
        video_id = ""

    if len(video_id) != 11:
        raise argparse.ArgumentTypeError("Lien YouTube invalide ou ID video introuvable.")
    return video_id


def playlist_video_ids(channel, limit=None):
    playlist_id = uploads_playlist_id(channel)
    video_ids = []
    page_token = None

    while True:
        page = youtube(
            "playlistItems",
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token or "",
        )
        video_ids += [item["contentDetails"]["videoId"] for item in page["items"]]
        if limit is not None and len(video_ids) >= limit:
            return video_ids[:limit]

        page_token = page.get("nextPageToken")
        if not page_token:
            return video_ids


def chunks(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def fetch_videos(channel, limit=None):
    videos = []
    for video_ids in chunks(playlist_video_ids(channel, limit=limit), 50):
        data = youtube(
            "videos",
            part="snippet,contentDetails,statistics",
            id=",".join(video_ids),
            maxResults=50,
        )
        videos += data["items"]
    return videos[:limit] if limit is not None else videos


def fetch_video(video_url):
    video_id = extract_youtube_video_id(video_url)
    data = youtube(
        "videos",
        part="snippet,contentDetails,statistics",
        id=video_id,
        maxResults=1,
    )
    if not data.get("items"):
        raise RuntimeError(f"Video YouTube introuvable: {video_url}")
    return data["items"]


def video_info_payload(video):
    snippet = video["snippet"]
    content = video["contentDetails"]
    video_id = video["id"]
    published_at = parse_datetime(snippet.get("publishedAt"))
    return {
        **video,
        "youtube_video_id": video_id,
        "title": snippet["title"],
        "description": snippet.get("description"),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "published_at": published_at.isoformat().replace("+00:00", "Z") if published_at else None,
        "duration_seconds": parse_duration(content["duration"]),
        "thumbnail_medium_url": snippet.get("thumbnails", {}).get("medium", {}).get("url"),
        "statistics": video.get("statistics", {}),
    }

def write_video_info(video, info_dir):
    payload = video_info_payload(video)
    info_dir.mkdir(parents=True, exist_ok=True)
    path = info_dir / f"{video['id']}{YOUTUBE_API_INFOS_SUFFIX}"
    return write_json(path, payload)


def sync_existing_video_info(video, cache_path, download_dir):
    video_dir = init_dir(download_dir) / video["id"]
    has_local_video = video_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        for path in video_dir.iterdir()
    )
    if not has_local_video:
        return None

    target = youtube_api_infos_path(video_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, target)
    return target


def ffmpeg_exe():
    source = Path(get_ffmpeg_exe())
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "ffmpeg.exe"
    if not target.exists():
        shutil.copy2(source, target)
    os.environ["PATH"] = f"{target.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return target


def download_video_360p(youtube_video_id):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path
        for path in DOWNLOAD_DIR.glob(f"{youtube_video_id}.*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if existing:
        return existing[0]

    url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    output_template = str(DOWNLOAD_DIR / "%(id)s.%(ext)s")
    options = {
        "format": DEFAULT_YTDLP_FORMAT_360P,
        "merge_output_format": DEFAULT_YTDLP_MERGE_FORMAT,
        "outtmpl": output_template,
        "ffmpeg_location": str(ffmpeg_exe()),
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.extract_info(url, download=True)

    downloaded = sorted(
        path
        for path in DOWNLOAD_DIR.glob(f"{youtube_video_id}.*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not downloaded:
        raise FileNotFoundError(f"Video telechargee introuvable pour {youtube_video_id}")
    return downloaded[0]


def normalized_comment(comment, parent_youtube_comment_id=None):
    snippet = comment["snippet"]
    return {
        "youtube_comment_id": comment["id"],
        "parent_youtube_comment_id": parent_youtube_comment_id,
        "author_name": snippet.get("authorDisplayName"),
        "text": snippet.get("textOriginal") or snippet.get("textDisplay") or "",
        "like_count": snippet.get("likeCount"),
        "published_at": snippet.get("publishedAt"),
        "updated_at": snippet.get("updatedAt"),
    }


def import_comment_replies(
    parent_comment_id,
    comments_by_id,
):
    page_token = None
    while True:
        page = youtube(
            "comments",
            part="snippet",
            parentId=parent_comment_id,
            maxResults=100,
            pageToken=page_token or "",
            textFormat="plainText",
        )

        for reply in page.get("items", []):
            comments_by_id[reply["id"]] = normalized_comment(
                reply,
                parent_youtube_comment_id=parent_comment_id,
            )

        page_token = page.get("nextPageToken")
        if not page_token:
            return


def fetch_comments(youtube_video_id):
    page_token = None
    comments_by_id = {}
    while True:
        page = youtube(
            "commentThreads",
            ignore_403=True,
            part="snippet,replies",
            videoId=youtube_video_id,
            maxResults=100,
            pageToken=page_token or "",
            textFormat="plainText",
        )

        for thread in page.get("items", []):
            top_comment = thread["snippet"]["topLevelComment"]
            comments_by_id[top_comment["id"]] = normalized_comment(top_comment)

            replies = thread.get("replies", {}).get("comments", [])
            for reply in replies:
                comments_by_id[reply["id"]] = normalized_comment(
                    reply,
                    parent_youtube_comment_id=top_comment["id"],
                )

            if thread["snippet"].get("totalReplyCount", 0) > len(replies):
                import_comment_replies(
                    top_comment["id"],
                    comments_by_id,
                )

        page_token = page.get("nextPageToken")
        if not page_token:
            return {
                "youtube_video_id": youtube_video_id,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "comments": list(comments_by_id.values()),
            }


def write_comments(video_id, payload, comments_dir):
    target_dir = Path(comments_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{video_id}{YOUTUBE_COMMENTS_SUFFIX}"
    return write_json(target, payload)


def sync_existing_comments(video, cache_path, download_dir):
    video_dir = init_dir(download_dir) / video["id"]
    has_local_video = video_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        for path in video_dir.iterdir()
    )
    if not has_local_video:
        return None

    target = youtube_comments_path(video_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, target)
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recupere les metadonnees et commentaires YouTube en JSON local."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit",
        type=int,
        help="Nombre maximum de videos a importer.",
    )
    selection.add_argument(
        "--video-url",
        help="Lien YouTube d'une video precise a importer.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DOWNLOAD_DIR),
        help="Dossier parent utilise pour stocker les metadonnees locales. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Conserve l'option de compatibilite; les transcripts ne sont plus importes dans cette step.",
    )
    parser.add_argument(
        "--skip-comments",
        action="store_true",
        help="Ne recupere pas les commentaires YouTube dans le cache JSON.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    videos = fetch_video(args.video_url) if args.video_url else fetch_videos(CHANNEL, limit=args.limit)
    info_dir = Path(args.download_dir) / "init" / "_00_info_videos"
    if info_dir.exists():
        shutil.rmtree(info_dir)
    comments_count = 0
    comments_dir = Path(args.download_dir) / "init" / "_00_info_comments"
    if not args.skip_comments and comments_dir.exists():
        shutil.rmtree(comments_dir)

    synced_count = 0
    comment_files_count = 0
    for video in videos:
        cache_path = write_video_info(video, info_dir)
        if sync_existing_video_info(video, cache_path, args.download_dir) is not None:
            synced_count += 1
        if not args.skip_comments:
            comments_payload = fetch_comments(video["id"])
            comments_path = write_comments(
                video["id"],
                comments_payload,
                comments_dir,
            )
            sync_existing_comments(video, comments_path, args.download_dir)
            imported_count = len(comments_payload["comments"])
            comments_count += imported_count
            comment_files_count += 1
            print(
                f"[comments] {video['id']}: {imported_count} commentaire(s)",
                flush=True,
            )

    print(
        f"{len(videos)} videos preparees en cache metadata; "
        f"{synced_count} dossiers video synchronises; "
        f"{comments_count} commentaires dans {comment_files_count} fichier(s) JSON"
    )


if __name__ == "__main__":
    main()
