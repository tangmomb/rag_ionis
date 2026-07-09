import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
import yt_dlp
from dotenv import load_dotenv
from imageio_ffmpeg import get_ffmpeg_exe


API = "https://www.googleapis.com/youtube/v3"
CHANNEL = "https://www.youtube.com/@IONIS-STM/videos"
DOWNLOAD_DIR = Path("downloads/youtube")
BIN_DIR = Path("downloads/bin")
DEFAULT_YOUTUBE_API_SLEEP_SECONDS = 0.5
DEFAULT_YTDLP_FORMAT_360P = "bestvideo[height<=360]+bestaudio/best[height<=360]/best"
DEFAULT_YTDLP_MERGE_FORMAT = "mp4"
YOUTUBE_API_INFOS_SUFFIX = ".youtube_api_infos.json"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def youtube(endpoint, ignore_403=False, **params):
    url = f"{API}/{endpoint}"
    query = {**params, "key": os.environ["YOUTUBE_API_KEY"]}
    print(f"{url}?{urlencode({**query, 'key': '***'})}")
    time.sleep(DEFAULT_YOUTUBE_API_SLEEP_SECONDS)

    response = requests.get(url, params=query, timeout=30)
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


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


def upsert_comment(cursor, video_db_id, comment, parent_db_id=None):
    snippet = comment["snippet"]
    cursor.execute(
        """
        INSERT INTO comments (
            video_id,
            parent_comment_id,
            youtube_comment_id,
            author_name,
            text,
            like_count,
            published_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (youtube_comment_id) DO UPDATE SET
            parent_comment_id = EXCLUDED.parent_comment_id,
            author_name = EXCLUDED.author_name,
            text = EXCLUDED.text,
            like_count = EXCLUDED.like_count,
            published_at = EXCLUDED.published_at
        RETURNING id
        """,
        (
            video_db_id,
            parent_db_id,
            comment["id"],
            snippet.get("authorDisplayName"),
            snippet.get("textOriginal") or snippet.get("textDisplay") or "",
            snippet.get("likeCount"),
            parse_datetime(snippet.get("publishedAt")),
        ),
    )
    return cursor.fetchone()[0]


def import_comment_replies(cursor, video_db_id, parent_comment_id, parent_db_id):
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

        for reply in page["items"]:
            upsert_comment(cursor, video_db_id, reply, parent_db_id)

        page_token = page.get("nextPageToken")
        if not page_token:
            return


def import_comments(cursor, video_db_id, youtube_video_id):
    page_token = None
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

        for thread in page["items"]:
            top_comment = thread["snippet"]["topLevelComment"]
            top_comment_db_id = upsert_comment(cursor, video_db_id, top_comment)

            replies = thread.get("replies", {}).get("comments", [])
            for reply in replies:
                upsert_comment(cursor, video_db_id, reply, top_comment_db_id)

            if thread["snippet"].get("totalReplyCount", 0) > len(replies):
                import_comment_replies(cursor, video_db_id, top_comment["id"], top_comment_db_id)

        page_token = page.get("nextPageToken")
        if not page_token:
            return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Importe les donnees YouTube de la chaine en base SQL."
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
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    videos = fetch_video(args.video_url) if args.video_url else fetch_videos(CHANNEL, limit=args.limit)
    info_dir = Path(args.download_dir) / "info_videos"
    if info_dir.exists():
        shutil.rmtree(info_dir)
    for video in videos:
        write_video_info(video, info_dir)

    print(f"{len(videos)} videos preparees en cache metadata")


if __name__ == "__main__":
    main()
