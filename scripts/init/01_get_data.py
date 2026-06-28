import os
import shutil
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlencode

import psycopg
import requests
import whisper
import yt_dlp
from dotenv import load_dotenv
from imageio_ffmpeg import get_ffmpeg_exe
from psycopg.types.json import Jsonb
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import IpBlocked, YouTubeTranscriptApiException


API = "https://www.googleapis.com/youtube/v3"
CHANNEL = "https://www.youtube.com/@IONIS-STM/videos"
TRANSCRIPT_LANGUAGES = ("fr", "en")
DOWNLOAD_DIR = Path("downloads/youtube")
BIN_DIR = Path("downloads/bin")
WHISPER_MODEL = None


def youtube(endpoint, ignore_403=False, **params):
    url = f"{API}/{endpoint}"
    query = {**params, "key": os.environ["YOUTUBE_API_KEY"]}
    print(f"{url}?{urlencode({**query, 'key': '***'})}")
    time.sleep(float(os.getenv("YOUTUBE_API_SLEEP_SECONDS", "0.5")))

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


def playlist_video_ids(channel):
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

        page_token = page.get("nextPageToken")
        if not page_token:
            return video_ids


def chunks(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def fetch_videos(channel):
    videos = []
    for video_ids in chunks(playlist_video_ids(channel), 50):
        data = youtube(
            "videos",
            part="snippet,contentDetails,statistics",
            id=",".join(video_ids),
            maxResults=50,
        )
        videos += data["items"]
    return videos


def upsert_video(cursor, video):
    snippet = video["snippet"]
    content = video["contentDetails"]
    video_id = video["id"]

    cursor.execute(
        """
        INSERT INTO videos (
            youtube_video_id,
            title,
            description,
            url,
            published_at,
            duration_seconds,
            raw_json,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (youtube_video_id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            url = EXCLUDED.url,
            published_at = EXCLUDED.published_at,
            duration_seconds = EXCLUDED.duration_seconds,
            raw_json = EXCLUDED.raw_json,
            updated_at = now()
        RETURNING id
        """,
        (
            video_id,
            snippet["title"],
            snippet.get("description"),
            f"https://www.youtube.com/watch?v={video_id}",
            parse_datetime(snippet.get("publishedAt")),
            parse_duration(content["duration"]),
            Jsonb(video),
        ),
    )
    return cursor.fetchone()[0]


def upsert_daily_stats(cursor, video_db_id, video):
    stats = video.get("statistics", {})
    cursor.execute(
        """
        INSERT INTO video_daily_stats (
            video_id,
            snapshot_date,
            view_count,
            like_count,
            comment_count,
            raw_json
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (video_id, snapshot_date) DO UPDATE SET
            view_count = EXCLUDED.view_count,
            like_count = EXCLUDED.like_count,
            comment_count = EXCLUDED.comment_count,
            raw_json = EXCLUDED.raw_json
        """,
        (
            video_db_id,
            date.today(),
            int(stats["viewCount"]) if "viewCount" in stats else None,
            int(stats["likeCount"]) if "likeCount" in stats else None,
            int(stats["commentCount"]) if "commentCount" in stats else None,
            Jsonb(stats),
        ),
    )


def fetch_transcript(youtube_video_id):
    transcript_list = YouTubeTranscriptApi().list(youtube_video_id)

    try:
        transcript = transcript_list.find_transcript(TRANSCRIPT_LANGUAGES)
    except YouTubeTranscriptApiException:
        transcript = next(iter(transcript_list))

    fetched = transcript.fetch()
    segments = [
        {
            "text": segment.text,
            "start": segment.start,
            "duration": segment.duration,
        }
        for segment in fetched
    ]
    text = "\n".join(segment["text"] for segment in segments)

    return {
        "language_code": transcript.language_code,
        "language_name": transcript.language,
        "is_generated": transcript.is_generated,
        "text": text,
        "segments": segments,
    }


def fetch_transcript_with_retries(youtube_video_id):
    retries = int(os.getenv("TRANSCRIPT_RETRIES", "2"))
    retry_seconds = int(os.getenv("TRANSCRIPT_RETRY_SECONDS", "10"))

    for attempt in range(retries + 1):
        try:
            return fetch_transcript(youtube_video_id)
        except IpBlocked:
            raise
        except YouTubeTranscriptApiException:
            if attempt == retries:
                raise
            time.sleep(retry_seconds * (attempt + 1))


def ffmpeg_exe():
    source = Path(get_ffmpeg_exe())
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "ffmpeg.exe"
    if not target.exists():
        shutil.copy2(source, target)
    os.environ["PATH"] = f"{target.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return target


def whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        model_name = os.getenv("WHISPER_MODEL", "small")
        WHISPER_MODEL = whisper.load_model(model_name)
    return WHISPER_MODEL


def download_video_360p(youtube_video_id):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path
        for path in DOWNLOAD_DIR.glob(f"{youtube_video_id}.*")
        if path.is_file() and path.suffix not in {".part", ".ytdl"}
    )
    if existing:
        return existing[0]

    url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    output_template = str(DOWNLOAD_DIR / "%(id)s.%(ext)s")
    options = {
        "format": os.getenv(
            "YTDLP_FORMAT",
            "bestvideo[height<=360]+bestaudio/best[height<=360]/best",
        ),
        "merge_output_format": os.getenv("YTDLP_MERGE_FORMAT", "mp4"),
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
        if path.is_file() and path.suffix not in {".part", ".ytdl"}
    )
    if not downloaded:
        raise FileNotFoundError(f"Video telechargee introuvable pour {youtube_video_id}")
    return downloaded[0]


def transcribe_with_whisper(youtube_video_id):
    ffmpeg_exe()
    video_path = download_video_360p(youtube_video_id)
    language = os.getenv("WHISPER_LANGUAGE", "fr") or None
    result = whisper_model().transcribe(str(video_path), language=language)
    segments = [
        {
            "text": segment["text"].strip(),
            "start": segment["start"],
            "duration": segment["end"] - segment["start"],
        }
        for segment in result.get("segments", [])
    ]
    text = "\n".join(segment["text"] for segment in segments).strip()

    return {
        "language_code": result.get("language") or language or "unknown",
        "language_name": f"Whisper {result.get('language') or language or 'unknown'}",
        "is_generated": True,
        "text": text,
        "segments": segments,
    }


def fetch_transcript_from_config(youtube_video_id):
    source = os.getenv("TRANSCRIPT_SOURCE", "whisper").lower()
    if source == "whisper":
        return transcribe_with_whisper(youtube_video_id)
    if source == "youtube":
        return fetch_transcript_with_retries(youtube_video_id)
    if source == "auto":
        try:
            return fetch_transcript_with_retries(youtube_video_id)
        except IpBlocked:
            print(f"IP bloquee pour les transcriptions YouTube, fallback Whisper pour {youtube_video_id}")
            return transcribe_with_whisper(youtube_video_id)
        except YouTubeTranscriptApiException:
            print(f"Transcription YouTube indisponible pour {youtube_video_id}, fallback Whisper")
            return transcribe_with_whisper(youtube_video_id)
    raise ValueError(f"TRANSCRIPT_SOURCE invalide: {source}")


def upsert_transcript(cursor, video_db_id, youtube_video_id):
    try:
        transcript = fetch_transcript_from_config(youtube_video_id)
    except IpBlocked:
        print(f"IP bloquee pour la transcription YouTube de {youtube_video_id}")
        return True
    except YouTubeTranscriptApiException as error:
        print(f"Transcription YouTube indisponible pour {youtube_video_id}: {error.__class__.__name__}")
        return True
    except Exception as error:
        print(f"Transcription Whisper indisponible pour {youtube_video_id}: {error.__class__.__name__} - {error}")
        return True

    cursor.execute(
        """
        INSERT INTO video_transcripts (
            video_id,
            language_code,
            language_name,
            is_generated,
            text,
            segments,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (video_id, language_code) DO UPDATE SET
            language_name = EXCLUDED.language_name,
            is_generated = EXCLUDED.is_generated,
            text = EXCLUDED.text,
            segments = EXCLUDED.segments,
            updated_at = now()
        """,
        (
            video_db_id,
            transcript["language_code"],
            transcript["language_name"],
            transcript["is_generated"],
            transcript["text"],
            Jsonb(transcript["segments"]),
        ),
    )
    time.sleep(int(os.getenv("TRANSCRIPT_SLEEP_SECONDS", "5")))
    return True


def upsert_comment(cursor, video_db_id, comment, parent_db_id=None):
    snippet = comment["snippet"]
    cursor.execute(
        """
        INSERT INTO comments (
            video_id,
            parent_comment_id,
            youtube_comment_id,
            author_name,
            author_channel_id,
            text,
            like_count,
            published_at,
            updated_at_youtube,
            raw_json,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (youtube_comment_id) DO UPDATE SET
            parent_comment_id = EXCLUDED.parent_comment_id,
            author_name = EXCLUDED.author_name,
            author_channel_id = EXCLUDED.author_channel_id,
            text = EXCLUDED.text,
            like_count = EXCLUDED.like_count,
            published_at = EXCLUDED.published_at,
            updated_at_youtube = EXCLUDED.updated_at_youtube,
            raw_json = EXCLUDED.raw_json,
            updated_at = now()
        RETURNING id
        """,
        (
            video_db_id,
            parent_db_id,
            comment["id"],
            snippet.get("authorDisplayName"),
            snippet.get("authorChannelId", {}).get("value"),
            snippet.get("textOriginal") or snippet.get("textDisplay") or "",
            snippet.get("likeCount"),
            parse_datetime(snippet.get("publishedAt")),
            parse_datetime(snippet.get("updatedAt")),
            Jsonb(comment),
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


def main():
    load_dotenv()
    videos = fetch_videos(CHANNEL)

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            for video in videos:
                video_db_id = upsert_video(cursor, video)
                upsert_daily_stats(cursor, video_db_id, video)
                upsert_transcript(cursor, video_db_id, video["id"])
                import_comments(cursor, video_db_id, video["id"])

    print(f"{len(videos)} videos importees en base")


if __name__ == "__main__":
    main()
