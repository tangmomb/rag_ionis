import argparse
import os
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
PLAIN_SUFFIX = "_transcript.txt"
CHUNKS_SUFFIX = "_transcript_chunks.json"
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 200
ALERT_WORD_THRESHOLD = 3000
API = "https://www.googleapis.com/youtube/v3"
DEFAULT_YOUTUBE_API_SLEEP_SECONDS = 0.5
SPEAKER_PATTERNS = (
    re.compile(r"\bJe suis ([A-ZÉÈÀÂÊÎÔÛÄËÏÖÜÇ][A-Za-zÀ-ÖØ-öø-ÿ'’ -]+?)(?:,|\.| je | j'| et |$)"),
    re.compile(r"\bJe m'appelle ([A-ZÉÈÀÂÊÎÔÛÄËÏÖÜÇ][A-Za-zÀ-ÖØ-öø-ÿ'’ -]+?)(?:,|\.| je | j'| et |$)"),
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def video_files(video_dir):
    direct_videos = []
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            direct_videos.append(path)

    if direct_videos:
        yield from direct_videos
        return

    for child in sorted(video_dir.iterdir()):
        if not child.is_dir():
            continue
        for path in sorted(child.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


def latest_video_dir(parent_dir):
    candidates = sorted(path for path in parent_dir.iterdir() if path.is_dir() and any(video_files(path)))
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


def transcript_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{PLAIN_SUFFIX}"


def chunks_path(video_path):
    return video_path.parent / "chunks" / f"{video_path.stem}{CHUNKS_SUFFIX}"


def youtube(endpoint, **params):
    url = f"{API}/{endpoint}"
    query = {**params, "key": os.environ["YOUTUBE_API_KEY"]}
    print(f"{url}?{urlencode({**query, 'key': '***'})}")
    time.sleep(DEFAULT_YOUTUBE_API_SLEEP_SECONDS)
    response = requests.get(url, params=query, timeout=30)
    response.raise_for_status()
    return response.json()


def published_at(video_path):
    info_path = video_path.with_suffix(".info.json")
    if info_path.exists():
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            timestamp = payload.get("release_timestamp") or payload.get("timestamp")
            if timestamp:
                return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")
            upload_date = payload.get("upload_date")
            if upload_date and len(upload_date) == 8:
                return datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            pass

    video = youtube("videos", part="snippet", id=video_path.stem)
    items = video.get("items", [])
    if items:
        published = items[0]["snippet"].get("publishedAt")
        if published:
            return published

    dt = datetime.fromtimestamp(video_path.stat().st_mtime, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def video_url(video_path):
    return f"https://www.youtube.com/watch?v={video_path.stem}"


def video_title(video_path):
    info_path = video_path.with_suffix(".info.json")
    if info_path.exists():
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            title = payload.get("title")
            if title:
                return title
        except Exception:
            pass
    return video_path.stem


def extract_intervenants(text):
    names = []
    seen = set()
    for pattern in SPEAKER_PATTERNS:
        for match in pattern.finditer(text):
            name = " ".join(match.group(1).split()).strip(" ,.;:!?")
            if name and name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
    return names


def word_count(text):
    return len([word for word in str(text).split() if word.strip()])


def normalize_text(text):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines).strip()


def split_into_chunks(text, max_chars=DEFAULT_MAX_CHARS, overlap_chars=DEFAULT_OVERLAP_CHARS):
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]

    chunks = []
    current = []
    current_len = 0

    def flush_chunk():
        nonlocal current, current_len
        if not current:
            return
        chunk_text = "\n\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        current = []
        current_len = 0

    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)
        if not paragraph:
            continue
        projected = current_len + len(paragraph) + (2 if current else 0)
        if current and projected > max_chars:
            flush_chunk()
        if len(paragraph) > max_chars:
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(end - overlap_chars, start + 1)
            continue
        current.append(paragraph)
        current_len += len(paragraph) + (2 if len(current) > 1 else 0)

    flush_chunk()
    return chunks


def build_chunks_payload(text):
    chunks = split_into_chunks(text)
    return {
        "chunking": {
            "max_chars": DEFAULT_MAX_CHARS,
            "overlap_chars": DEFAULT_OVERLAP_CHARS,
        },
        "chunks": [
            {
                "chunk_index": index + 1,
                "meta_data": meta_data,
                "content": chunk,
                "alert": word_count(chunk) > ALERT_WORD_THRESHOLD,
                "alert_reason": "over_3000_words" if word_count(chunk) > ALERT_WORD_THRESHOLD else None,
                "char_count": len(chunk),
            }
            for index, chunk in enumerate(chunks)
        ],
    }


def create_chunks(video_path, force=False):
    source = transcript_path(video_path)
    target = chunks_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] transcript introuvable: {source}")
        return None

    text = source.read_text(encoding="utf-8")
    normalized = normalize_text(text)
    if not normalized:
        print(f"[skip] transcript vide: {source}")
        return None

    meta_data = {
        "video_name": video_path.stem,
        "video_title": video_title(video_path),
        "published_at": published_at(video_path),
        "intervenants": extract_intervenants(normalized),
        "video_url": video_url(video_path),
    }
    payload = build_chunks_payload(normalized)
    payload["meta_data"] = meta_data
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {target} ({len(payload['chunks'])} chunks)")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cree des chunks JSON a partir des transcripts sans timecodes."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier contenant les videos. Defaut: dernier sous-dossier de downloads/youtube",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent utilise si --video-dir est absent. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere les chunks meme s'ils existent deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if create_chunks(video_path, force=args.force):
            done += 1

    print(f"{done} transcripts chunkes.")


if __name__ == "__main__":
    main()
