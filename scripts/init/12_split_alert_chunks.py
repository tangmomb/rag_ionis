import argparse
import json
import sys
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
CHUNKS_SUFFIX = "_transcript_chunks.json"
ALERT_WORD_THRESHOLD = 3000

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


def chunks_path(video_path):
    return video_path.parent / "chunks" / f"{video_path.stem}{CHUNKS_SUFFIX}"


def word_count(text):
    return len([word for word in str(text).split() if word.strip()])


def split_alert_chunk(chunk):
    content = str(chunk.get("content", ""))
    if not chunk.get("alert"):
        return [chunk]

    body = content.strip()
    words = body.split()
    if len(words) <= ALERT_WORD_THRESHOLD:
        chunk["alert"] = False
        chunk["alert_reason"] = None
        return [chunk]

    midpoint = max(1, len(words) // 2)
    first = " ".join(words[:midpoint]).strip()
    second = " ".join(words[midpoint:]).strip()
    if not first or not second:
        chunk["alert"] = False
        chunk["alert_reason"] = None
        return [chunk]

    base_meta = chunk.get("meta_data", {})
    return [
        {
            **chunk,
            "chunk_index": f"{chunk.get('chunk_index')}.1",
            "meta_data": {**base_meta, "split_from_alert": True},
            "content": first,
            "alert": len(first.split()) > ALERT_WORD_THRESHOLD,
            "alert_reason": "over_3000_words" if len(first.split()) > ALERT_WORD_THRESHOLD else None,
            "char_count": len(first),
        },
        {
            **chunk,
            "chunk_index": f"{chunk.get('chunk_index')}.2",
            "meta_data": {**base_meta, "split_from_alert": True},
            "content": second,
            "alert": len(second.split()) > ALERT_WORD_THRESHOLD,
            "alert_reason": "over_3000_words" if len(second.split()) > ALERT_WORD_THRESHOLD else None,
            "char_count": len(second),
        },
    ]


def rewrite_chunks(video_path, force=False):
    source = chunks_path(video_path)
    if not source.exists():
        print(f"[skip] chunks introuvables: {source}")
        return None

    payload = json.loads(source.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    updated = []
    changed = False
    for chunk in chunks:
        content = str(chunk.get("content", ""))
        if chunk.get("alert") or word_count(content) > ALERT_WORD_THRESHOLD:
            changed = True
            updated.extend(split_alert_chunk(chunk))
        else:
            updated.append(chunk)

    if not changed and not force:
        print(f"[skip] {source.name} ne contient pas de chunk alert")
        return source

    payload["chunks"] = updated
    payload["chunking"] = {
        **payload.get("chunking", {}),
        "alert_word_threshold": ALERT_WORD_THRESHOLD,
    }
    source.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {source} ({len(updated)} chunks)")
    return source


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decoupe les chunks marqués ALERT dans le dossier chunks/."
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
        help="Regenere meme si rien n'est marqué ALERT.",
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
        if rewrite_chunks(video_path, force=args.force):
            done += 1

    print(f"{done} fichiers chunks relus.")


if __name__ == "__main__":
    main()
