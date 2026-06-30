import argparse
import json
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
OCR_PROCESSED_SUFFIX = "_ocr_processed.json"
OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


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


def processed_ocr_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{OCR_PROCESSED_SUFFIX}"


def subtitle_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{OCR_SUBTITLE_SUFFIX}"


def load_processed_items(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    return sorted(items, key=lambda item: (item.get("second", 0), item.get("image", ""), item.get("text", "")))


def normalize_text(text):
    return "".join(str(text).casefold().split())


def one_edit_apart(left, right):
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False

    if len(left) == len(right):
        differences = sum(1 for left_char, right_char in zip(left, right) if left_char != right_char)
        return differences <= 1

    shorter, longer = sorted((left, right), key=len)
    short_index = 0
    long_index = 0
    differences = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        differences += 1
        if differences > 1:
            return False
        long_index += 1
    return True


def is_duplicate_subtitle(normalized, seen_normalized):
    if normalized in seen_normalized:
        return True
    if len(normalized) < 12:
        return False
    return any(one_edit_apart(normalized, previous) for previous in seen_normalized if len(previous) >= 12)


def render_subtitles(items):
    subtitles = []
    seen_normalized = set()
    for item in items:
        if str(item.get("kind", "")).strip().lower() != "subtitle":
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        cleaned = " ".join(text.split())
        normalized = normalize_text(cleaned)
        if is_duplicate_subtitle(normalized, seen_normalized):
            continue
        seen_normalized.add(normalized)
        subtitles.append(cleaned)
    return " ".join(subtitles)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolide les sous-titres OCR a partir du JSON OCR traite."
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
        "--limit-videos",
        type=int,
        help="Nombre maximum de videos a analyser.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le fichier OCR subtitle meme s'il existe deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        source = processed_ocr_path(video_path)
        target = subtitle_path(video_path)
        if target.exists() and not args.force:
            print(f"[skip] {target.name} existe deja")
            done += 1
            continue
        if not source.exists():
            print(f"[skip] OCR traite introuvable: {source}")
            continue

        items = load_processed_items(source)
        text = render_subtitles(items)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + ("\n" if text else ""), encoding="utf-8")
        print(f"[ok] {target}")
        done += 1

    print(f"{done} subtitles OCR generes.")


if __name__ == "__main__":
    main()
