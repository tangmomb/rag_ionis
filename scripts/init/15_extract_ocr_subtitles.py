import argparse
import json
from pathlib import Path

from pipeline_analysis import analysed_infos_path, update_analysed_infos
from pipeline_paths import existing_ocr_dir, relative_to_video_dir, transcripts_dir

DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
OCR_DIR_NAME = "ocr"
TRANSCRIPT_OCR_DIR_NAME = "transcripts"
OCR_PROCESSED_NAME = "processed_ocr_items.json"
OCR_PROCESSED_CORRECTED_NAME = "corrected_ocr_items.json"
LEGACY_OCR_PROCESSED_NAME = "ocr_processed.json"
LEGACY_OCR_PROCESSED_CORRECTED_NAME = "ocr_processed_corrected.json"
OCR_SUBTITLE_NAME = "ocr_subtitles.txt"
OCR_SUBTITLE_TIMECODES_NAME = "ocr_subtitles_timecoded.txt"
LEGACY_OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
LEGACY_OCR_SUBTITLE_TIMECODES_SUFFIX = "_ocr_subtitle_timecodes.txt"
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
    ocr_dir = existing_ocr_dir(video_path)
    corrected = ocr_dir / OCR_PROCESSED_CORRECTED_NAME
    legacy_corrected = ocr_dir / LEGACY_OCR_PROCESSED_CORRECTED_NAME
    if corrected.exists():
        return corrected
    if legacy_corrected.exists():
        return legacy_corrected
    processed = ocr_dir / OCR_PROCESSED_NAME
    legacy_processed = ocr_dir / LEGACY_OCR_PROCESSED_NAME
    if legacy_processed.exists() and not processed.exists():
        return legacy_processed
    return processed


def subtitle_path(video_path):
    transcript_dir = transcripts_dir(video_path)
    preferred = transcript_dir / OCR_SUBTITLE_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_OCR_SUBTITLE_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def subtitle_timecodes_path(video_path):
    transcript_dir = transcripts_dir(video_path)
    preferred = transcript_dir / OCR_SUBTITLE_TIMECODES_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_OCR_SUBTITLE_TIMECODES_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_processed_items(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    return sorted(items, key=lambda item: (item.get("second", 0), item.get("image", ""), item.get("text", "")))


def analysed_has_subtitles(video_path):
    path = analysed_infos_path(video_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("has_subtitles")
    return value if isinstance(value, bool) else None


def has_ocr_subtitles(items):
    return any(str(item.get("kind", "")).strip().lower() == "subtitle" for item in items)


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


def format_timecode(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def collect_subtitles(items):
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
        subtitles.append(
            {
                "text": cleaned,
                "second": int(item.get("second", 0)),
            }
        )
    return subtitles


def render_subtitles(items):
    return " ".join(item["text"] for item in items)


def render_subtitles_timecodes(items):
    lines = []
    for item in items:
        lines.append(f"[{format_timecode(item['second'])}] {item['text']}")
    return "\n".join(lines)


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
        has_subtitles = analysed_has_subtitles(video_path)
        if has_subtitles is not True:
            print(f"[skip] {video_path.name}: pipeline_analysis.has_subtitles n'est pas true")
            continue

        source = processed_ocr_path(video_path)
        target = subtitle_path(video_path)
        target_timecodes = subtitle_timecodes_path(video_path)
        if target.exists() and not args.force:
            print(f"[skip] {target.name} existe deja")
            if target_timecodes.exists() and not args.force:
                done += 1
                continue
        if not source.exists():
            print(f"[skip] OCR traite introuvable: {source}")
            continue

        items = load_processed_items(source)
        if not has_ocr_subtitles(items):
            print(f"[skip] aucun kind=subtitle dans {source.name}")
            continue
        subtitles = collect_subtitles(items)
        text = render_subtitles(subtitles)
        timecoded_text = render_subtitles_timecodes(subtitles)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + ("\n" if text else ""), encoding="utf-8")
        target_timecodes.write_text(timecoded_text + ("\n" if timecoded_text else ""), encoding="utf-8")
        update_analysed_infos(
            video_path,
            "ocr_subtitles",
            {
                "status": "done",
                "source": relative_to_video_dir(source, video_path),
                "subtitle_file": relative_to_video_dir(target, video_path),
                "subtitle_timecodes_file": relative_to_video_dir(target_timecodes, video_path),
                "subtitle_count": len(subtitles),
            },
        )
        print(f"[ok] {target}")
        print(f"[ok] {target_timecodes}")
        done += 1

    print(f"{done} subtitles OCR generes.")


if __name__ == "__main__":
    main()
