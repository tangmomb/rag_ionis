import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PIPELINE_TRANSCRIPTS_DIR_NAME"] = "transcripts_ocr"

from pipeline.support.analysis import update_analysed_infos  # noqa: E402
from pipeline.support.paths import existing_transcripts_dir, relative_to_video_dir  # noqa: E402


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SOURCE_NAME = "ocr_subtitles_timecoded_corrected.txt"
LEGACY_SOURCE_SUFFIX = "_ocr_subtitle_timecodes_corrected.txt"
IONIS_STM_PATTERNS = (
    re.compile(r"(?i)\bl['’]?\s*ionis\s*-\s*stm\b"),
    re.compile(r"(?i)\bl['’]?\s*ionis\s+stm\b"),
    re.compile(r"(?i)\bl['’]?\s*onis\s*-\s*stm\b"),
    re.compile(r"(?i)\bl['’]?\s*onis\s+stm\b"),
    re.compile(r"(?i)\bl\s+ionis\s*-\s*stm\b"),
    re.compile(r"(?i)\bl\s+ionis\s+stm\b"),
    re.compile(r"(?i)\bl\s+onis\s*-\s*stm\b"),
    re.compile(r"(?i)\bl\s+onis\s+stm\b"),
    re.compile(r"(?i)\blonis\s*-\s*stm\b"),
    re.compile(r"(?i)\blonis\s+stm\b"),
    re.compile(r"(?i)\bionis\s*-\s*stm\b"),
    re.compile(r"(?i)\bionis\s+stm\b"),
    re.compile(r"(?i)\bonis\s*-\s*stm\b"),
    re.compile(r"(?i)\bonis\s+stm\b"),
)
TARGET_TEXT = "Ionis-STM"


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
    transcript_dir = existing_transcripts_dir(video_path)
    preferred = transcript_dir / SOURCE_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_SOURCE_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def normalize_ionis_stm(text):
    updated = str(text)
    replacements = 0
    for pattern in IONIS_STM_PATTERNS:
        updated, count = pattern.subn(TARGET_TEXT, updated)
        replacements += count
    return updated, replacements


def process_video(video_path, force=False):
    target = transcript_path(video_path)
    if not target.exists():
        print(f"[skip] transcript OCR corrige introuvable: {target}")
        return False

    original = target.read_text(encoding="utf-8")
    updated, replacements = normalize_ionis_stm(original)
    if replacements == 0 and not force:
        print(f"[skip] {target.name}: aucune variante Ionis-STM detectee")
        return False

    target.write_text(updated, encoding="utf-8")
    update_analysed_infos(
        video_path,
        "normalize_ionis_stm",
        {
            "status": "done",
            "target_file": relative_to_video_dir(target, video_path),
            "replacement_count": replacements,
            "target_text": TARGET_TEXT,
        },
    )
    print(f"[ok] {target} ({replacements} remplacement(s))")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalise les variantes de Ionis-STM dans le transcript OCR corrige."
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
        help="Reecrit le fichier meme si aucune variante n'est detectee.",
    )
    return parser.parse_args()


def main():
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if process_video(video_path, force=args.force):
            done += 1

    print(f"{done} transcript(s) normalise(s).")


if __name__ == "__main__":
    main()
