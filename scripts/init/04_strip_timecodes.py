import argparse
import re
import sys
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
TIMECODED_SUFFIX = "_transcript_timecodes.txt"
PLAIN_SUFFIX = "_transcript.txt"
TIMECODE_PREFIX = re.compile(
    r"^\[(?:\d{2}:)?\d{2}:\d{2}-(?:\d{2}:)?\d{2}:\d{2}\]\s*(?:[A-Z][A-Z0-9_-]*:\s*)?"
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def video_files(video_dir):
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def latest_video_dir(parent_dir):
    candidates = sorted(
        path
        for path in parent_dir.iterdir()
        if path.is_dir() and any(video_files(path))
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


def strip_timecodes(text):
    cleaned_lines = []
    for line in text.splitlines():
        cleaned = TIMECODE_PREFIX.sub("", line).strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    return " ".join(cleaned_lines)


def output_path(input_path):
    name = input_path.name.removesuffix(TIMECODED_SUFFIX) + PLAIN_SUFFIX
    return input_path.with_name(name)


def convert_file(input_path, force=False):
    target = output_path(input_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target

    text = input_path.read_text(encoding="utf-8")
    cleaned = strip_timecodes(text)
    target.write_text(cleaned + "\n", encoding="utf-8")
    print(f"[ok] {target}")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cree des transcriptions sans timecodes depuis les fichiers *_transcript_timecodes.txt."
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
        help="Regenere les fichiers sans timecodes meme s'ils existent deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    transcript_dir = video_dir / "transcript"
    inputs = sorted(transcript_dir.glob(f"*{TIMECODED_SUFFIX}"))

    if not inputs:
        print(f"Aucun fichier *{TIMECODED_SUFFIX} trouve dans {transcript_dir}")
        return

    print(f"Dossier transcriptions: {transcript_dir}")
    done = 0
    for input_path in inputs:
        convert_file(input_path, force=args.force)
        done += 1

    print(f"{done} fichiers traites.")


if __name__ == "__main__":
    main()
