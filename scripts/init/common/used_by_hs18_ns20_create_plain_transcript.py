import argparse
import re
import sys
from pathlib import Path

from common.pipeline_analysis import update_analysed_infos
from common.pipeline_paths import existing_transcripts_dir, relative_to_video_dir

DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
CORRECTED_TIMECODED_NAMES = (
    "whisper_transcript_timecoded_corrected.txt",
    "ocr_subtitles_timecoded_corrected.txt",
)
LEGACY_CORRECTED_TIMECODED_SUFFIXES = (
    "_transcript_timecodes_corrected.txt",
    "_ocr_subtitle_timecodes_corrected.txt",
)
PLAIN_NAME = "plain_transcript.txt"
LEGACY_PLAIN_SUFFIX = "_transcript.txt"
TIMECODE_PREFIX = re.compile(
    r"^\[(?:\d{2}:)?\d{2}:\d{2}-(?:\d{2}:)?\d{2}:\d{2}\]\s*(?:[A-Z][A-Z0-9_-]*:\s*)?"
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
    if input_path.name in CORRECTED_TIMECODED_NAMES:
        return input_path.with_name(PLAIN_NAME)
    name = input_path.name
    for suffix in LEGACY_CORRECTED_TIMECODED_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)] + LEGACY_PLAIN_SUFFIX
            break
    else:
        name = input_path.stem + LEGACY_PLAIN_SUFFIX
    return input_path.with_name(name)


def timecoded_inputs(transcript_dir):
    inputs = [transcript_dir / name for name in CORRECTED_TIMECODED_NAMES if (transcript_dir / name).exists()]
    if inputs:
        return inputs

    legacy_inputs = []
    for suffix in LEGACY_CORRECTED_TIMECODED_SUFFIXES:
        legacy_inputs.extend(sorted(transcript_dir.glob(f"*{suffix}")))
    return legacy_inputs


def convert_file(video_path, input_path, force=False):
    target = output_path(input_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target

    text = input_path.read_text(encoding="utf-8")
    cleaned = strip_timecodes(text)
    target.write_text(cleaned + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "strip_timecodes",
        {
            "status": "done",
            "source": relative_to_video_dir(input_path, video_path),
            "plain_transcript_file": relative_to_video_dir(target, video_path),
            "char_count": len(cleaned),
        },
    )
    print(f"[ok] {target}")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cree des transcriptions sans timecodes depuis les fichiers timecoded corrected."
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
    print(f"Dossier videos: {video_dir}")
    done = 0
    videos = list(video_files(video_dir))
    for video_path in videos:
        for input_path in timecoded_inputs(existing_transcripts_dir(video_path)):
            convert_file(video_path, input_path, force=args.force)
            done += 1

    if not done:
        print(f"Aucun fichier timecoded corrected trouve dans {video_dir}")
        return

    print(f"{done} fichiers traites.")


if __name__ == "__main__":
    main()
