import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
TIMECODED_SUFFIX = "_transcript_timecodes.txt"
CORRECTED_SUFFIX = "_transcript_timecodes_corrected.txt"
ENRICHED_SUFFIX = "_transcript_timecodes_enrichi.txt"
TRANSCRIPT_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})-((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_timecode(value):
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def format_timecode(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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
        path for path in parent_dir.iterdir() if path.is_dir() and any(video_files(path))
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


def transcript_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{CORRECTED_SUFFIX}"


def enriched_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{ENRICHED_SUFFIX}"


def analyse_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}_ocr_processed.json"


def parse_transcript(path):
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TRANSCRIPT_LINE.match(line)
        if not match:
            continue
        start, end, text = match.groups()
        segments.append({"start": parse_timecode(start), "end": parse_timecode(end), "line": line})
    return segments


def parse_analyse(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for item in payload.get("items", []):
        if str(item.get("kind", "")).strip().lower() == "subtitle":
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        second = item.get("second")
        if second is None:
            timecode = str(item.get("timecode", "")).strip()
            if not timecode:
                continue
            second = parse_timecode(timecode)
        items.append({"second": int(second), "text": text})
    return sorted(items, key=lambda item: item["second"])


def enrich_transcript(video_path, force=False):
    source = transcript_path(video_path)
    analyse = analyse_path(video_path)
    target = enriched_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] transcript introuvable: {source}")
        return None
    if not analyse.exists():
        print(f"[skip] analyse introuvable: {analyse}")
        return None

    segments = parse_transcript(source)
    overlays = parse_analyse(analyse)
    lines = []
    overlay_index = 0

    for segment in segments:
        while overlay_index < len(overlays) and overlays[overlay_index]["second"] <= segment["start"]:
            overlay = overlays[overlay_index]
            lines.append(format_overlay_line(overlay))
            overlay_index += 1

        while overlay_index < len(overlays) and segment["start"] < overlays[overlay_index]["second"] <= segment["end"]:
            overlay = overlays[overlay_index]
            lines.append(format_overlay_line(overlay))
            overlay_index += 1

        lines.append(segment["line"])

    while overlay_index < len(overlays):
        lines.append(format_overlay_line(overlays[overlay_index]))
        overlay_index += 1

    target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"[ok] {target}")
    return target


def format_overlay_line(overlay):
    timecode = format_timecode(overlay["second"])
    return f"[{timecode}] TEXTE ECRIT SUR LA VIDEO: {overlay['text']}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ajoute les textes visibles a l'ecran dans les transcripts timecodes."
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
        help="Regenere les transcripts enrichis meme s'ils existent deja.",
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
        if enrich_transcript(video_path, force=args.force):
            done += 1

    print(f"{done} transcripts enrichis.")


if __name__ == "__main__":
    main()
