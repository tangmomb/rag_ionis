import argparse
import json
import re
import sys
from pathlib import Path

from pipeline_analysis import update_analysed_infos
from ocr_processed_filtering import filtered_ocr_path, format_timecode
from pipeline_paths import existing_transcripts_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
CORRECTED_SUFFIX = "_corrected.txt"
ENRICHED_SUFFIX = "_enriched.txt"
LEGACY_ENRICHED_SUFFIX = "_enrichi.txt"
TRANSCRIPT_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})-((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
SUBTITLE_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")

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


def timecodes_path(video_path):
    transcript_dir = existing_transcripts_dir(video_path)
    candidates = (
        transcript_dir / "whisper_transcript_timecoded_corrected.txt",
        transcript_dir / "ocr_subtitles_timecoded_corrected.txt",
        transcript_dir / f"{video_path.stem}_transcript_timecodes_corrected.txt",
        transcript_dir / f"{video_path.stem}_ocr_subtitle_timecodes_corrected.txt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Aucun fichier timecodes corrige trouve pour {video_path.stem} dans {transcript_dir}")


def enriched_path(source_path):
    if source_path.name.endswith(CORRECTED_SUFFIX):
        return source_path.with_name(source_path.name[: -len(".txt")] + ENRICHED_SUFFIX)
    return source_path.with_name(f"{source_path.stem}{ENRICHED_SUFFIX}")


def parse_timecoded_source(path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        transcript_match = TRANSCRIPT_LINE.match(line)
        if transcript_match:
            start, _, _ = transcript_match.groups()
            items.append({"second": parse_timecode(start), "line": line})
            continue
        subtitle_match = SUBTITLE_LINE.match(line)
        if subtitle_match:
            second, _ = subtitle_match.groups()
            items.append({"second": parse_timecode(second), "line": line})
            continue
        items.append({"second": 10**12, "line": line})
    return items


def load_filtered_overlays(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    kinds = payload.get("kinds", {})
    overlays = []
    for kind, values in kinds.items():
        if kind == "subtitle":
            continue
        if not isinstance(values, dict):
            continue
        for timecode, text in values.items():
            cleaned = " ".join(str(text).split())
            if not cleaned:
                continue
            overlays.append(
                {
                    "kind": kind,
                    "second": parse_timecode(timecode),
                    "text": cleaned,
                }
            )
    return sorted(overlays, key=lambda item: item["second"])


def overlay_label_key(item):
    kind = item.get("kind")
    if kind == "question_intertitle":
        return "question_intertitle"
    if kind == "outro":
        return "outro"
    if kind == "graphic":
        return "graphic"
    return "on_footage"


def format_overlay_line(overlay):
    timecode = format_timecode(overlay["second"])
    labels = {
        "question_intertitle": "INTERCALAIRE QUESTION",
        "on_footage": "ON_FOOTAGE",
        "outro": "OUTRO",
        "graphic": "GRAPHIC",
    }
    label = labels[overlay_label_key(overlay)]
    return f"[{timecode}] {label}: {overlay['text']}"


def enrich_transcript(video_path, force=False):
    source = timecodes_path(video_path)
    analyse = filtered_ocr_path(video_path)
    target = enriched_path(source)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] timecodes corrige introuvable: {source}")
        return None
    if not analyse.exists():
        print(f"[skip] analyse filtree introuvable: {analyse}")
        return None

    source_lines = parse_timecoded_source(source)
    overlays = load_filtered_overlays(analyse)
    lines = []
    source_index = 0
    overlay_index = 0

    while source_index < len(source_lines):
        current_second = source_lines[source_index]["second"]
        while overlay_index < len(overlays) and overlays[overlay_index]["second"] <= current_second:
            lines.append(format_overlay_line(overlays[overlay_index]))
            overlay_index += 1

        lines.append(source_lines[source_index]["line"])
        source_index += 1

    while overlay_index < len(overlays):
        lines.append(format_overlay_line(overlays[overlay_index]))
        overlay_index += 1

    target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "enrich_transcripts",
        {
            "status": "done",
            "source": relative_to_video_dir(source, video_path),
            "analysis_source": relative_to_video_dir(analyse, video_path),
            "enriched_file": relative_to_video_dir(target, video_path),
            "overlay_count": len(overlays),
        },
    )
    print(f"[ok] {target}")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ajoute les textes visibles a l'ecran dans les timecodes corriges."
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
        help="Regenere les fichiers enrichis meme s'ils existent deja.",
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

    print(f"{done} fichiers enrichis.")


if __name__ == "__main__":
    main()
