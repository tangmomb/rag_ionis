import argparse
import json
import re
import sys
from pathlib import Path

from pipeline.support.analysis import analysed_infos_path, update_analysed_infos
from pipeline.support.ocr_filtering import enriched_ocr_source_path, format_timecode
from pipeline.support.paths import (
    existing_speakers_dir,
    existing_transcripts_dir,
    relative_to_video_dir,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
CORRECTED_SUFFIX = "_corrected.txt"
ENRICHED_SUFFIX = "_enriched.txt"
LEGACY_ENRICHED_SUFFIX = "_enrichi.txt"
TRANSCRIPT_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})-((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
SUBTITLE_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
SPEAKER_LABEL_PATTERN = re.compile(r"\bSPEAKER_(\d+)\b")
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"
ENRICHED_GROUP_ORDER = ("speaker", "animations", "intercalaire")

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


def analysed_video_type(video_path):
    path = analysed_infos_path(video_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("video_type")
    return value if isinstance(value, str) else None


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


def whisper_timecoded_path(video_path):
    transcript_dir = existing_transcripts_dir(video_path)
    return transcript_dir / "whisper_transcript_timecoded.txt"


def enriched_path(source_path):
    if source_path.name.endswith(CORRECTED_SUFFIX):
        return source_path.with_name(source_path.name[: -len(".txt")] + ENRICHED_SUFFIX)
    return source_path.with_name(f"{source_path.stem}{ENRICHED_SUFFIX}")


def validated_speakers_path(video_path):
    return existing_speakers_dir(video_path) / SPEAKERS_VALIDATED_NAME


def load_validated_speakers(video_path):
    path = validated_speakers_path(video_path)
    if not path.exists():
        return [], path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] speakers valides illisibles pour {Path(video_path).stem}: {exc}")
        return [], path
    speakers = [
        " ".join(str(name).split()).strip()
        for name in payload.get("speakers", []) or []
        if str(name).strip()
    ]
    return speakers, path


def replace_speaker_labels(text, speakers):
    replacements = 0

    def replacement(match):
        nonlocal replacements
        index = int(match.group(1))
        if index >= len(speakers):
            return match.group(0)
        replacements += 1
        return speakers[index]

    return SPEAKER_LABEL_PATTERN.sub(replacement, str(text)), replacements


def parse_timecoded_source(path, speakers=None):
    items = []
    speakers = speakers or []
    for line in path.read_text(encoding="utf-8").splitlines():
        line, replacement_count = replace_speaker_labels(line, speakers)
        transcript_match = TRANSCRIPT_LINE.match(line)
        if transcript_match:
            start, _, _ = transcript_match.groups()
            items.append(
                {
                    "second": parse_timecode(start),
                    "line": line,
                    "speaker_label_replacement_count": replacement_count,
                }
            )
            continue
        subtitle_match = SUBTITLE_LINE.match(line)
        if subtitle_match:
            second, _ = subtitle_match.groups()
            items.append(
                {
                    "second": parse_timecode(second),
                    "line": line,
                    "speaker_label_replacement_count": replacement_count,
                }
            )
            continue
        items.append(
            {
                "second": 10**12,
                "line": line,
                "speaker_label_replacement_count": replacement_count,
            }
        )
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
    if kind == "graphic":
        return "graphic"
    return "on_footage"


def format_overlay_line(overlay):
    timecode = format_timecode(overlay["second"])
    labels = {
        "on_footage": "ANIMATIONS",
        "graphic": "INTERCALAIRE",
    }
    label = labels[overlay_label_key(overlay)]
    return f"[{timecode}] {label}: {overlay['text']}"


def format_plain_overlay_line(overlay):
    timecode = format_timecode(overlay["second"])
    return f"[{timecode}] {overlay['text']}"


def grouped_lines(blocks):
    lines = []
    blocks_by_group = {group: [] for group in ENRICHED_GROUP_ORDER}
    for block in blocks:
        line = str(block.get("line", "")).strip()
        if not line:
            continue
        group = block.get("group")
        blocks_by_group.setdefault(group, []).append(line)

    for group in ENRICHED_GROUP_ORDER:
        group_lines = blocks_by_group.get(group, [])
        if not group_lines:
            continue
        if lines:
            lines.append("")
        lines.extend(group_lines)

    for group, group_lines in blocks_by_group.items():
        if group in ENRICHED_GROUP_ORDER or not group_lines:
            continue
        if lines:
            lines.append("")
        lines.extend(group_lines)
    return lines


def is_empty_text_file(path):
    if not path.exists():
        return False
    try:
        return not path.read_text(encoding="utf-8").strip()
    except Exception:
        return False


def enrich_transcript(video_path, force=False):
    source = timecodes_path(video_path)
    analyse = enriched_ocr_source_path(video_path)
    target = enriched_path(source)
    speakers, speakers_source = load_validated_speakers(video_path)

    if not source.exists():
        print(f"[skip] timecodes corrige introuvable: {source}")
        return None
    if not analyse.exists():
        print(f"[skip] analyse filtree introuvable: {analyse}")
        return None
    dependencies = [source, analyse]
    if speakers_source.exists():
        dependencies.append(speakers_source)
    if target.exists() and not force and target.stat().st_mtime >= max(
        dependency.stat().st_mtime for dependency in dependencies
    ):
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: transcript corrige ou OCR plus recent")

    video_type = (analysed_video_type(video_path) or "").strip().lower()
    plain_motion_design_overlays = (
        video_type == "motion_design" and is_empty_text_file(whisper_timecoded_path(video_path))
    )
    source_lines = parse_timecoded_source(source, speakers)
    speaker_label_replacement_count = sum(
        item["speaker_label_replacement_count"]
        for item in source_lines
    )
    overlays = load_filtered_overlays(analyse)
    source_index = 0
    overlay_index = 0
    overlay_formatter = format_plain_overlay_line if plain_motion_design_overlays else format_overlay_line
    blocks = []

    while source_index < len(source_lines):
        current_second = source_lines[source_index]["second"]
        while overlay_index < len(overlays) and overlays[overlay_index]["second"] <= current_second:
            overlay = overlays[overlay_index]
            blocks.append(
                {
                    "group": (
                        "intercalaire"
                        if overlay_label_key(overlay) == "graphic"
                        else "animations"
                    ),
                    "line": overlay_formatter(overlay),
                }
            )
            overlay_index += 1

        blocks.append({"group": "speaker", "line": source_lines[source_index]["line"]})
        source_index += 1

    while overlay_index < len(overlays):
        overlay = overlays[overlay_index]
        blocks.append(
            {
                "group": (
                    "intercalaire"
                    if overlay_label_key(overlay) == "graphic"
                    else "animations"
                ),
                "line": overlay_formatter(overlay),
            }
        )
        overlay_index += 1

    lines = grouped_lines(blocks)
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
            "speakers_source": (
                relative_to_video_dir(speakers_source, video_path)
                if speakers_source.exists()
                else None
            ),
            "speaker_label_replacement_count": speaker_label_replacement_count,
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
    parser.add_argument(
        "--has-subtitles",
        choices=("true", "false", "all"),
        default="all",
        help="Filtre optionnel sur pipeline_analysis.has_subtitles. Defaut: all.",
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
    expected_has_subtitles = None if args.has_subtitles == "all" else args.has_subtitles == "true"
    for video_path in videos:
        has_subtitles = analysed_has_subtitles(video_path)
        if expected_has_subtitles is not None and has_subtitles is not expected_has_subtitles:
            print(
                f"[skip] {video_path.name}: pipeline_analysis.has_subtitles n'est pas {str(expected_has_subtitles).lower()}"
            )
            continue
        if enrich_transcript(video_path, force=args.force):
            done += 1

    print(f"{done} fichiers enrichis.")


if __name__ == "__main__":
    main()
