import argparse
import json
import re
import sys
from pathlib import Path

from common.pipeline_analysis import update_analysed_infos
from common.pipeline_paths import (
    OUTPUTS_DIR_NAME,
    TRANSCRIPTS_DIR_NAME,
    existing_transcripts_dir,
    existing_youtube_api_infos_path,
    relative_to_video_dir,
)

DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
ENRICHED_SUFFIX = "_enriched.txt"
LEGACY_ENRICHED_SUFFIX = "_enrichi.txt"
SUMMARY_NAME = "video_summary.md"
TIMECODE_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
TIMECODE_WITH_RANGE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})-((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
INSERT_LINE = re.compile(r"^INSERT:\s*(.*)$", re.IGNORECASE)
GRAPHIC_LINE = re.compile(r"^GRAPHIC:\s*(.*)$", re.IGNORECASE)
ON_FOOTAGE_LINE = re.compile(r"^ON_FOOTAGE:\s*(.*)$", re.IGNORECASE)
YOUTUBE_API_INFOS_SUFFIX = ".youtube_api_infos.json"
LEGACY_INFO_SUFFIX = ".info.json"

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


def enriched_inputs(transcript_dir):
    return sorted(transcript_dir.glob(f"*{ENRICHED_SUFFIX}")) or sorted(
        transcript_dir.glob(f"*{LEGACY_ENRICHED_SUFFIX}")
    )


def video_dir_for_input(input_path):
    if (
        input_path.parent.name.startswith(TRANSCRIPTS_DIR_NAME)
        and input_path.parent.parent.name == OUTPUTS_DIR_NAME
    ):
        return input_path.parent.parent.parent
    return input_path.parent.parent


def video_file_for_dir(video_dir):
    matches = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return matches[0] if matches else video_dir


def summary_path(input_path):
    return input_path.with_name(SUMMARY_NAME)


def repair_mojibake(text):
    value = str(text)
    markers = ("Ã", "â", "ðŸ", "ï¸", "œ", "�")
    if not any(marker in value for marker in markers):
        return value
    for source_encoding in ("latin-1", "cp1252"):
        try:
            repaired = value.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired != value:
            return repaired
    return value


def normalize_text(text):
    return " ".join(repair_mojibake(text).split()).strip()


def video_title_for_input(input_path):
    video_dir = video_dir_for_input(input_path)
    candidates = [
        existing_youtube_api_infos_path(video_dir),
        video_dir / f"{video_dir.name}{YOUTUBE_API_INFOS_SUFFIX}",
        input_path.parent / f"{video_dir.name}{YOUTUBE_API_INFOS_SUFFIX}",
        video_dir / f"{video_dir.name}{LEGACY_INFO_SUFFIX}",
        input_path.parent / f"{video_dir.name}{LEGACY_INFO_SUFFIX}",
    ]
    for info_path in candidates:
        if not info_path.exists():
            continue
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = payload.get("title")
        if title:
            return normalize_text(title)
    return normalize_text(input_path.stem)


def parse_enriched_lines(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = TIMECODE_WITH_RANGE.match(line)
        if match:
            start, end, body = match.groups()
            rows.append((start, end, body.strip()))
            continue
        match = TIMECODE_LINE.match(line)
        if match:
            timecode, body = match.groups()
            rows.append((timecode, timecode, body.strip()))
    return rows


def escape_markdown(text):
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def normalize_chapter_title(text):
    normalized = normalize_text(text).replace("/", " ").strip()
    if normalized == "Suivez notre actualité sur":
        return "Suivez notre actualité sur les réseaux sociaux"
    if normalized.endswith("Suivez notre actualité sur"):
        prefix = normalized[: -len("Suivez notre actualité sur")].rstrip()
        if prefix:
            return f"{prefix} Suivez notre actualité sur les réseaux sociaux"
        return "Suivez notre actualité sur les réseaux sociaux"
    return normalized


def split_chapters(rows):
    chapters = []
    current = None
    first_chapter_has_insert = False
    last_body_was_graphic = False

    for start, end, body in rows:
        insert_match = INSERT_LINE.match(body)
        if insert_match:
            if current:
                chapters.append(current)
            if not chapters:
                first_chapter_has_insert = True
            current = {
                "title": normalize_chapter_title(insert_match.group(1).strip()) or "Sans titre",
                "start": start,
                "end": end,
                "speech": [],
                "animations": [],
                "is_intro": False,
                "is_outro": False,
            }
            last_body_was_graphic = False
            continue

        if current is None:
            current = {
                "title": "Sans titre",
                "start": start,
                "end": end,
                "speech": [],
                "animations": [],
                "is_intro": False,
                "is_outro": False,
            }

        current["end"] = end

        graphic_match = GRAPHIC_LINE.match(body)
        if graphic_match:
            graphic_text = normalize_text(graphic_match.group(1))
            if current:
                chapters.append(current)
            if not chapters:
                first_chapter_has_insert = True
            current = {
                "title": normalize_chapter_title(graphic_text) or "Sans titre",
                "start": start,
                "end": end,
                "speech": [],
                "animations": [],
                "is_intro": False,
                "is_outro": False,
            }
            last_body_was_graphic = True
            continue

        on_footage_match = ON_FOOTAGE_LINE.match(body)
        if on_footage_match:
            current["animations"].append(normalize_text(on_footage_match.group(1)))
            last_body_was_graphic = False
            continue

        if body:
            current["speech"].append(normalize_text(body))
            last_body_was_graphic = False

    if current:
        if last_body_was_graphic:
            current["is_outro"] = True
        chapters.append(current)

    if chapters and not first_chapter_has_insert:
        chapters[0]["title"] = "Début de la vidéo"
        chapters[0]["is_intro"] = True

    return chapters


def render_table(chapter_index, chapter):
    if chapter.get("is_intro"):
        heading = "## Début de la vidéo"
    elif chapter.get("is_outro"):
        heading = f"## Outro : *{escape_markdown(chapter['title'])}*"
    else:
        heading = f"## Intercalaire {chapter_index:02d} : *{escape_markdown(chapter['title'])}*"
    if chapter.get("is_outro"):
        return [heading, ""]

    lines = [heading, ""]
    lines.extend(
        [
            "| Timecode | Parole | Animation |",
            "| --- | --- | --- |",
        ]
    )
    timecode = chapter["start"] if chapter["start"] == chapter["end"] else f"{chapter['start']} - {chapter['end']}"
    speech = " ".join(chapter["speech"]) if chapter["speech"] else ""
    animation = " / ".join(chapter["animations"]) if chapter["animations"] else ""
    lines.append(f"| {escape_markdown(timecode)} | {escape_markdown(speech)} | {escape_markdown(animation)} |")
    lines.append("")
    return lines


def render_summary(rows, video_title):
    lines = ["# Sommaire de la vidéo", f"# {escape_markdown(video_title)}", ""]
    chapters = split_chapters(rows)
    if not chapters:
        lines.extend(
            [
                "## Début de la vidéo",
                "",
                "| Timecode | Parole | Animation |",
                "| --- | --- | --- |",
                "|  |  |  |",
            ]
        )
        return "\n".join(lines) + "\n"

    for index, chapter in enumerate(chapters, start=1):
        lines.extend(render_table(index, chapter))
    return "\n".join(lines) + "\n"


def summarize_file(input_path, force=False):
    target = summary_path(input_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target

    rows = parse_enriched_lines(input_path.read_text(encoding="utf-8"))
    target.write_text(render_summary(rows, video_title_for_input(input_path)), encoding="utf-8")
    video_dir = video_dir_for_input(input_path)
    video_path = video_file_for_dir(video_dir)
    update_analysed_infos(
        video_path,
        "video_summary",
        {
            "status": "done",
            "source": relative_to_video_dir(input_path, video_path),
            "summary_file": relative_to_video_dir(target, video_path),
            "row_count": len(rows),
        },
    )
    print(f"[ok] {target}")
    return target


def parse_args():
    parser = argparse.ArgumentParser(description="Cree un resume Markdown a partir des fichiers *_enriched.txt.")
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
        help="Regenere les fichiers video_summary meme s'ils existent deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    transcript_dirs = sorted({existing_transcripts_dir(video_path) for video_path in video_files(video_dir)})
    inputs = []
    for transcript_dir in transcript_dirs:
        inputs.extend(enriched_inputs(transcript_dir))

    if not inputs:
        print(f"Aucun fichier *{ENRICHED_SUFFIX} trouve dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for input_path in inputs:
        if summarize_file(input_path, force=args.force):
            done += 1

    print(f"{done} fichiers résumés.")


if __name__ == "__main__":
    main()
