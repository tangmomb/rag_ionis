import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
ENRICHED_SUFFIX = "_transcript_timecodes_enrichi.txt"
MIN_OVERLAY_SCORE = 0.9
OVERLAY_KINDS = {"name", "lower_third", "question_intertitle", "title"}
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


def processed_ocr_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}_ocr_processed.json"


def timecodes_path(video_path):
    transcript_dir = video_path.parent / "transcript"
    subtitle_candidates = sorted(transcript_dir.glob(f"{video_path.stem}_ocr_subtitle_timecodes*.txt"))
    if subtitle_candidates:
        return subtitle_candidates[0]

    transcript_candidates = sorted(
        path
        for path in transcript_dir.glob(f"{video_path.stem}*timecodes*.txt")
        if "ocr_subtitle_timecodes" not in path.name and not path.name.endswith(ENRICHED_SUFFIX)
    )
    if transcript_candidates:
        return transcript_candidates[0]

    raise FileNotFoundError(f"Aucun fichier timecodes trouve pour {video_path.stem} dans {transcript_dir}")


def enriched_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{ENRICHED_SUFFIX}"


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


def normalize_text(text):
    return re.sub(r"\W+", "", str(text).casefold())


def merge_question_parts(items):
    merged = []
    for item in items:
        if (
            item.get("kind") == "question_intertitle"
            and merged
            and merged[-1].get("kind") != "question_intertitle"
            and item["second"] - merged[-1]["second"] <= 1
        ):
            previous = merged.pop()
            item = dict(item)
            item["second"] = previous["second"]
            item["text"] = f"{previous['text']} {item['text']}"
        merged.append(item)
    return merged


def remove_overlay_fragments(items):
    kept = []
    for index, item in enumerate(items):
        normalized = normalize_text(item["text"])
        is_fragment = False
        for other in items[index + 1 :]:
            if other["second"] - item["second"] > 4:
                break
            other_normalized = normalize_text(other["text"])
            if len(normalized) < 4 or len(other_normalized) <= len(normalized) + 1:
                continue
            if normalized and normalized in other_normalized:
                is_fragment = True
                break
        if not is_fragment:
            kept.append(item)
    return kept


def confidence_score(item):
    try:
        return float(item.get("score"))
    except (TypeError, ValueError):
        pass

    label = str(item.get("confidence", "")).strip().lower()
    if label == "high":
        return 1.0
    if label == "medium":
        return 0.7
    if label == "low":
        return 0.4
    return 0.0


def parse_processed_non_subtitles(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = []
    seen_normalized = set()
    for item in payload.get("items", []):
        kind = str(item.get("kind", "")).strip().lower()
        if kind == "subtitle" or (kind and kind not in OVERLAY_KINDS):
            continue
        if confidence_score(item) < MIN_OVERLAY_SCORE:
            continue
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        normalized = normalize_text(text)
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        second = item.get("second")
        if second is None:
            timecode = str(item.get("timecode", "")).strip()
            if not timecode:
                continue
            second = parse_timecode(timecode)
        items.append({"kind": kind, "second": int(second), "text": text})
    return remove_overlay_fragments(merge_question_parts(sorted(items, key=lambda item: item["second"])))


def enrich_transcript(video_path, force=False):
    source = timecodes_path(video_path)
    analyse = processed_ocr_path(video_path)
    target = enriched_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] timecodes introuvable: {source}")
        return None
    if not analyse.exists():
        print(f"[skip] analyse introuvable: {analyse}")
        return None

    source_lines = parse_timecoded_source(source)
    overlays = parse_processed_non_subtitles(analyse)
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
    print(f"[ok] {target}")
    return target


def format_overlay_line(overlay):
    timecode = format_timecode(overlay["second"])
    label = "INTERCALAIRE QUESTION" if overlay.get("kind") == "question_intertitle" else "TEXTE ECRIT SUR LA VIDEO"
    return f"[{timecode}] {label}: {overlay['text']}"


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
