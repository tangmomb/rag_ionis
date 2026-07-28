import re

from pipeline.steps.speakers.correct_speaker_transcripts import (
    candidates_path,
    correct_speaker_files,
    validated_path,
)
from pipeline.support.json_io import read_json
from pipeline.support.ocr_filtering import (
    enriched_ocr_source_path,
    format_timecode,
    processed_ocr_source_path,
)
from pipeline.support.paths import (
    CANONICAL_TRANSCRIPTS_DIR_NAME,
    existing_transcripts_dir,
)
from pipeline.steps.transcripts.artifacts import (
    LEGACY_TRANSCRIPT_2_NAMES,
    LEGACY_TRANSCRIPT_3_WITH_SPEAKERS_NAMES,
    LEGACY_TRANSCRIPT_ENRICHED_NAMES,
    TRANSCRIPT_2_CORRECTED_NAME,
    TRANSCRIPT_3_ENRICHED_NAME,
)


CORRECTED_SUFFIX = "_corrected.txt"
ENRICHED_SUFFIX = "_enriched.txt"
LEGACY_ENRICHED_SUFFIX = "_enrichi.txt"
TRANSCRIPT_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})-((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
SUBTITLE_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")

def parse_timecode(value):
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def corrected_timecodes_path(video_path, *, transcripts_dir_name=None):
    transcript_dir = existing_transcripts_dir(
        video_path,
        name=transcripts_dir_name,
    )
    candidates = (
        transcript_dir / TRANSCRIPT_2_CORRECTED_NAME,
        *(transcript_dir / name for name in LEGACY_TRANSCRIPT_2_NAMES),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Aucun transcript corrige trouve pour l'enrichissement de "
        f"{video_path.stem} dans {transcript_dir}"
    )


def enriched_path(source_path):
    if source_path.parent.name == CANONICAL_TRANSCRIPTS_DIR_NAME:
        return source_path.with_name(TRANSCRIPT_3_ENRICHED_NAME)
    if source_path.name.endswith(CORRECTED_SUFFIX):
        return source_path.with_name(source_path.name[: -len(".txt")] + ENRICHED_SUFFIX)
    return source_path.with_name(f"{source_path.stem}{ENRICHED_SUFFIX}")


def speaker_dependencies(video_path):
    validated = validated_path(video_path)
    if not validated.exists():
        return []
    dependencies = [validated]
    validated_payload = read_json(validated)
    candidates = candidates_path(video_path, validated_payload)
    if candidates.exists():
        dependencies.append(candidates)
    ocr_source = processed_ocr_source_path(video_path)
    if ocr_source.exists():
        dependencies.append(ocr_source)
    return dependencies


def remove_obsolete_transcript_files(target):
    for name in (
        *LEGACY_TRANSCRIPT_3_WITH_SPEAKERS_NAMES,
        *LEGACY_TRANSCRIPT_ENRICHED_NAMES,
    ):
        obsolete = target.with_name(name)
        if obsolete != target and obsolete.exists():
            obsolete.unlink()


def parse_timecoded_source(path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        transcript_match = TRANSCRIPT_LINE.match(line)
        if transcript_match:
            start, _, _ = transcript_match.groups()
            items.append(
                {
                    "second": parse_timecode(start),
                    "line": line,
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
                }
            )
            continue
        items.append(
            {
                "second": 10**12,
                "line": line,
            }
        )
    return items


def load_intercalaires(path):
    payload = read_json(path)
    graphics = payload.get("kinds", {}).get("graphic", {})
    overlays = []
    if not isinstance(graphics, dict):
        return overlays
    for timecode, text in graphics.items():
        cleaned = " ".join(str(text).split())
        if not cleaned:
            continue
        overlays.append(
            {
                "second": parse_timecode(timecode),
                "text": cleaned,
            }
        )
    return sorted(overlays, key=lambda item: item["second"])


def format_intercalaire_line(overlay):
    timecode = format_timecode(overlay["second"])
    return f"[{timecode}] INTERCALAIRE: {overlay['text']}"


def enrich_transcript(
    video_path,
    force=False,
    *,
    transcripts_dir_name=None,
):
    source = corrected_timecodes_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    analyse = enriched_ocr_source_path(video_path)
    target = enriched_path(source)

    if not source.exists():
        print(f"[skip] transcript corrige introuvable: {source}")
        return None
    dependencies = [source]
    if analyse.exists():
        dependencies.append(analyse)
    dependencies.extend(speaker_dependencies(video_path))
    if target.exists() and not force and target.stat().st_mtime >= max(
        dependency.stat().st_mtime for dependency in dependencies
    ):
        remove_obsolete_transcript_files(target)
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: transcript corrige ou OCR plus recent")

    source_lines = [
        item["line"]
        for item in parse_timecoded_source(source)
        if item["line"].strip()
    ]
    intercalaires = load_intercalaires(analyse) if analyse.exists() else []
    lines = list(source_lines)
    if intercalaires:
        if lines:
            lines.append("")
        lines.extend(format_intercalaire_line(item) for item in intercalaires)

    target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    speaker_result = correct_speaker_files(
        video_path,
        [target],
        force=force,
        replace_speaker_labels=True,
    )
    if speaker_result is None:
        target.unlink(missing_ok=True)
        print("[skip] application des speakers impossible")
        return None
    remove_obsolete_transcript_files(target)
    print(f"[ok] {target}")
    return target
