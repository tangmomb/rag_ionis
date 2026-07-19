import json
import re
import shutil

from pipeline.support.json_io import read_json
from pipeline.support.paths import (
    CANONICAL_TRANSCRIPTS_DIR_NAME,
    OCR_CORRECTION_TRANSCRIPTS_DIR_NAME,
    existing_speakers_dir,
    existing_transcripts_dir,
    transcripts_dir,
)
from pipeline.steps.transcripts.artifacts import (
    LEGACY_TRANSCRIPT_2_NAMES,
    LEGACY_TRANSCRIPT_ENRICHED_NAMES,
    TRANSCRIPT_2_CORRECTED_NAME,
    TRANSCRIPT_3_WITH_SPEAKERS_NAME,
    TRANSCRIPT_ENRICHED_NAME,
    TRANSCRIPT_PLAIN_NAME,
)

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
    r"^\[(?:(?:\d{2}:)?\d{2}:\d{2}-(?:\d{2}:)?\d{2}:\d{2}|(?:\d{2}:)?\d{2}:\d{2})\]\s*"
)
SYSTEM_SPEAKER_PREFIX = re.compile(r"^SPEAKER[_ -]?\d+\s*:\s*", re.IGNORECASE)
INTERCALAIRE_PREFIX = re.compile(r"^INTERCALAIRE\s*:", re.IGNORECASE)
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"

def strip_timecodes(text, speaker_names=None):
    cleaned_lines = []
    speaker_names = speaker_names or []
    speaker_patterns = [
        re.compile(rf"^{re.escape(name)}\s*:\s*", re.IGNORECASE)
        for name in sorted(set(speaker_names), key=len, reverse=True)
        if str(name).strip()
    ]
    for line in text.splitlines():
        cleaned = TIMECODE_PREFIX.sub("", line).strip()
        cleaned = SYSTEM_SPEAKER_PREFIX.sub("", cleaned, count=1).strip()
        for pattern in speaker_patterns:
            cleaned, count = pattern.subn("", cleaned, count=1)
            if count:
                cleaned = cleaned.strip()
                break
        if cleaned and not INTERCALAIRE_PREFIX.match(cleaned):
            cleaned_lines.append(cleaned)
    return " ".join(cleaned_lines)


def load_validated_speakers(video_path):
    path = existing_speakers_dir(video_path) / SPEAKERS_VALIDATED_NAME
    if not path.exists():
        return []
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return []
    return [
        " ".join(str(name).split()).strip()
        for name in payload.get("speakers", []) or []
        if str(name).strip()
    ]


def output_path(input_path):
    if (
        input_path.parent.name == CANONICAL_TRANSCRIPTS_DIR_NAME
        or input_path.name
        in {
            TRANSCRIPT_ENRICHED_NAME,
            TRANSCRIPT_3_WITH_SPEAKERS_NAME,
            TRANSCRIPT_2_CORRECTED_NAME,
            *LEGACY_TRANSCRIPT_2_NAMES,
            *LEGACY_TRANSCRIPT_ENRICHED_NAMES,
        }
    ):
        return input_path.with_name(TRANSCRIPT_PLAIN_NAME)
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


def ocr_plain_output_path(video_path):
    return (
        transcripts_dir(
            video_path,
            name=OCR_CORRECTION_TRANSCRIPTS_DIR_NAME,
        )
        / PLAIN_NAME
    )


def remove_obsolete_ocr_transcripts(target):
    directory = target.parent
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child == target:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def convert_ocr_correction_file(
    video_path,
    input_path,
    force=False,
):
    target = ocr_plain_output_path(video_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    remove_obsolete_ocr_transcripts(target)

    if (
        target.exists()
        and not force
        and target.stat().st_mtime >= input_path.stat().st_mtime
    ):
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: OCR interne plus recent")

    cleaned = strip_timecodes(input_path.read_text(encoding="utf-8"))
    target.write_text(cleaned + "\n", encoding="utf-8")
    print(f"[ok] {target}")
    return target


def timecoded_inputs(transcript_dir):
    if transcript_dir.name == CANONICAL_TRANSCRIPTS_DIR_NAME:
        canonical_names = (
            TRANSCRIPT_3_WITH_SPEAKERS_NAME,
            TRANSCRIPT_2_CORRECTED_NAME,
            *LEGACY_TRANSCRIPT_2_NAMES,
            TRANSCRIPT_ENRICHED_NAME,
            *LEGACY_TRANSCRIPT_ENRICHED_NAMES,
        )
        for name in canonical_names:
            candidate = transcript_dir / name
            if candidate.exists():
                return [candidate]

    inputs = [
        transcript_dir / name
        for name in CORRECTED_TIMECODED_NAMES
        if (transcript_dir / name).exists()
    ]
    if inputs:
        return inputs

    legacy_inputs = []
    for suffix in LEGACY_CORRECTED_TIMECODED_SUFFIXES:
        legacy_inputs.extend(sorted(transcript_dir.glob(f"*{suffix}")))
    return legacy_inputs


def convert_file(
    video_path,
    input_path,
    force=False,
    *,
    transcripts_dir_name=None,
):
    target = output_path(input_path)

    source_path = input_path
    if target.exists() and not force and target.stat().st_mtime >= source_path.stat().st_mtime:
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: transcript timecode plus recent")

    text = source_path.read_text(encoding="utf-8")
    cleaned = strip_timecodes(text, load_validated_speakers(video_path))
    target.write_text(cleaned + "\n", encoding="utf-8")
    print(f"[ok] {target}")
    return target
