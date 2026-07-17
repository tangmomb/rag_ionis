import json
import re
from pathlib import Path

from pipeline.support.analysis import analysed_infos_path
from pipeline.support.json_io import read_json
from pipeline.support.paths import (
    existing_speakers_dir,
    existing_transcripts_dir,
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
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"
ENRICHED_SUFFIX = "_enriched.txt"
LEGACY_ENRICHED_SUFFIX = "_enrichi.txt"
MOTION_DESIGN_OCR_PREFIX = "Textes présents sur la vidéo :"

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
        if cleaned:
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


def analysed_video_type(video_path):
    path = analysed_infos_path(video_path)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    value = payload.get("video_type")
    return value if isinstance(value, str) else None


def whisper_timecoded_path(video_path, *, transcripts_dir_name=None):
    transcript_dir = existing_transcripts_dir(
        video_path,
        name=transcripts_dir_name,
    )
    return transcript_dir / "whisper_transcript_timecoded.txt"


def is_empty_text_file(path):
    if not path.exists():
        return False
    try:
        return not path.read_text(encoding="utf-8").strip()
    except Exception:
        return False


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


def enriched_input_path(input_path):
    if input_path.name.endswith(".txt"):
        candidate = input_path.with_name(input_path.name[: -len(".txt")] + ENRICHED_SUFFIX)
        if candidate.exists():
            return candidate

    candidate = input_path.with_name(f"{input_path.stem}{LEGACY_ENRICHED_SUFFIX}")
    if candidate.exists():
        return candidate
    return None


def timecoded_inputs(transcript_dir):
    inputs = [transcript_dir / name for name in CORRECTED_TIMECODED_NAMES if (transcript_dir / name).exists()]
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

    video_type = (analysed_video_type(video_path) or "").strip().lower()
    plain_motion_design_overlays = (
        video_type == "motion_design"
        and is_empty_text_file(
            whisper_timecoded_path(
                video_path,
                transcripts_dir_name=transcripts_dir_name,
            )
        )
    )
    source_path = input_path
    if plain_motion_design_overlays:
        enriched_path = enriched_input_path(input_path)
        if enriched_path is not None:
            source_path = enriched_path
    if target.exists() and not force and target.stat().st_mtime >= source_path.stat().st_mtime:
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: transcript timecode plus recent")

    text = source_path.read_text(encoding="utf-8")
    cleaned = strip_timecodes(text, load_validated_speakers(video_path))
    if plain_motion_design_overlays and source_path != input_path:
        cleaned = f"{MOTION_DESIGN_OCR_PREFIX}\n{cleaned}" if cleaned else MOTION_DESIGN_OCR_PREFIX
    target.write_text(cleaned + "\n", encoding="utf-8")
    print(f"[ok] {target}")
    return target
