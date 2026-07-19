import re

from pipeline.support.paths import (
    existing_ocr_dir,
)


SOURCE_NAME = "ocr_subtitles_timecoded.txt"
LEGACY_SOURCE_SUFFIX = "_ocr_subtitle_timecodes.txt"
OBSOLETE_SPACING_ARTIFACTS = (
    "ocr_subtitles_timecoded_corrected.txt",
    "ocr_spacing_batch_state.json",
    "ocr_spacing_batch_input.jsonl",
    "ocr_spacing_batch_output.jsonl",
    "ocr_spacing_batch_error.jsonl",
)
IONIS_STM_PATTERNS = (
    re.compile(r"(?i)\bl['’]?\s*ionis\s*-\s*stm\b"),
    re.compile(r"(?i)\bl['’]?\s*ionis\s+stm\b"),
    re.compile(r"(?i)\bl['’]?\s*onis\s*-\s*stm\b"),
    re.compile(r"(?i)\bl['’]?\s*onis\s+stm\b"),
    re.compile(r"(?i)\bl\s+ionis\s*-\s*stm\b"),
    re.compile(r"(?i)\bl\s+ionis\s+stm\b"),
    re.compile(r"(?i)\bl\s+onis\s*-\s*stm\b"),
    re.compile(r"(?i)\bl\s+onis\s+stm\b"),
    re.compile(r"(?i)\blonis\s*-\s*stm\b"),
    re.compile(r"(?i)\blonis\s+stm\b"),
    re.compile(r"(?i)\bionis\s*-\s*stm\b"),
    re.compile(r"(?i)\bionis\s+stm\b"),
    re.compile(r"(?i)\bonis\s*-\s*stm\b"),
    re.compile(r"(?i)\bonis\s+stm\b"),
)
TARGET_TEXT = "Ionis-STM"


def transcript_path(video_path):
    transcript_dir = existing_ocr_dir(video_path)
    preferred = transcript_dir / SOURCE_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_SOURCE_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def normalize_ionis_stm(text):
    updated = str(text)
    replacements = 0
    for pattern in IONIS_STM_PATTERNS:
        updated, count = pattern.subn(TARGET_TEXT, updated)
        replacements += count
    return updated, replacements


def process_video(video_path, force=False):
    target = transcript_path(video_path)
    for name in OBSOLETE_SPACING_ARTIFACTS:
        obsolete = existing_ocr_dir(video_path) / name
        if obsolete.exists():
            obsolete.unlink()
    if not target.exists():
        print(f"[skip] transcript OCR timecode introuvable: {target}")
        return False

    original = target.read_text(encoding="utf-8")
    updated, replacements = normalize_ionis_stm(original)
    if replacements == 0 and not force:
        print(f"[skip] {target.name}: aucune variante Ionis-STM detectee")
        return False

    target.write_text(updated, encoding="utf-8")
    print(f"[ok] {target} ({replacements} remplacement(s))")
    return True
