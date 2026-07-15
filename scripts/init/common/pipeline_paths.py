import os
import re
import shutil
from pathlib import Path


INIT_DIR_NAME = "init"
LEGACY_INIT_DIR_PATTERN = re.compile(r"^\d{8}_\d{4}_init(?:_\d+)?$")
OUTPUTS_DIR_NAME = "outputs"
METADATA_DIR_NAME = "metadata"
IMAGES_DIR_NAME = "images"
INTERVIEW_DIR_NAME = "interview"
OCR_DIR_NAME = "ocr"
OCR_RAW_DIR_NAME = "raw"
TRANSCRIPTS_DIR_NAME = "transcripts"
CHUNKS_DIR_NAME = "chunks"
ANALYSED_INFOS_NAME = "pipeline_analysis.json"
YOUTUBE_API_INFOS_NAME = "youtube_video_metadata.json"
LEGACY_ANALYSED_INFOS_NAME = "analysed_infos.json"
LEGACY_YOUTUBE_API_INFOS_NAME = "youtube_api_infos.json"
LEGACY_YOUTUBE_API_INFOS_SUFFIX = ".youtube_api_infos.json"
LEGACY_INFO_SUFFIX = ".info.json"


def init_dir(parent_dir):
    return Path(parent_dir) / INIT_DIR_NAME


def legacy_init_dirs(parent_dir):
    parent = Path(parent_dir)
    if not parent.is_dir():
        return []
    return sorted(
        path
        for path in parent.iterdir()
        if path.is_dir() and LEGACY_INIT_DIR_PATTERN.fullmatch(path.name)
    )


def _merge_missing(source_dir, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        target = target_dir / source.name
        if not target.exists():
            shutil.move(str(source), str(target))
        elif source.is_dir() and target.is_dir():
            _merge_missing(source, target)


def consolidate_init_dir(parent_dir):
    parent = Path(parent_dir)
    parent.mkdir(parents=True, exist_ok=True)
    target = init_dir(parent)
    legacy_dirs = legacy_init_dirs(parent)

    if not target.exists() and len(legacy_dirs) == 1:
        legacy_dirs[0].rename(target)
        return target

    target.mkdir(parents=True, exist_ok=True)
    # Le dossier fixe existant reste prioritaire. Sinon, le run date le plus
    # recent est fusionne en premier et les plus anciens completent les trous.
    for legacy_dir in reversed(legacy_dirs):
        _merge_missing(legacy_dir, target)
        shutil.rmtree(legacy_dir)
    return target


def video_base_dir(video_path):
    path = Path(video_path)
    return path if path.is_dir() else path.parent


def video_id(video_path):
    path = Path(video_path)
    return path.name if path.is_dir() else path.stem


def outputs_dir(video_path):
    return video_base_dir(video_path) / OUTPUTS_DIR_NAME


def metadata_dir(video_path):
    return video_base_dir(video_path) / METADATA_DIR_NAME


def images_dir(video_path):
    return outputs_dir(video_path) / IMAGES_DIR_NAME


def interview_dir(video_path):
    return outputs_dir(video_path) / INTERVIEW_DIR_NAME


def ocr_dir(video_path):
    return outputs_dir(video_path) / OCR_DIR_NAME


def ocr_raw_dir(video_path):
    return ocr_dir(video_path) / OCR_RAW_DIR_NAME


def transcripts_dir(video_path):
    return outputs_dir(video_path) / os.environ.get("PIPELINE_TRANSCRIPTS_DIR_NAME", TRANSCRIPTS_DIR_NAME)


def chunks_dir(video_path):
    return outputs_dir(video_path) / CHUNKS_DIR_NAME


def _existing(preferred, legacy):
    return legacy if legacy.exists() and not preferred.exists() else preferred


def existing_images_dir(video_path):
    return _existing(images_dir(video_path), video_base_dir(video_path) / IMAGES_DIR_NAME)


def existing_interview_dir(video_path):
    return _existing(interview_dir(video_path), video_base_dir(video_path) / "is_interview")


def existing_ocr_dir(video_path):
    return _existing(ocr_dir(video_path), video_base_dir(video_path) / OCR_DIR_NAME)


def existing_ocr_raw_dir(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / OCR_RAW_DIR_NAME
    if preferred.exists():
        return preferred
    return video_ocr_dir


def existing_transcripts_dir(video_path):
    base_dir = video_base_dir(video_path)
    preferred = transcripts_dir(video_path)
    outputs_root = outputs_dir(video_path)
    output_candidates = (
        outputs_root / "transcripts_whisper",
        outputs_root / "transcripts_ocr",
    )
    for candidate in output_candidates:
        if candidate.exists() and not preferred.exists():
            return candidate
    for legacy_name in ("transcript", "transcript_whisper", "transcript_ocr", "transcripts", "transcripts_whisper", "transcripts_ocr"):
        legacy = base_dir / legacy_name
        if legacy.exists() and not preferred.exists():
            return legacy
    return preferred


def existing_chunks_dir(video_path):
    return _existing(chunks_dir(video_path), video_base_dir(video_path) / CHUNKS_DIR_NAME)


def analysed_infos_path(video_path):
    preferred = metadata_dir(video_path) / ANALYSED_INFOS_NAME
    base_dir = video_base_dir(video_path)
    legacy_candidates = (
        metadata_dir(video_path) / LEGACY_ANALYSED_INFOS_NAME,
        base_dir / f"{video_id(video_path)}_analysed_infos.json",
        base_dir / LEGACY_ANALYSED_INFOS_NAME,
    )
    for legacy in legacy_candidates:
        if legacy.exists() and not preferred.exists():
            return legacy
    return preferred


def youtube_api_infos_path(video_path):
    return metadata_dir(video_path) / YOUTUBE_API_INFOS_NAME


def existing_youtube_api_infos_path(video_path):
    preferred = youtube_api_infos_path(video_path)
    base_dir = video_base_dir(video_path)
    legacy_candidates = (
        metadata_dir(video_path) / LEGACY_YOUTUBE_API_INFOS_NAME,
        base_dir / f"{video_id(video_path)}{LEGACY_YOUTUBE_API_INFOS_SUFFIX}",
        base_dir / f"{video_id(video_path)}{LEGACY_INFO_SUFFIX}",
    )
    for legacy in legacy_candidates:
        if legacy.exists() and not preferred.exists():
            return legacy
    return preferred


def relative_to_video_dir(path, video_path):
    return Path(path).relative_to(video_base_dir(video_path)).as_posix()
