from __future__ import annotations

import shutil
from pathlib import Path

from pipeline.steps.speakers.correct_speaker_transcripts import (
    correct_speaker_files,
)
from pipeline.steps.transcripts.artifacts import (
    LEGACY_TRANSCRIPT_2_NAMES,
    TRANSCRIPT_2_CORRECTED_NAME,
    TRANSCRIPT_3_WITH_SPEAKERS_NAME,
)
from pipeline.support.paths import (
    CANONICAL_TRANSCRIPTS_DIR_NAME,
    existing_transcripts_dir,
    output_is_current,
)


def corrected_source_path(video_path: Path) -> Path:
    directory = existing_transcripts_dir(
        video_path,
        name=CANONICAL_TRANSCRIPTS_DIR_NAME,
    )
    preferred = directory / TRANSCRIPT_2_CORRECTED_NAME
    if preferred.exists():
        return preferred
    return next(
        (
            directory / name
            for name in LEGACY_TRANSCRIPT_2_NAMES
            if (directory / name).exists()
        ),
        preferred,
    )


def with_speakers_path(video_path: Path) -> Path:
    return (
        existing_transcripts_dir(
            video_path,
            name=CANONICAL_TRANSCRIPTS_DIR_NAME,
        )
        / TRANSCRIPT_3_WITH_SPEAKERS_NAME
    )


def apply_speakers(video_path: Path, *, force: bool = False) -> Path | None:
    source = corrected_source_path(video_path)
    target = with_speakers_path(video_path)
    if not source.exists():
        print(f"[skip] transcript corrige introuvable: {source}")
        return None
    if target.exists() and not force and output_is_current(target, [source]):
        correction_result = correct_speaker_files(
            video_path,
            [target],
            force=False,
            replace_speaker_labels=True,
        )
        if correction_result is not None:
            return target

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    correct_speaker_files(
        video_path,
        [target],
        force=force,
        replace_speaker_labels=True,
    )
    print(f"[ok] {target}")
    return target
