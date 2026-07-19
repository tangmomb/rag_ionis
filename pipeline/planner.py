from __future__ import annotations

from .context import PipelineContext
from .contracts import PlannedTask


INSPECTION_TASKS = (
    "frames.extract",
    "frames.classify",
    "video.detect_interview",
    "video.infer_type",
    "ocr.extract_raw",
    "ocr.extract_boxes",
    "video.detect_subtitles",
)

COMMON_PROCESSING_TASKS = (
    "ocr.build_processed",
    "ocr.filter_overlays",
    "ocr.extract_review_candidates",
    "ocr.review_other_text",
    "ocr.apply_review",
)

CANONICAL_AFTER_CORRECTION_TASKS = (
    "speakers.propose",
    "speakers.validate",
)

CANONICAL_FINALIZATION_TASKS = (
    "transcript.apply_speakers",
    "transcript.enrich",
    "transcript.create_plain",
)

OCR_TRANSCRIPT_PREPARATION_TASKS = (
    "transcript.extract_ocr",
    "transcript.normalize_brand",
    "transcript.create_plain_ocr",
)


def inspection_plan() -> list[PlannedTask]:
    return [PlannedTask(task_id, "inspection_required") for task_id in INSPECTION_TASKS]


def processing_plan(context: PipelineContext) -> list[PlannedTask]:
    if not context.routing_ready:
        raise RuntimeError(
            "Le plan de traitement exige la detection has_subtitles et video_type. "
            "Execute d'abord `python -m pipeline inspect ...`."
        )
    tasks = [
        PlannedTask(task_id, "all_videos")
        for task_id in COMMON_PROCESSING_TASKS
    ]
    tasks.append(
        PlannedTask("transcript.whisper", "canonical_transcript=whisperx")
    )
    if context.has_subtitles is True:
        tasks.extend(
            PlannedTask(task_id, "ocr_transcript_for_reconciliation")
            for task_id in OCR_TRANSCRIPT_PREPARATION_TASKS
        )
    correction_task = (
        "transcript.reconcile_ocr"
        if context.has_subtitles is True
        else "transcript.correct_whisper"
    )
    correction_reason = (
        "reconcile_whisperx_with_ocr"
        if context.has_subtitles is True
        else "correct_whisperx_with_visual_ocr"
    )
    tasks.append(PlannedTask(correction_task, correction_reason))
    tasks.extend(
        PlannedTask(task_id, "canonical_transcript=whisperx_corrected")
        for task_id in CANONICAL_AFTER_CORRECTION_TASKS
    )
    tasks.extend(
        PlannedTask(task_id, "canonical_transcript=whisperx_corrected")
        for task_id in CANONICAL_FINALIZATION_TASKS
    )

    tasks.append(
        PlannedTask("chunks.create", f"duration_{context.chunk_strategy}")
    )
    if context.chunk_strategy == "long":
        tasks.extend(
            [
                PlannedTask(
                    "chunks.summarize_sections",
                    "duration_seconds>600",
                ),
                PlannedTask(
                    "chunks.summarize_video",
                    "duration_seconds>600",
                ),
            ]
        )
    tasks.append(PlannedTask("embeddings.create", "chunks_ready"))
    return tasks
