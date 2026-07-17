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

OCR_TRANSCRIPT_TASKS = (
    "transcript.extract_ocr",
    "transcript.correct_ocr_spacing",
    "transcript.normalize_brand",
    "speakers.propose",
    "speakers.validate",
    "speakers.assign_ocr",
    "transcript.create_plain",
    "transcript.enrich",
)

WHISPER_TRANSCRIPT_TASKS = (
    "transcript.whisper",
    "speakers.propose",
    "speakers.validate",
    "transcript.correct_whisper",
    "transcript.enrich",
    "transcript.create_plain",
)

def inspection_plan() -> list[PlannedTask]:
    return [PlannedTask(task_id, "inspection_required") for task_id in INSPECTION_TASKS]


def processing_plan(context: PipelineContext) -> list[PlannedTask]:
    if not context.routing_ready:
        raise RuntimeError(
            "Le plan de traitement exige has_subtitles et video_type. "
            "Execute d'abord `python -m pipeline inspect ...`."
        )
    tasks = [
        PlannedTask(task_id, "all_videos")
        for task_id in COMMON_PROCESSING_TASKS
    ]
    if context.transcript_strategy == "ocr":
        tasks.extend(
            PlannedTask(task_id, "has_subtitles=true")
            for task_id in OCR_TRANSCRIPT_TASKS
        )
    else:
        tasks.extend(
            PlannedTask(task_id, "has_subtitles=false")
            for task_id in WHISPER_TRANSCRIPT_TASKS
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
