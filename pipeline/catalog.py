from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .context import VideoContext
from .options import PipelineOptions


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskSpec:
    id: str
    phase: Literal["inspection", "processing"]
    title: str
    target: str
    supports_force: bool = True


TASKS = {
    "frames.extract": TaskSpec(
        "frames.extract",
        "inspection",
        "Extraire les frames",
        "pipeline.steps.inspection.extract_frames",
    ),
    "frames.classify": TaskSpec(
        "frames.classify",
        "inspection",
        "Classifier les frames",
        "pipeline.steps.inspection.classify_frames",
    ),
    "video.detect_interview": TaskSpec(
        "video.detect_interview",
        "inspection",
        "Detecter les interviews",
        "pipeline.steps.inspection.detect_interviews",
    ),
    "video.infer_type": TaskSpec(
        "video.infer_type",
        "inspection",
        "Inferer le type de video",
        "pipeline.steps.inspection.infer_video_type",
    ),
    "ocr.extract_raw": TaskSpec(
        "ocr.extract_raw",
        "inspection",
        "Extraire l'OCR brut",
        "pipeline.steps.inspection.extract_raw_ocr",
    ),
    "ocr.extract_boxes": TaskSpec(
        "ocr.extract_boxes",
        "inspection",
        "Extraire les positions OCR",
        "pipeline.steps.inspection.extract_ocr_boxes",
    ),
    "video.detect_subtitles": TaskSpec(
        "video.detect_subtitles",
        "inspection",
        "Detecter les sous-titres incrustes",
        "pipeline.steps.inspection.detect_subtitles",
    ),
    "ocr.build_processed": TaskSpec(
        "ocr.build_processed",
        "processing",
        "Construire l'OCR traite",
        "pipeline.steps.ocr.build_processed_ocr",
    ),
    "ocr.filter_overlays": TaskSpec(
        "ocr.filter_overlays",
        "processing",
        "Filtrer les overlays OCR",
        "pipeline.steps.ocr.filter_processed_ocr",
    ),
    "ocr.extract_review_candidates": TaskSpec(
        "ocr.extract_review_candidates",
        "processing",
        "Extraire les textes a verifier",
        "pipeline.steps.ocr.extract_other_text_candidates",
    ),
    "ocr.review_other_text": TaskSpec(
        "ocr.review_other_text",
        "processing",
        "Verifier les autres textes",
        "pipeline.steps.ocr.review_other_text_candidates",
    ),
    "ocr.apply_review": TaskSpec(
        "ocr.apply_review",
        "processing",
        "Appliquer la verification OCR",
        "pipeline.steps.ocr.apply_other_text_review",
    ),
    "transcript.extract_ocr": TaskSpec(
        "transcript.extract_ocr",
        "processing",
        "Construire le transcript depuis les sous-titres OCR",
        "pipeline.steps.transcripts.extract_ocr_subtitles",
    ),
    "transcript.correct_ocr_spacing": TaskSpec(
        "transcript.correct_ocr_spacing",
        "processing",
        "Corriger les espaces du transcript OCR",
        "pipeline.steps.transcripts.correct_ocr_subtitle_spacing",
    ),
    "transcript.normalize_brand": TaskSpec(
        "transcript.normalize_brand",
        "processing",
        "Normaliser Ionis-STM",
        "pipeline.steps.transcripts.normalize_ionis_stm",
    ),
    "transcript.whisper": TaskSpec(
        "transcript.whisper",
        "processing",
        "Transcrire l'audio avec WhisperX",
        "pipeline.steps.transcripts.transcribe_with_whisper",
    ),
    "speakers.propose": TaskSpec(
        "speakers.propose",
        "processing",
        "Proposer les speakers",
        "pipeline.steps.speakers.propose_speakers",
    ),
    "speakers.validate": TaskSpec(
        "speakers.validate",
        "processing",
        "Valider les speakers",
        "pipeline.steps.speakers.validate_speakers",
    ),
    "speakers.assign_ocr": TaskSpec(
        "speakers.assign_ocr",
        "processing",
        "Attribuer les speakers au transcript OCR",
        "pipeline.steps.speakers.assign_ocr_speakers",
    ),
    "transcript.correct_whisper": TaskSpec(
        "transcript.correct_whisper",
        "processing",
        "Corriger le transcript Whisper",
        "pipeline.steps.transcripts.correct_whisper_transcript",
    ),
    "transcript.enrich": TaskSpec(
        "transcript.enrich",
        "processing",
        "Enrichir le transcript avec les textes visuels",
        "pipeline.steps.transcripts.enrich_transcripts",
    ),
    "transcript.create_plain": TaskSpec(
        "transcript.create_plain",
        "processing",
        "Creer le transcript sans timecodes",
        "pipeline.steps.transcripts.create_plain_transcript",
    ),
    "chunks.create": TaskSpec(
        "chunks.create",
        "processing",
        "Creer les chunks detail",
        "pipeline.steps.chunks.create_transcript_chunks",
    ),
    "chunks.summarize_sections": TaskSpec(
        "chunks.summarize_sections",
        "processing",
        "Creer les resumes de sections",
        "pipeline.steps.chunks.hierarchical_chunks",
    ),
    "chunks.summarize_video": TaskSpec(
        "chunks.summarize_video",
        "processing",
        "Creer le resume global",
        "pipeline.steps.chunks.hierarchical_chunks",
    ),
    "embeddings.create": TaskSpec(
        "embeddings.create",
        "processing",
        "Creer les embeddings",
        "pipeline.steps.embeddings.create_chunk_embeddings",
    ),
}


def task_args(
    task_id: str,
    context: VideoContext,
    options: PipelineOptions,
) -> list[str]:
    args = ["--video-dir", str(context.video_dir)]
    if task_id == "frames.extract":
        args += ["--interval", str(options.frame_interval_seconds)]
    elif task_id == "ocr.build_processed":
        args += ["--transcript-strategy", str(context.transcript_strategy)]
    elif task_id == "ocr.review_other_text":
        args += ["--model", options.image_review_model]
        if options.review_scope == "duo":
            args += ["--limit-images", "1"]
    elif task_id == "transcript.extract_ocr":
        args += ["--has-subtitles", "true"]
    elif task_id == "transcript.correct_ocr_spacing":
        args += [
            "--model",
            options.speaker_validation_model,
            "--mode",
            options.openai_mode,
        ]
    elif task_id == "transcript.whisper":
        args += ["--has-subtitles", "false"]
    elif task_id == "speakers.validate":
        args += [
            "--model",
            options.speaker_validation_model,
            "--mode",
            options.openai_mode,
        ]
    elif task_id == "transcript.correct_whisper":
        args += [
            "--mode",
            options.correction_mode,
            "--has-subtitles",
            "false",
        ]
    elif task_id == "transcript.enrich":
        args += [
            "--has-subtitles",
            "true" if context.transcript_strategy == "ocr" else "false",
        ]
    elif task_id == "chunks.create":
        args += ["--profile", context.chunk_strategy]
    elif task_id == "chunks.summarize_sections":
        args += [
            "--operation",
            "sections",
            "--details-per-section",
            str(options.details_per_section),
        ]
    elif task_id == "chunks.summarize_video":
        args += ["--operation", "video"]

    if options.force and TASKS[task_id].supports_force:
        args.append("--force")
    return args


def task_environment(context: VideoContext, options: PipelineOptions) -> dict[str, str]:
    transcript_dir = (
        "transcripts_ocr"
        if context.transcript_strategy == "ocr"
        else "transcripts_whisper"
    )
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(PROJECT_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    return {
        "PYTHONUTF8": "1",
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "PIPELINE_OPENAI_MODE": options.openai_mode,
        "PIPELINE_TRANSCRIPTS_DIR_NAME": transcript_dir,
    }


def task_command(
    task_id: str,
    context: VideoContext,
    options: PipelineOptions,
) -> list[str]:
    spec = TASKS[task_id]
    prefix = [sys.executable, "-m", spec.target]
    return [*prefix, *task_args(task_id, context, options)]
