from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from . import step_handlers
from .context import PipelineContext


StepHandler = Callable[[PipelineContext], Any]


@dataclass(frozen=True)
class TaskSpec:
    id: str
    phase: Literal["inspection", "processing"]
    title: str
    handler: StepHandler

    @property
    def entrypoint(self) -> str:
        return f"{self.handler.__module__}.{self.handler.__name__}"


TASKS = {
    "frames.extract": TaskSpec(
        "frames.extract",
        "inspection",
        "Extraire les frames",
        step_handlers.extract_frames,
    ),
    "frames.classify": TaskSpec(
        "frames.classify",
        "inspection",
        "Classifier les frames",
        step_handlers.classify_frames,
    ),
    "video.detect_interview": TaskSpec(
        "video.detect_interview",
        "inspection",
        "Detecter les interviews",
        step_handlers.detect_interview,
    ),
    "video.infer_type": TaskSpec(
        "video.infer_type",
        "inspection",
        "Inferer le type de video",
        step_handlers.infer_video_type,
    ),
    "ocr.extract_raw": TaskSpec(
        "ocr.extract_raw",
        "inspection",
        "Extraire l'OCR brut",
        step_handlers.extract_raw_ocr,
    ),
    "ocr.extract_boxes": TaskSpec(
        "ocr.extract_boxes",
        "inspection",
        "Extraire les positions OCR",
        step_handlers.extract_ocr_boxes,
    ),
    "video.detect_subtitles": TaskSpec(
        "video.detect_subtitles",
        "inspection",
        "Detecter les sous-titres incrustes",
        step_handlers.detect_subtitles,
    ),
    "ocr.build_processed": TaskSpec(
        "ocr.build_processed",
        "processing",
        "Construire l'OCR traite",
        step_handlers.build_processed_ocr,
    ),
    "ocr.filter_overlays": TaskSpec(
        "ocr.filter_overlays",
        "processing",
        "Filtrer les overlays OCR",
        step_handlers.filter_ocr_overlays,
    ),
    "ocr.extract_review_candidates": TaskSpec(
        "ocr.extract_review_candidates",
        "processing",
        "Extraire les textes a verifier",
        step_handlers.extract_review_candidates,
    ),
    "ocr.review_other_text": TaskSpec(
        "ocr.review_other_text",
        "processing",
        "Verifier les autres textes",
        step_handlers.review_other_text,
    ),
    "ocr.apply_review": TaskSpec(
        "ocr.apply_review",
        "processing",
        "Appliquer la verification OCR",
        step_handlers.apply_ocr_review,
    ),
    "transcript.extract_ocr": TaskSpec(
        "transcript.extract_ocr",
        "processing",
        "Construire le transcript depuis les sous-titres OCR",
        step_handlers.extract_ocr_transcript,
    ),
    "transcript.correct_ocr_spacing": TaskSpec(
        "transcript.correct_ocr_spacing",
        "processing",
        "Corriger les espaces du transcript OCR",
        step_handlers.correct_ocr_spacing,
    ),
    "transcript.normalize_brand": TaskSpec(
        "transcript.normalize_brand",
        "processing",
        "Normaliser Ionis-STM",
        step_handlers.normalize_brand,
    ),
    "transcript.whisper": TaskSpec(
        "transcript.whisper",
        "processing",
        "Transcrire l'audio avec WhisperX",
        step_handlers.transcribe_whisper,
    ),
    "speakers.propose": TaskSpec(
        "speakers.propose",
        "processing",
        "Proposer les speakers",
        step_handlers.propose_speakers,
    ),
    "speakers.validate": TaskSpec(
        "speakers.validate",
        "processing",
        "Valider les speakers",
        step_handlers.validate_speakers,
    ),
    "speakers.assign_ocr": TaskSpec(
        "speakers.assign_ocr",
        "processing",
        "Attribuer les speakers au transcript OCR",
        step_handlers.assign_ocr_speakers,
    ),
    "transcript.correct_whisper": TaskSpec(
        "transcript.correct_whisper",
        "processing",
        "Corriger le transcript Whisper",
        step_handlers.correct_whisper_transcript,
    ),
    "transcript.enrich": TaskSpec(
        "transcript.enrich",
        "processing",
        "Enrichir le transcript avec les textes visuels",
        step_handlers.enrich_transcript,
    ),
    "transcript.create_plain": TaskSpec(
        "transcript.create_plain",
        "processing",
        "Creer le transcript sans timecodes",
        step_handlers.create_plain_transcript,
    ),
    "chunks.create": TaskSpec(
        "chunks.create",
        "processing",
        "Creer les chunks detail",
        step_handlers.create_chunks,
    ),
    "chunks.summarize_sections": TaskSpec(
        "chunks.summarize_sections",
        "processing",
        "Creer les resumes de sections",
        step_handlers.summarize_sections,
    ),
    "chunks.summarize_video": TaskSpec(
        "chunks.summarize_video",
        "processing",
        "Creer le resume global",
        step_handlers.summarize_video,
    ),
    "embeddings.create": TaskSpec(
        "embeddings.create",
        "processing",
        "Creer les embeddings",
        step_handlers.create_embeddings,
    ),
}
