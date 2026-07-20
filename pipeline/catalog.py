from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from . import step_handlers
from .context import PipelineContext
from .contracts import TaskResult


StepHandler = Callable[[PipelineContext], TaskResult]
TaskPostcondition = Callable[[PipelineContext], bool]


def _frames_extracted(context: PipelineContext) -> bool:
    images = context.outputs_dir / "images"
    return images.is_dir() and any(
        candidate.is_file()
        and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        for candidate in images.rglob("*")
    )


@dataclass(frozen=True)
class TaskSpec:
    id: str
    phase: Literal["inspection", "processing"]
    title: str
    handler: StepHandler
    version: str = "1"
    postcondition: TaskPostcondition | None = None

    @property
    def entrypoint(self) -> str:
        module = getattr(
            self.handler,
            "__module__",
            self.handler.__class__.__module__,
        )
        name = getattr(
            self.handler,
            "__name__",
            self.handler.__class__.__qualname__,
        )
        return f"{module}.{name}"

    def postcondition_satisfied(self, context: PipelineContext) -> bool:
        return bool(self.postcondition and self.postcondition(context))


_TASK_SPECS = (
    TaskSpec(
        "frames.extract",
        "inspection",
        "Extraire les frames",
        step_handlers.extract_frames,
        postcondition=_frames_extracted,
    ),
    TaskSpec(
        "frames.classify",
        "inspection",
        "Classifier les frames",
        step_handlers.classify_frames,
    ),
    TaskSpec(
        "video.detect_interview",
        "inspection",
        "Detecter les interviews",
        step_handlers.detect_interview,
    ),
    TaskSpec(
        "video.infer_type",
        "inspection",
        "Inferer le type de video",
        step_handlers.infer_video_type,
        version="2",
        postcondition=lambda context: context.video_type is not None,
    ),
    TaskSpec(
        "ocr.extract_raw",
        "inspection",
        "Extraire l'OCR brut",
        step_handlers.extract_raw_ocr,
    ),
    TaskSpec(
        "ocr.extract_boxes",
        "inspection",
        "Extraire les positions OCR",
        step_handlers.extract_ocr_boxes,
    ),
    TaskSpec(
        "video.detect_subtitles",
        "inspection",
        "Detecter les sous-titres incrustes",
        step_handlers.detect_subtitles,
        postcondition=lambda context: context.has_subtitles is not None,
    ),
    TaskSpec(
        "ocr.build_processed",
        "processing",
        "Construire l'OCR traite",
        step_handlers.build_processed_ocr,
    ),
    TaskSpec(
        "ocr.filter_overlays",
        "processing",
        "Filtrer les overlays OCR",
        step_handlers.filter_ocr_overlays,
    ),
    TaskSpec(
        "ocr.extract_review_candidates",
        "processing",
        "Extraire les textes a verifier",
        step_handlers.extract_review_candidates,
    ),
    TaskSpec(
        "ocr.review_other_text",
        "processing",
        "Verifier les autres textes",
        step_handlers.review_other_text,
    ),
    TaskSpec(
        "ocr.apply_review",
        "processing",
        "Appliquer la verification OCR",
        step_handlers.apply_ocr_review,
    ),
    TaskSpec(
        "transcript.extract_ocr",
        "processing",
        "Construire le transcript depuis les sous-titres OCR",
        step_handlers.extract_ocr_transcript,
    ),
    TaskSpec(
        "transcript.normalize_brand",
        "processing",
        "Normaliser Ionis-STM",
        step_handlers.normalize_brand,
    ),
    TaskSpec(
        "transcript.whisper",
        "processing",
        "Transcrire l'audio avec WhisperX",
        step_handlers.transcribe_whisper,
    ),
    TaskSpec(
        "speakers.propose",
        "processing",
        "Proposer les speakers",
        step_handlers.propose_speakers,
    ),
    TaskSpec(
        "speakers.validate",
        "processing",
        "Valider les speakers",
        step_handlers.validate_speakers,
    ),
    TaskSpec(
        "transcript.correct_whisper",
        "processing",
        "Corriger le transcript Whisper",
        step_handlers.correct_whisper_transcript,
    ),
    TaskSpec(
        "transcript.reconcile_ocr",
        "processing",
        "Corriger WhisperX par rapprochement avec le transcript OCR",
        step_handlers.reconcile_whisper_with_ocr,
    ),
    TaskSpec(
        "transcript.apply_speakers",
        "processing",
        "Creer le transcript avec les speakers valides",
        step_handlers.apply_transcript_speakers,
    ),
    TaskSpec(
        "transcript.enrich",
        "processing",
        "Ajouter les intercalaires au transcript avec speakers",
        step_handlers.enrich_transcript,
    ),
    TaskSpec(
        "transcript.create_plain",
        "processing",
        "Creer le transcript sans timecodes",
        step_handlers.create_plain_transcript,
        version="2",
    ),
    TaskSpec(
        "transcript.create_plain_ocr",
        "processing",
        "Creer le plain transcript OCR de correction",
        step_handlers.create_plain_ocr,
    ),
    TaskSpec(
        "chunks.create",
        "processing",
        "Creer les chunks detail",
        step_handlers.create_chunks,
    ),
    TaskSpec(
        "chunks.summarize_sections",
        "processing",
        "Creer les resumes de sections",
        step_handlers.summarize_sections,
    ),
    TaskSpec(
        "chunks.summarize_video",
        "processing",
        "Creer le resume global",
        step_handlers.summarize_video,
    ),
    TaskSpec(
        "embeddings.create",
        "processing",
        "Creer les embeddings",
        step_handlers.create_embeddings,
    ),
)

TASKS = {spec.id: spec for spec in _TASK_SPECS}
if len(TASKS) != len(_TASK_SPECS):
    raise RuntimeError("Le catalogue contient un identifiant de tache duplique.")
