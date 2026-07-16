from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PipelineOptions:
    force: bool = False
    openai_mode: str = "normal"
    review_scope: str = "duo"
    image_review_model: str = field(
        default_factory=lambda: os.getenv(
            "OCR_OTHERS_REVIEW_MODEL",
            "gpt-5.6-luna",
        )
    )
    speaker_validation_model: str = field(
        default_factory=lambda: os.getenv(
            "CHUNK_SPEAKER_VALIDATION_MODEL",
            "gpt-5.4-nano",
        )
    )
    correction_mode: str = "balanced"
    frame_interval_seconds: float = 0.5
    details_per_section: int = 6

    def __post_init__(self) -> None:
        if self.openai_mode not in {"normal", "batch"}:
            raise ValueError("openai_mode doit valoir 'normal' ou 'batch'.")
        if self.review_scope not in {"duo", "all"}:
            raise ValueError("review_scope doit valoir 'duo' ou 'all'.")
        if self.correction_mode not in {"conservative", "balanced", "aggressive"}:
            raise ValueError("correction_mode invalide.")
        if self.frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds doit etre positif.")
        if self.details_per_section <= 0:
            raise ValueError("details_per_section doit etre positif.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
