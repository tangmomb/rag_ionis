from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .probe import probe_video


LONG_VIDEO_THRESHOLD_SECONDS = 600
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
ANALYSIS_NAME = "pipeline_analysis.json"
MANIFEST_NAME = "video_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def video_in_directory(directory: str | Path) -> Path:
    root = Path(directory)
    videos = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(videos) != 1:
        raise RuntimeError(
            f"{root} doit contenir exactement une video; trouve: {len(videos)}."
        )
    return videos[0]


@dataclass(frozen=True)
class VideoContext:
    video_path: Path
    media: dict[str, Any]
    analysis: dict[str, Any]

    @classmethod
    def inspect(cls, video_path: str | Path) -> "VideoContext":
        candidate = Path(video_path)
        video = video_in_directory(candidate) if candidate.is_dir() else candidate
        analysis = load_json(video.parent / "metadata" / ANALYSIS_NAME)
        return cls(video_path=video, media=probe_video(video), analysis=analysis)

    @property
    def video_dir(self) -> Path:
        return self.video_path.parent

    @property
    def video_id(self) -> str:
        return self.video_path.stem

    @property
    def metadata_dir(self) -> Path:
        return self.video_dir / "metadata"

    @property
    def manifest_path(self) -> Path:
        return self.metadata_dir / MANIFEST_NAME

    @property
    def duration_seconds(self) -> float:
        value = self.media.get("duration_seconds")
        if value is None:
            raise RuntimeError(f"Duree video introuvable: {self.video_path}")
        return float(value)

    @property
    def is_long_video(self) -> bool:
        return self.duration_seconds > LONG_VIDEO_THRESHOLD_SECONDS

    @property
    def has_subtitles(self) -> bool | None:
        value = self.analysis.get("has_subtitles")
        return value if isinstance(value, bool) else None

    @property
    def video_type(self) -> str | None:
        value = self.analysis.get("video_type")
        normalized = str(value).strip() if value is not None else ""
        return normalized or None

    @property
    def visual_strategy(self) -> str | None:
        return self.video_type

    @property
    def transcript_strategy(self) -> str | None:
        if self.has_subtitles is True:
            return "ocr"
        if self.has_subtitles is False:
            return "whisper"
        return None

    @property
    def chunk_strategy(self) -> str:
        return "long" if self.is_long_video else "short"

    @property
    def routing_ready(self) -> bool:
        return self.transcript_strategy is not None and self.visual_strategy is not None

    def routing(self) -> dict[str, Any]:
        missing = []
        if self.has_subtitles is None:
            missing.append("has_subtitles")
        if self.visual_strategy is None:
            missing.append("video_type")
        parts = [
            self.chunk_strategy,
            self.transcript_strategy,
            self.visual_strategy,
        ]
        return {
            "status": "ready" if not missing else "needs_content_inspection",
            "pipeline_id": ".".join(part for part in parts if part),
            "transcript_strategy": self.transcript_strategy,
            "chunk_strategy": self.chunk_strategy,
            "visual_strategy": self.visual_strategy,
            "missing_features": missing,
        }
