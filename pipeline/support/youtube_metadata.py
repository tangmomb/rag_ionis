from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from pipeline.support.json_io import read_json
from pipeline.support.paths import existing_youtube_api_infos_path


class YoutubeMetadataError(RuntimeError):
    """Raised when the metadata required by the pipeline is unavailable."""


def load_youtube_metadata(video_path: str | Path) -> dict[str, Any]:
    target = existing_youtube_api_infos_path(video_path)
    if not target.is_file():
        raise YoutubeMetadataError(
            f"Metadonnees YouTube introuvables: {target}. "
            "Lance d'abord l'ingestion des metadonnees."
        )
    try:
        payload = read_json(target)
    except (OSError, UnicodeError, ValueError) as error:
        raise YoutubeMetadataError(
            f"Metadonnees YouTube invalides ou illisibles: {target}."
        ) from error
    if not isinstance(payload, dict):
        raise YoutubeMetadataError(
            f"Les metadonnees YouTube doivent contenir un objet JSON: {target}."
        )
    return payload


def youtube_duration_seconds(metadata: Mapping[str, Any]) -> float:
    value = metadata.get("duration_seconds")
    if isinstance(value, bool):
        value = None
    try:
        duration = float(value)
    except (TypeError, ValueError) as error:
        raise YoutubeMetadataError(
            "duration_seconds est absent ou invalide dans les metadonnees YouTube."
        ) from error
    if not math.isfinite(duration) or duration <= 0:
        raise YoutubeMetadataError(
            "duration_seconds doit etre un nombre strictement positif "
            "dans les metadonnees YouTube."
        )
    return duration
