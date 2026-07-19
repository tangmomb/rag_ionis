from __future__ import annotations

from collections.abc import Mapping

from pipeline.support.json_io import read_json
from pipeline.support.paths import analysed_infos_path, metadata_dir


MANIFEST_NAME = "video_manifest.json"
ALLOWED_KEYS = {
    "video_type",
    "has_subtitles",
    "has_subtitles_details",
}


def _allowed_data(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        key: payload[key]
        for key in ALLOWED_KEYS
        if key in payload
    }


def load_routing_facts(
    video_path,
    *,
    legacy_fallback: bool = True,
) -> dict[str, object]:
    """Charge les faits depuis le manifeste, avec migration des anciens runs."""
    manifest = read_json(
        metadata_dir(video_path) / MANIFEST_NAME,
        default={},
    )
    if isinstance(manifest, Mapping):
        facts = _allowed_data(manifest.get("routing_facts"))
        if facts:
            return facts

    if not legacy_fallback:
        return {}

    legacy_path = analysed_infos_path(video_path)
    return _allowed_data(read_json(legacy_path, default={}))


def routing_fact(video_path, key: str):
    if key not in ALLOWED_KEYS:
        raise KeyError(f"Fait de routage inconnu: {key}")
    return load_routing_facts(video_path).get(key)
