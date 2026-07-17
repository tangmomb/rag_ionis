from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import ANALYSED_INFOS_NAME
from pipeline.support.paths import analysed_infos_path as pipeline_analysed_infos_path
from pipeline.support.paths import metadata_dir
from pipeline.support.paths import video_base_dir, video_id


ANALYSED_INFOS_SUFFIX = "_analysed_infos.json"
LEGACY_ANALYSED_INFOS_NAME = "analysed_infos.json"
ALLOWED_KEYS = {
    "video_type",
    "has_subtitles",
    "has_subtitles_details",
}


def analysed_infos_path(video_path):
    return pipeline_analysed_infos_path(video_path)


def _load_allowed_data(target):
    payload = read_json(target, default={})
    if not isinstance(payload, dict):
        return {}
    return {
        key: payload[key]
        for key in ALLOWED_KEYS
        if key in payload
    }


def update_routing_facts(video_path, payload=None, **facts):
    target = metadata_dir(video_path) / ANALYSED_INFOS_NAME
    data = _load_allowed_data(target)
    if not data:
        data = _load_allowed_data(analysed_infos_path(video_path))
    if not data:
        legacy = video_base_dir(video_path) / f"{video_id(video_path)}{ANALYSED_INFOS_SUFFIX}"
        data = _load_allowed_data(legacy)
    updates = dict(payload or {})
    updates.update(facts)
    data.update(
        (key, updates[key])
        for key in ALLOWED_KEYS
        if key in updates
    )
    return write_json(target, data)


def update_analysed_infos(video_path, stage=None, payload=None):
    """Compatibility alias for legacy steps; stage was never persisted."""
    del stage
    return update_routing_facts(video_path, payload)
