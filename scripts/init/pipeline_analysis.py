import json
from pathlib import Path

from pipeline_paths import ANALYSED_INFOS_NAME
from pipeline_paths import analysed_infos_path as pipeline_analysed_infos_path
from pipeline_paths import metadata_dir
from pipeline_paths import video_base_dir, video_id


ANALYSED_INFOS_SUFFIX = "_analysed_infos.json"
LEGACY_ANALYSED_INFOS_NAME = "analysed_infos.json"
ALLOWED_KEYS = {
    "video_type",
    "has_subtitles",
}


def analysed_infos_path(video_path):
    return pipeline_analysed_infos_path(video_path)


def _load_allowed_data(target):
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        key: payload[key]
        for key in ALLOWED_KEYS
        if key in payload
    }


def update_analysed_infos(video_path, stage, payload=None):
    del stage
    target = metadata_dir(video_path) / ANALYSED_INFOS_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _load_allowed_data(target)
    if not data:
        data = _load_allowed_data(analysed_infos_path(video_path))
    if not data:
        legacy = video_base_dir(video_path) / f"{video_id(video_path)}{ANALYSED_INFOS_SUFFIX}"
        data = _load_allowed_data(legacy)
    if payload:
        for key in ALLOWED_KEYS:
            if key in payload:
                data[key] = payload[key]
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
