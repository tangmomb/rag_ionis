import json
from pathlib import Path


ANALYSED_INFOS_SUFFIX = "_analysed_infos.json"
LEGACY_ANALYSED_INFOS_NAME = "analysed_infos.json"
ALLOWED_KEYS = {
    "video_type",
    "has_subtitles",
}


def analysed_infos_path(video_path):
    path = Path(video_path)
    if path.is_dir():
        video_id = path.name
        base_dir = path
    else:
        video_id = path.stem
        base_dir = path.parent
    preferred = base_dir / f"{video_id}{ANALYSED_INFOS_SUFFIX}"
    legacy = base_dir / LEGACY_ANALYSED_INFOS_NAME
    if preferred.exists() or not legacy.exists():
        return preferred
    return legacy


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
    target = analysed_infos_path(video_path)
    data = _load_allowed_data(target)
    if payload:
        for key in ALLOWED_KEYS:
            if key in payload:
                data[key] = payload[key]
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
