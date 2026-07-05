import json
from datetime import datetime, timezone
from pathlib import Path


def analysed_infos_path(video_path):
    path = Path(video_path)
    if path.is_dir():
        return path / "analysed_infos.json"
    return path.parent / "analysed_infos.json"


def iso_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deep_merge(base, incoming):
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def update_analysed_infos(video_path, stage, payload=None):
    video_path = Path(video_path)
    target = analysed_infos_path(video_path)
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("video_id", video_path.stem)
    data.setdefault("video_file", video_path.name)
    data.setdefault("video_dir", video_path.parent.name)
    data.setdefault("stages", {})
    stage_payload = data["stages"].get(stage, {})
    if payload:
        stage_payload = _deep_merge(stage_payload, dict(payload))
        if payload.get("video_type"):
            data["video_type"] = payload["video_type"]
    stage_payload["updated_at"] = iso_now()
    data["stages"][stage] = stage_payload
    data["updated_at"] = iso_now()
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
