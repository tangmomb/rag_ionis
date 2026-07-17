import json
from pathlib import Path

from pipeline.probe import probe_video
from pipeline.support.analysis import update_routing_facts
from pipeline.support.json_io import read_json
from pipeline.support.paths import (
    existing_images_dir,
    existing_interview_dir,
    existing_youtube_api_infos_path,
)

MANIFEST_NAME = "frame_classification_manifest.json"
LEGACY_MANIFEST_NAME = "manifest.json"
INTERVIEW_MANIFEST_NAME = "interview_detection_manifest.json"
MOTION_DESIGN_MAX_FOOTAGE_RATIO = 0.15
MOTION_DESIGN_MAX_DURATION_SECONDS = 180


def manifest_path(video_path):
    images_dir = existing_images_dir(video_path)
    preferred = images_dir / MANIFEST_NAME
    legacy = images_dir / LEGACY_MANIFEST_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def interview_manifest_path(video_path):
    interview_dir = existing_interview_dir(video_path)
    preferred = interview_dir / INTERVIEW_MANIFEST_NAME
    legacy = interview_dir / LEGACY_MANIFEST_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_json(path):
    return read_json(path)


def manifest_class_counts(payload):
    class_counts = payload.get("class_counts")
    if isinstance(class_counts, dict):
        footage_count = int(class_counts.get("footage", 0) or 0)
        graphic_count = int(class_counts.get("graphic", 0) or 0)
        mixture_count = int(class_counts.get("mixture", 0) or 0)
        return footage_count, graphic_count, mixture_count

    items = payload.get("items", [])
    counts = {"footage": 0, "graphic": 0, "mixture": 0}
    for item in items:
        label = str(item.get("pred_label", "")).strip().lower()
        if label in counts:
            counts[label] += 1
    return counts["footage"], counts["graphic"], counts["mixture"]


def infer_video_type_from_manifest(payload, duration_seconds=None):
    footage_count, graphic_count, mixture_count = manifest_class_counts(payload)
    total_count = footage_count + graphic_count + mixture_count
    labels = {
        str(item.get("pred_label", "")).strip().lower()
        for item in payload.get("items", [])
        if str(item.get("pred_label", "")).strip()
    }
    footage_ratio = (footage_count / total_count) if total_count > 0 else 0.0
    short_enough_for_motion_design = (
        isinstance(duration_seconds, (int, float))
        and not isinstance(duration_seconds, bool)
        and duration_seconds < MOTION_DESIGN_MAX_DURATION_SECONDS
    )
    if (
        short_enough_for_motion_design
        and total_count > 0
        and footage_ratio < MOTION_DESIGN_MAX_FOOTAGE_RATIO
    ):
        return "motion_design"
    if short_enough_for_motion_design and labels and "footage" not in labels:
        return "motion_design"
    return "video_recording"


def video_duration_seconds(video_path):
    sources = [
        (existing_youtube_api_infos_path(video_path), ("duration_seconds",)),
        (
            Path(video_path).parent / "metadata" / "video_manifest.json",
            ("video", "duration_seconds"),
        ),
    ]
    for source, keys in sources:
        if not source.exists():
            continue
        try:
            value = load_json(source)
            for key in keys:
                value = value[key]
            if isinstance(value, bool):
                continue
            return float(value)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

    try:
        value = probe_video(video_path).get("duration_seconds")
        return float(value) if value is not None else None
    except (OSError, RuntimeError, ValueError):
        return None


def write_analysed_infos(video_path, video_type):
    return update_routing_facts(video_path, video_type=video_type)


def infer_for_video(video_path, force=False):
    source = manifest_path(video_path)
    if not source.exists():
        print(f"[skip] manifest introuvable: {source}")
        return None

    interview_source = interview_manifest_path(video_path)
    if interview_source.exists():
        interview_payload = load_json(interview_source)
        if bool(interview_payload.get("is_interview")):
            video_type = "interview"
            write_analysed_infos(video_path, video_type)
            print(f"[ok] {video_path.name}: video_type={video_type}", flush=True)
            return True

    payload = load_json(source)
    duration_seconds = video_duration_seconds(video_path)
    video_type = infer_video_type_from_manifest(payload, duration_seconds=duration_seconds)
    write_analysed_infos(video_path, video_type)
    duration_label = f"{duration_seconds:g}s" if duration_seconds is not None else "inconnue"
    print(
        f"[ok] {video_path.name}: video_type={video_type}, duree={duration_label}",
        flush=True,
    )
    return True
