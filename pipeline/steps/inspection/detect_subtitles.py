import statistics
from pathlib import Path

from pipeline.support.analysis import analysed_infos_path, update_routing_facts
from pipeline.support.json_io import read_json
from pipeline.support.paddle_ocr import (
    MIN_SUBTITLE_CLUSTER_SECONDS,
    anchored_subtitle_match,
    box_bounds,
    box_geometry,
    image_size,
    infer_subtitle_anchors,
    seconds_from_image_name,
)
from pipeline.support.paths import existing_images_dir, existing_ocr_dir, relative_to_video_dir


LOCATION_NAME = "ocr_box_locations.json"
LEGACY_LOCATION_NAME = "ocr_location.json"
BOXES_DIRNAME = "ocr_boxes_images"
LEGACY_BOXES_DIRNAME = "ocr_boxes"


def location_path(ocr_dir):
    new_path = ocr_dir / BOXES_DIRNAME / LOCATION_NAME
    legacy_boxes_path = ocr_dir / LEGACY_BOXES_DIRNAME / LOCATION_NAME
    legacy_path = ocr_dir / LEGACY_LOCATION_NAME
    direct_path = ocr_dir / LOCATION_NAME
    if new_path.exists():
        return new_path
    if legacy_boxes_path.exists():
        return legacy_boxes_path
    if direct_path.exists():
        return direct_path
    if not legacy_path.exists():
        return direct_path
    return legacy_path


def load_json(path):
    return read_json(path)


def write_has_subtitles(video_path, has_subtitles, details):
    return update_routing_facts(
        video_path,
        has_subtitles=bool(has_subtitles),
        has_subtitles_details=details,
    )


def subtitle_entries_from_boxes(payload, images_dir):
    entries = []
    sizes = {}

    for item in payload.get("items", []):
        image_name = item.get("image")
        if not image_name:
            continue

        image_path = images_dir / image_name
        if image_name not in sizes:
            sizes[image_name] = image_size(image_path)
        size = sizes[image_name]
        second = seconds_from_image_name(Path(image_name).name)

        for poly in item.get("boxes", []):
            bounds = box_bounds(poly)
            if not bounds:
                continue
            geometry = box_geometry(bounds, size)
            entries.append(
                {
                    "image": image_name,
                    "geometry": geometry,
                    "second": second,
                }
            )

    return entries


def longest_continuous_second_run(seconds):
    ordered = sorted({second for second in seconds if second is not None})
    if not ordered:
        return []
    if len(ordered) == 1:
        return ordered

    positive_gaps = [
        current - previous
        for previous, current in zip(ordered, ordered[1:])
        if current > previous
    ]
    expected_step = min(positive_gaps) if positive_gaps else 0.5
    max_allowed_gap = max(expected_step * 1.5, 0.75)

    best_start = 0
    best_end = 0
    current_start = 0

    for index in range(1, len(ordered)):
        if ordered[index] - ordered[index - 1] > max_allowed_gap:
            if (ordered[best_end] - ordered[best_start]) < (
                ordered[index - 1] - ordered[current_start]
            ):
                best_start = current_start
                best_end = index - 1
            current_start = index

    if (ordered[best_end] - ordered[best_start]) < (
        ordered[-1] - ordered[current_start]
    ):
        best_start = current_start
        best_end = len(ordered) - 1

    return ordered[best_start : best_end + 1]


def analyze_subtitle_anchor(entries):
    anchors = infer_subtitle_anchors(entries)
    if len(anchors) < 2:
        return {
            "has_subtitles": False,
            "reason": "not_enough_anchor_candidates",
            "anchor_count": len(anchors),
            "required_continuous_seconds": float(MIN_SUBTITLE_CLUSTER_SECONDS),
            "total_matching_seconds_count": 0,
            "all_matching_seconds": [],
            "longest_continuous_seconds_count": 0,
            "longest_continuous_seconds_duration": 0.0,
            "longest_continuous_seconds": [],
            "matching_images": [],
        }

    anchor_cx = statistics.median(anchor["cx"] for anchor in anchors)
    anchor_cy = statistics.median(anchor["cy"] for anchor in anchors)
    anchor_height = statistics.median(anchor["relative_height"] for anchor in anchors)
    anchor_widths = [anchor["relative_width"] for anchor in anchors]
    x_tolerance = max(0.06, min(0.16, statistics.median(anchor_widths) * 0.25))
    y_tolerance = max(0.04, min(0.075, anchor_height * 1.6))
    matching_entries = [
        entry
        for entry in entries
        if entry["second"] is not None
        and anchored_subtitle_match(
            entry["geometry"],
            anchor_cx,
            anchor_cy,
            x_tolerance,
            y_tolerance,
        )
    ]
    matching_seconds = sorted({entry["second"] for entry in matching_entries})
    longest_run_seconds = longest_continuous_second_run(matching_seconds)
    longest_run_duration = (
        longest_run_seconds[-1] - longest_run_seconds[0]
        if len(longest_run_seconds) >= 2
        else 0.0
    )
    matching_images = []
    seen_images = set()
    for entry in matching_entries:
        image_name = entry["image"]
        if image_name in seen_images:
            continue
        seen_images.add(image_name)
        matching_images.append(image_name)

    required_continuous_seconds = float(MIN_SUBTITLE_CLUSTER_SECONDS)
    has_subtitles = longest_run_duration >= required_continuous_seconds
    return {
        "has_subtitles": has_subtitles,
        "reason": (
            "stable_anchor_across_continuous_seconds"
            if has_subtitles
            else "not_enough_continuous_matching_seconds"
        ),
        "anchor_count": len(anchors),
        "anchor": {
            "cx": round(anchor_cx, 4),
            "cy": round(anchor_cy, 4),
            "relative_height": round(anchor_height, 4),
            "median_relative_width": round(statistics.median(anchor_widths), 4),
        },
        "tolerances": {
            "x": round(x_tolerance, 4),
            "y": round(y_tolerance, 4),
        },
        "required_continuous_seconds": required_continuous_seconds,
        "total_matching_seconds_count": len(matching_seconds),
        "all_matching_seconds": matching_seconds,
        "longest_continuous_seconds_count": len(longest_run_seconds),
        "longest_continuous_seconds_duration": round(longest_run_duration, 3),
        "longest_continuous_seconds": longest_run_seconds,
        "matching_images": matching_images,
    }


def detect_for_video(video_path, force=False):
    ocr_dir = existing_ocr_dir(video_path)
    images_dir = existing_images_dir(video_path)
    source = location_path(ocr_dir)
    target = analysed_infos_path(video_path)

    if target.exists() and not force:
        try:
            if "has_subtitles" in load_json(target):
                print(f"[skip] {video_path.name}: has_subtitles existe deja")
                return None
        except Exception:
            pass

    if not source.exists():
        print(f"[skip] OCR location introuvable: {source}")
        return None

    payload = load_json(source)
    entries = subtitle_entries_from_boxes(payload, images_dir)
    analysis = analyze_subtitle_anchor(entries)
    details = {
        "source": relative_to_video_dir(source, video_path),
        **analysis,
    }
    has_subtitles = analysis["has_subtitles"]
    write_has_subtitles(video_path, has_subtitles, details)
    print(f"[ok] {video_path.name}: has_subtitles={str(has_subtitles).lower()}", flush=True)
    return True
