from pathlib import Path

from pipeline.support.json_io import read_json
from pipeline.support.paddle_ocr import (
    box_bounds,
    box_geometry,
    image_size,
    seconds_from_image_name,
    select_subtitle_anchor,
    text_key,
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

        texts = item.get("texts")
        has_text_metadata = isinstance(texts, list)
        for index, poly in enumerate(item.get("boxes", [])):
            bounds = box_bounds(poly)
            if not bounds:
                continue
            geometry = box_geometry(bounds, size)
            entry = {
                "image": image_name,
                "geometry": geometry,
                "second": second,
            }
            if has_text_metadata:
                text = texts[index] if index < len(texts) else ""
                entry["key"] = text_key(text)
            entries.append(entry)

    return entries


def analyze_subtitle_anchor(entries):
    analysis = select_subtitle_anchor(entries)
    matching_entries = analysis.pop("_matching_entries", [])
    analysis.pop("_anchors", None)
    matching_images = []
    seen_images = set()
    for entry in matching_entries:
        image_name = entry["image"]
        if image_name in seen_images:
            continue
        seen_images.add(image_name)
        matching_images.append(image_name)
    analysis["matching_images"] = matching_images
    return analysis


def detect_for_video(video_path, force=False):
    del force
    ocr_dir = existing_ocr_dir(video_path)
    images_dir = existing_images_dir(video_path)
    source = location_path(ocr_dir)

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
    print(f"[ok] {video_path.name}: has_subtitles={str(has_subtitles).lower()}", flush=True)
    return details
