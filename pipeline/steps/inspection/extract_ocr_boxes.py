from pathlib import Path

from pipeline.support.json_io import read_json, write_json
from pipeline.support.paddle_ocr import (
    box_text_score_records_from_raw_result,
    seconds_from_image_name,
)
from pipeline.support.paths import existing_ocr_dir


RAW_GROUPS = ("footage", "graphic", "mixture")
LOCATION_NAME = "ocr_box_locations.json"
LEGACY_LOCATION_NAME = "ocr_location.json"
DEFAULT_MIN_CONFIDENCE = 0.9


def raw_paths(ocr_dir):
    paths = {}
    for group_name in RAW_GROUPS:
        preferred = ocr_dir / "raw" / f"raw_ocr_{group_name}_frames.json"
        legacy = ocr_dir / f"ocr_{group_name}.json"
        paths[group_name] = legacy if legacy.exists() and not preferred.exists() else preferred
    return paths


def location_path(ocr_dir):
    preferred = ocr_dir / LOCATION_NAME
    legacy = ocr_dir / LEGACY_LOCATION_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_json(path):
    return read_json(path)


def sort_key(item):
    image_name = str(item.get("image", ""))
    parsed_second = seconds_from_image_name(Path(image_name).name)
    return (
        parsed_second if parsed_second is not None else float("inf"),
        image_name,
    )


def write_outputs(ocr_dir, result):
    json_path = location_path(ocr_dir)
    write_json(json_path, result)
    print(f"[write] location -> {json_path}", flush=True)


def extract_for_video(video_path, *, force=False):
    ocr_dir = existing_ocr_dir(video_path)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    sources = raw_paths(ocr_dir)
    target = location_path(ocr_dir)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    available_sources = {
        group_name: path
        for group_name, path in sources.items()
        if path.exists()
    }
    if not available_sources:
        print(f"[skip] OCR brut introuvable dans: {ocr_dir}")
        return None

    box_items = []
    total_boxes = 0
    source_names = []
    total_images = 0
    for group_name in RAW_GROUPS:
        source = available_sources.get(group_name)
        if source is None:
            continue
        payload = load_json(source)
        raw_items = payload.get("items", [])
        source_names.append(source.name)
        for index, raw_item in enumerate(raw_items, start=1):
            image_name = raw_item.get("image")
            if not image_name:
                continue
            records = box_text_score_records_from_raw_result(raw_item.get("raw"))
            boxes = [record["box"] for record in records]
            texts = [record["text"] for record in records]
            scores = [record["score"] for record in records]
            total_boxes += len(boxes)
            total_images += 1
            box_items.append(
                {
                    "image": image_name,
                    "boxes": boxes,
                    "texts": texts,
                    "scores": scores,
                }
            )
            print(
                f"[boxes {group_name} {index}/{len(raw_items)}] {image_name}: {len(boxes)} box(es)",
                flush=True,
            )

    box_items.sort(key=sort_key)
    write_outputs(
        ocr_dir,
        {
            "sources": source_names,
            "min_confidence": DEFAULT_MIN_CONFIDENCE,
            "items": box_items,
        },
    )
    print(f"[done] {video_path.name}: {total_boxes} box(es)", flush=True)
    return target
