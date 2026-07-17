from pathlib import Path

from pipeline.support.json_io import read_json, write_json
from pipeline.support.paddle_ocr import (
    box_bounds,
    box_geometry,
    collapse_answer_overlay_items,
    collapse_graphic_sequence_items,
    deduplicate_items,
    filter_decor_items,
    graphic_kind_for_image,
    image_size,
    mark_last_graphic_sequence_as_outro,
    ocr_items_from_raw_result,
    refine_subtitle_kinds,
    seconds_from_image_name,
    subtitle_text_signal,
)
from pipeline.support.paths import existing_images_dir, existing_ocr_dir


DEFAULT_MIN_CONFIDENCE = 0.9
RAW_GROUPS = ("footage", "graphic", "mixture")
PROCESSED_NAME = "01_processed_ocr_items.json"


def raw_paths(ocr_dir):
    paths = {}
    for group_name in RAW_GROUPS:
        preferred = ocr_dir / "raw" / f"raw_ocr_{group_name}_frames.json"
        legacy = ocr_dir / f"ocr_{group_name}.json"
        paths[group_name] = legacy if legacy.exists() and not preferred.exists() else preferred
    return paths


def processed_path(ocr_dir):
    return ocr_dir / PROCESSED_NAME


def load_json(path):
    return read_json(path)


def sort_key(item):
    image_name = str(item.get("image", ""))
    parsed_second = seconds_from_image_name(Path(image_name).name)
    return (
        parsed_second if parsed_second is not None else float("inf"),
        image_name,
    )


def strip_subtitle_kind(items, images_dir):
    sizes = {}
    sanitized = []
    for item in items:
        if item.get("kind") != "subtitle":
            sanitized.append(item)
            continue

        image_name = item.get("image")
        if image_name and image_name not in sizes:
            sizes[image_name] = image_size(images_dir / image_name)
        geometry = box_geometry(box_bounds(item.get("box")), sizes.get(image_name))
        _word_count, _has_sentence_punctuation, _has_subtitle_signal = subtitle_text_signal(
            item.get("text", ""),
            geometry["relative_width"],
        )
        replacement = dict(item)
        if graphic_kind_for_image(image_name) == "graphic":
            replacement["kind"] = "graphic"
        else:
            replacement["kind"] = "others"
        sanitized.append(replacement)
    return sanitized


def normalize_output_kinds(items):
    normalized_items = []
    for item in items:
        replacement = dict(item)
        if item.get("kind") == "subtitle":
            replacement["kind"] = "subtitle"
        elif graphic_kind_for_image(item.get("image")) == "graphic":
            replacement["kind"] = "graphic"
        else:
            replacement["kind"] = "others"
        normalized_items.append(replacement)
    return normalized_items


def write_outputs(ocr_dir, result):
    json_path = processed_path(ocr_dir)
    write_json(json_path, result)
    print(f"[write] {len(result.get('items', []))} items -> {json_path}", flush=True)


def process_video(video_path, min_confidence_override=None, force=False, strip_subtitles=False):
    images_dir = existing_images_dir(video_path)
    ocr_dir = existing_ocr_dir(video_path)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    sources = raw_paths(ocr_dir)
    target = processed_path(ocr_dir)
    if target.exists() and not force:
        print(f"[skip] {video_path.name}: {target.name} existe deja")
        return False

    available_sources = {group_name: path for group_name, path in sources.items() if path.exists()}
    if not available_sources:
        print(f"[skip] {video_path.name}: OCR brut introuvable dans: {ocr_dir}")
        return False

    min_confidence = min_confidence_override
    source_names = []
    raw_items = []
    for group_name in RAW_GROUPS:
        source = available_sources.get(group_name)
        if source is None:
            continue
        payload = load_json(source)
        source_names.append(source.name)
        if min_confidence is None:
            min_confidence = float(payload.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
        raw_items.extend(payload.get("items", []))
    if min_confidence is None:
        min_confidence = DEFAULT_MIN_CONFIDENCE

    items = []
    raw_items.sort(key=sort_key)
    for index, raw_item in enumerate(raw_items, start=1):
        image_name = raw_item.get("image")
        if not image_name:
            continue
        image_path = images_dir / image_name
        image_items = ocr_items_from_raw_result(
            raw_item.get("raw"),
            image_name=image_name,
            image_path=image_path,
            min_confidence=min_confidence,
        )
        items.extend(image_items)
        print(f"[process {index}/{len(raw_items)}] {video_path.name} {image_name}: {len(image_items)} texte(s)", flush=True)

    refined_items = refine_subtitle_kinds(items, images_dir)
    if strip_subtitles:
        refined_items = strip_subtitle_kind(refined_items, images_dir)
    filtered_items = filter_decor_items(refined_items, images_dir)
    graphic_collapsed_items = collapse_graphic_sequence_items(filtered_items)
    outro_marked_items = mark_last_graphic_sequence_as_outro(graphic_collapsed_items, images_dir)
    answer_collapsed_items = collapse_answer_overlay_items(outro_marked_items, images_dir)
    processed_items = deduplicate_items(answer_collapsed_items, images_dir)
    processed_items = normalize_output_kinds(processed_items)
    processed_items.sort(key=sort_key)
    write_outputs(
        ocr_dir,
        {
            "sources": source_names,
            "min_confidence": min_confidence,
            "items": processed_items,
        },
    )
    print(f"[done] {video_path.name}: {len(items)} items intermediaires", flush=True)
    return True
