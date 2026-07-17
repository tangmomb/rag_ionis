from pipeline.support.json_io import read_json, write_json
from pipeline.support.ocr_filtering import (
    build_filtered_payload,
    filter_overlay_items,
    filtered_ocr_path,
    group_items_by_kind,
    load_groupable_items,
    load_overlay_items,
    normalize_text,
    processed_ocr_source_path,
)
from pipeline.support.paths import existing_ocr_dir, relative_to_video_dir


OCR_FOOTAGE_NAME = "raw_ocr_footage_frames.json"
LEGACY_OCR_FOOTAGE_NAME = "ocr_footage.json"


def footage_ocr_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / "raw" / OCR_FOOTAGE_NAME
    legacy = video_ocr_dir / LEGACY_OCR_FOOTAGE_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_footage_occurrences(path):
    payload = read_json(path)
    occurrences_by_text = {}

    for item in payload.get("items", []):
        image_name = item.get("image")
        if not image_name:
            continue
        for raw in item.get("raw", []):
            polys = list(raw.get("dt_polys", []))
            texts = list(raw.get("rec_texts", []))
            scores = list(raw.get("rec_scores", []))
            count = min(len(polys), len(texts))
            for index in range(count):
                text = " ".join(str(texts[index]).split()).strip()
                normalized = normalize_text(text)
                if not normalized:
                    continue
                box = polys[index]
                occurrence = {
                    "image": image_name,
                    "text": text,
                    "box": box,
                }
                if index < len(scores):
                    try:
                        occurrence["score"] = float(scores[index])
                    except (TypeError, ValueError):
                        pass
                occurrences_by_text.setdefault(normalized, []).append(occurrence)

    return occurrences_by_text


def enrich_others_with_footage_occurrences(grouped_kind_details, video_path):
    others_details = grouped_kind_details.get("others")
    if not isinstance(others_details, dict) or not others_details:
        return

    footage_path = footage_ocr_path(video_path)
    if not footage_path.exists():
        print(f"[skip] OCR footage introuvable: {footage_path}")
        return

    occurrences_by_text = load_footage_occurrences(footage_path)
    for timecode, detail in others_details.items():
        if not isinstance(detail, dict):
            continue
        text = " ".join(str(detail.get("text", "")).split()).strip()
        normalized = normalize_text(text)
        occurrences = list(occurrences_by_text.get(normalized, []))
        detail["all_occurrences"] = occurrences
        detail["all_occurrence_count"] = len(occurrences)
        detail["all_occurrences_source"] = relative_to_video_dir(footage_path, video_path)


def filter_processed_ocr(video_path, force=False):
    source = processed_ocr_source_path(video_path)
    target = filtered_ocr_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] OCR processed introuvable: {source}")
        return None

    all_items, payload = load_groupable_items(source)
    overlay_items, _ = load_overlay_items(source)
    filtered_items = filter_overlay_items(overlay_items)
    grouped_kinds, grouped_kind_details = group_items_by_kind(all_items, filtered_items)
    enrich_others_with_footage_occurrences(grouped_kind_details, video_path)
    filtered_payload = build_filtered_payload(
        payload,
        source.name,
        filtered_items,
        grouped_kinds,
        grouped_kind_details,
    )
    write_json(target, filtered_payload)
    print(f"[ok] {target}")
    return target
