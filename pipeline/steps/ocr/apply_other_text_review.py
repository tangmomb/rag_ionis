from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import existing_ocr_dir, relative_to_video_dir


SOURCE_FILTERED_NAME = "02_filtered_ocr_overlays.json"
SOURCE_REVIEW_DIRNAME = "other_text_gpt_review"
LEGACY_SOURCE_REVIEW_DIRNAME = "ocr_processed_filtered_others_boxes_review"
SOURCE_REVIEW_SUMMARY_NAME = "review_summary.json"
LEGACY_SOURCE_REVIEW_SUMMARY_NAME = "summary.json"
OUTPUT_FILTERED_NAME = "03_reviewed_ocr_overlays.json"

def filtered_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    return video_ocr_dir / SOURCE_FILTERED_NAME


def review_summary_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / SOURCE_REVIEW_DIRNAME / SOURCE_REVIEW_SUMMARY_NAME
    legacy = video_ocr_dir / LEGACY_SOURCE_REVIEW_DIRNAME / LEGACY_SOURCE_REVIEW_SUMMARY_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def output_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    return video_ocr_dir / OUTPUT_FILTERED_NAME


def load_json(path):
    return read_json(path)


def review_items_by_entry_id(summary_payload):
    indexed = {}
    for item in summary_payload.get("items", []):
        entry_id = str(item.get("entry_id", "")).strip()
        if entry_id:
            indexed[entry_id] = item
            continue
        timecode = str(item.get("timecode", "")).strip()
        if timecode:
            indexed[timecode] = item
    return indexed


def apply_review_to_filtered(filtered_payload, summary_payload, video_path):
    review_by_entry_id = review_items_by_entry_id(summary_payload)
    kinds = filtered_payload.get("kinds", {})
    kinds_details = filtered_payload.get("kinds_details", {})
    others = dict(kinds.get("others", {}))
    others_details = dict(kinds_details.get("others", {}))

    removed = []
    corrected = []
    untouched = []

    for entry_id, text in list(others.items()):
        review = review_by_entry_id.get(entry_id)
        if not review:
            untouched.append(entry_id)
            continue

        detail = others_details.get(entry_id)
        if not isinstance(detail, dict):
            detail = {"timecode": entry_id, "text": text}
            others_details[entry_id] = detail

        is_added = bool(review.get("is_added_in_edit"))
        corrected_text = " ".join(str(review.get("corrected_text", "")).split()).strip()
        has_ocr_error = bool(review.get("has_ocr_error"))

        if not is_added:
            others.pop(entry_id, None)
            others_details.pop(entry_id, None)
            removed.append(entry_id)
            continue

        if has_ocr_error and corrected_text:
            others[entry_id] = corrected_text
            detail["text"] = corrected_text
            corrected.append(entry_id)

        detail["review"] = {
            "has_ocr_error": has_ocr_error,
            "corrected_text": corrected_text or str(detail.get("text", text)),
            "is_added_in_edit": is_added,
            "confidence": review.get("confidence"),
            "reason": review.get("reason"),
            "review_dir": review.get("review_dir"),
            "crop": review.get("crop"),
        }

        if entry_id not in corrected:
            untouched.append(entry_id)

    filtered_payload["kinds"]["others"] = others
    filtered_payload["kinds_details"]["others"] = others_details
    filtered_payload["others_review"] = {
        "source_summary": relative_to_video_dir(review_summary_path(video_path), video_path),
        "output_file": relative_to_video_dir(output_path(video_path), video_path),
        "reviewed_count": len(review_by_entry_id),
        "removed_count": len(removed),
        "corrected_count": len(corrected),
        "kept_count": len(others),
        "removed_entry_ids": removed,
        "corrected_entry_ids": corrected,
    }
    return filtered_payload


def apply_review(video_path, force=False):
    source_filtered = filtered_path(video_path)
    source_summary = review_summary_path(video_path)
    target = output_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source_filtered.exists():
        print(f"[skip] filtered introuvable: {source_filtered}")
        return None
    if not source_summary.exists():
        print(f"[skip] summary introuvable: {source_summary}")
        return None

    filtered_payload = load_json(source_filtered)
    summary_payload = load_json(source_summary)
    output_payload = apply_review_to_filtered(filtered_payload, summary_payload, video_path)
    write_json(target, output_payload)
    print(f"[ok] {target}")
    return target
