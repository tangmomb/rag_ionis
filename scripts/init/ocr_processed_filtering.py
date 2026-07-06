import json
import re
from collections import defaultdict
from pathlib import Path


OCR_DIR_NAME = "ocr"
OCR_PROCESSED_NAME = "ocr_processed.json"
OCR_PROCESSED_CORRECTED_NAME = "ocr_processed_corrected.json"
OCR_PROCESSED_FILTERED_NAME = "ocr_processed_filtered.json"
SUBTITLE_REPEAT_IMAGE_WINDOW = 10
MIN_OVERLAY_SCORE = 0.9
GRAPHIC_SEQUENCE_GAP_SECONDS = 5
ON_FOOTAGE_SEQUENCE_GAP_SECONDS = 1
OVERLAY_KINDS = {"name", "lower_third", "question_intertitle", "title"}


def normalize_text(text):
    return re.sub(r"\W+", "", str(text).casefold())


def normalize_kind(kind):
    return str(kind or "").strip().lower()


def parse_timecode(value):
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def parse_image_second(image_name):
    stem = Path(str(image_name or "")).stem
    parts = stem.split("_")
    if len(parts) == 2:
        minutes, seconds = parts
        milliseconds = 0
    elif len(parts) == 3:
        minutes, seconds, milliseconds = parts
    elif len(parts) == 4:
        hours, minutes, seconds, milliseconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    else:
        return None
    return int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def format_timecode(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_precise_timecode(seconds):
    total_milliseconds = int(round(float(seconds or 0) * 1000))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        base = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        base = f"{minutes:02d}:{seconds:02d}"
    if milliseconds:
        return f"{base}.{milliseconds:03d}"
    return base


def confidence_score(item):
    try:
        return float(item.get("score"))
    except (TypeError, ValueError):
        pass

    label = str(item.get("confidence", "")).strip().lower()
    if label == "high":
        return 1.0
    if label == "medium":
        return 0.7
    if label == "low":
        return 0.4
    return 0.0


def is_overlay_kind(kind):
    normalized = normalize_kind(kind)
    return normalized in OVERLAY_KINDS or normalized in {"graphic", "outro"}


def is_graphic_kind(kind):
    normalized = normalize_kind(kind)
    return normalized in {"graphic", "outro"}


def overlay_label_key(item):
    if item.get("kind") == "question_intertitle":
        return "question_intertitle"
    if item.get("kind") == "outro":
        return "outro"
    if is_graphic_kind(item.get("kind")):
        return "graphic"
    return "on_footage"


def merge_question_parts(items):
    merged = []
    for item in items:
        if (
            item.get("kind") == "question_intertitle"
            and merged
            and merged[-1].get("kind") != "question_intertitle"
            and item["second"] - merged[-1]["second"] <= 1
        ):
            previous = merged.pop()
            item = dict(item)
            item["second"] = previous["second"]
            item["text"] = f"{previous['text']} {item['text']}"
        merged.append(item)
    return merged


def remove_overlay_fragments(items):
    kept = []
    for index, item in enumerate(items):
        normalized = normalize_text(item["text"])
        is_fragment = False
        for other in items[index + 1 :]:
            if other["second"] - item["second"] > 4:
                break
            other_normalized = normalize_text(other["text"])
            if len(normalized) < 4 or len(other_normalized) <= len(normalized) + 1:
                continue
            if normalized and normalized in other_normalized:
                is_fragment = True
                break
        if not is_fragment:
            kept.append(item)
    return kept


def merge_same_second_overlays(items):
    merged = []
    for item in items:
        if (
            merged
            and item.get("second") == merged[-1].get("second")
            and overlay_label_key(item) == overlay_label_key(merged[-1])
        ):
            previous = merged[-1]
            texts = previous.setdefault("_texts", [previous["text"]])
            if item["text"] not in texts:
                texts.append(item["text"])
                separator = " " if overlay_label_key(previous) == "graphic" else " / "
                previous["text"] = separator.join(texts)
            continue

        item = dict(item)
        item["_texts"] = [item["text"]]
        merged.append(item)

    for item in merged:
        item.pop("_texts", None)
    return merged


def remove_exact_overlay_duplicates(items):
    deduped = []
    seen = set()
    for item in items:
        key = (
            overlay_label_key(item),
            format_timecode(item["second"]),
            normalize_text(item["text"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def image_group_score(items):
    joined_text = " ".join(" ".join(str(item.get("text", "")).split()) for item in items)
    normalized = normalize_text(joined_text)
    best_second = min(float(item.get("second", 0)) for item in items)
    return (len(normalized), len(joined_text.split()), -best_second)


def joined_group_text(items):
    return " ".join(" ".join(str(item.get("text", "")).split()) for item in items).strip()


def texts_are_progressive(text_a, text_b):
    normalized_a = normalize_text(text_a)
    normalized_b = normalize_text(text_b)
    if not normalized_a or not normalized_b:
        return False
    return normalized_a in normalized_b or normalized_b in normalized_a


def group_items_by_image(items):
    groups = []
    current_group = []

    for item in items:
        if (
            current_group
            and item.get("kind") == current_group[-1].get("kind")
            and item.get("image") == current_group[-1].get("image")
            and item.get("second") == current_group[-1].get("second")
        ):
            current_group.append(item)
            continue

        if current_group:
            groups.append(current_group)
        current_group = [item]

    if current_group:
        groups.append(current_group)
    return groups


def collapse_graphic_time_groups(items, min_separator_seconds=GRAPHIC_SEQUENCE_GAP_SECONDS):
    collapsed = []
    visual_group = []

    def flush_visual_group():
        nonlocal visual_group
        if not visual_group:
            return
        by_image = {}
        for item in visual_group:
            image_key = item.get("image") or f"__no_image__:{item['second']}:{item['text']}"
            by_image.setdefault(image_key, []).append(item)
        selected_items = max(by_image.values(), key=image_group_score)
        collapsed.extend(selected_items)
        visual_group = []

    for item in items:
        if item.get("kind") not in {"graphic", "outro"}:
            flush_visual_group()
            collapsed.append(item)
            continue

        if not visual_group:
            visual_group.append(item)
            continue

        same_kind = item.get("kind") == visual_group[-1].get("kind")
        close_enough = item["second"] - visual_group[-1]["second"] < min_separator_seconds
        if same_kind and close_enough:
            visual_group.append(item)
            continue

        flush_visual_group()
        visual_group.append(item)

    flush_visual_group()
    return collapsed


def collapse_on_footage_progressions(items, max_gap_seconds=ON_FOOTAGE_SEQUENCE_GAP_SECONDS):
    collapsed = []
    groups = group_items_by_image(items)
    index = 0

    while index < len(groups):
        group = groups[index]
        kind = group[0].get("kind")
        if kind not in OVERLAY_KINDS or kind == "question_intertitle":
            collapsed.extend(group)
            index += 1
            continue

        best_group = group
        best_text = joined_group_text(best_group)
        next_index = index + 1

        while next_index < len(groups):
            candidate = groups[next_index]
            candidate_kind = candidate[0].get("kind")
            gap = float(candidate[0]["second"]) - float(groups[next_index - 1][0]["second"])
            if candidate_kind != kind or gap > max_gap_seconds:
                break

            candidate_text = joined_group_text(candidate)
            if not texts_are_progressive(best_text, candidate_text):
                break

            if image_group_score(candidate) > image_group_score(best_group):
                best_group = candidate
                best_text = candidate_text
            next_index += 1

        collapsed.extend(best_group)
        index = next_index

    return collapsed


def processed_ocr_source_path(video_path):
    ocr_dir = video_path.parent / OCR_DIR_NAME
    corrected = ocr_dir / OCR_PROCESSED_CORRECTED_NAME
    if corrected.exists():
        return corrected
    return ocr_dir / OCR_PROCESSED_NAME


def filtered_ocr_path(video_path):
    ocr_dir = video_path.parent / OCR_DIR_NAME
    return ocr_dir / OCR_PROCESSED_FILTERED_NAME


def load_overlay_items(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = []
    seen_image_texts = set()
    for item in payload.get("items", []):
        kind = str(item.get("kind", "")).strip().lower()
        if kind == "subtitle" or (kind and not is_overlay_kind(kind)):
            continue
        if confidence_score(item) < MIN_OVERLAY_SCORE:
            continue
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        normalized = normalize_text(text)
        image_name = item.get("image")
        dedupe_key = (kind, image_name, normalized)
        if not normalized or dedupe_key in seen_image_texts:
            continue
        seen_image_texts.add(dedupe_key)
        second = item.get("second")
        if second is None:
            timecode = str(item.get("timecode", "")).strip()
            if timecode:
                second = parse_timecode(timecode)
            else:
                second = parse_image_second(image_name)
        if second is None:
            continue
        items.append(
            {
                "kind": kind,
                "second": float(second),
                "text": text,
                "image": image_name,
                "box": item.get("box"),
            }
        )
    return items, payload


def load_groupable_items(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for item in payload.get("items", []):
        kind = normalize_kind(item.get("kind"))
        text = " ".join(str(item.get("text", "")).split())
        if not kind or not text:
            continue
        second = item.get("second")
        if second is None:
            timecode = str(item.get("timecode", "")).strip()
            if timecode:
                second = parse_timecode(timecode)
            else:
                second = parse_image_second(item.get("image"))
        if second is None:
            continue
        items.append(
            {
                "kind": kind,
                "second": float(second),
                "text": text,
                "image": item.get("image"),
                "box": item.get("box"),
            }
        )
    return items, payload


def filter_overlay_items(items):
    filtered_items = merge_question_parts(sorted(items, key=lambda item: item["second"]))
    filtered_items = collapse_on_footage_progressions(filtered_items)
    filtered_items = remove_overlay_fragments(filtered_items)
    filtered_items = collapse_graphic_time_groups(filtered_items)
    return remove_exact_overlay_duplicates(merge_same_second_overlays(filtered_items))


def item_sort_key(item):
    return (item["second"], item.get("image", ""), item["text"])


def image_order_map(items):
    order = {}
    for item in sorted(items, key=item_sort_key):
        image_name = item.get("image")
        if image_name and image_name not in order:
            order[image_name] = len(order)
    return order


def filter_subtitle_entries(entries, image_orders):
    kept = []
    last_seen_by_text = {}
    for entry in sorted(entries, key=item_sort_key):
        normalized = normalize_text(entry["text"])
        if not normalized:
            continue
        image_name = entry.get("image")
        image_index = image_orders.get(image_name)
        last_seen = last_seen_by_text.get(normalized)
        if (
            last_seen is not None
            and image_index is not None
            and last_seen is not None
            and image_index - last_seen <= SUBTITLE_REPEAT_IMAGE_WINDOW
        ):
            continue
        kept.append(entry)
        if image_index is not None:
            last_seen_by_text[normalized] = image_index
    return kept


def group_items_by_kind(all_items, filtered_overlay_items):
    grouped = defaultdict(list)
    image_orders = image_order_map(all_items)
    for item in all_items:
        if is_overlay_kind(item.get("kind")):
            continue
        grouped[item["kind"]].append(item)
    for item in filtered_overlay_items:
        grouped[item["kind"]].append(item)

    serialized = {}
    serialized_details = {}
    for kind in sorted(grouped):
        entries = sorted(grouped[kind], key=item_sort_key)
        if kind == "subtitle":
            entries = filter_subtitle_entries(entries, image_orders)
        values = {}
        details = {}
        timecode_counts = defaultdict(int)
        for entry in entries:
            timecode = format_timecode(entry["second"])
            timecode_counts[timecode] += 1
            entry_key = timecode
            if timecode_counts[timecode] > 1:
                entry_key = f"{timecode}#{timecode_counts[timecode]}"
            text = entry["text"]
            previous = values.get(entry_key)
            if previous is None:
                values[entry_key] = text
                details[entry_key] = {
                    "timecode": timecode,
                    "text": text,
                    "image": entry.get("image"),
                    "box": entry.get("box"),
                }
                continue
            continue
        serialized[kind] = values
        serialized_details[kind] = details
    return serialized, serialized_details


def build_filtered_payload(source_payload, source_name, filtered_items, grouped_kinds, grouped_kind_details):
    return {
        "source": source_name,
        "filtering": {
            "min_overlay_score": MIN_OVERLAY_SCORE,
            "graphic_sequence_gap_seconds": GRAPHIC_SEQUENCE_GAP_SECONDS,
            "on_footage_sequence_gap_seconds": ON_FOOTAGE_SEQUENCE_GAP_SECONDS,
        },
        "kinds": grouped_kinds,
        "kinds_details": grouped_kind_details,
        "source_item_count": len(source_payload.get("items", [])),
        "filtered_item_count": len(filtered_items),
    }
