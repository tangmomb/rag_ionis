import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.support.paths import existing_ocr_dir


OCR_DIR_NAME = "ocr"
OCR_PROCESSED_NAME = "01_processed_ocr_items.json"
OCR_PROCESSED_CORRECTED_NAME = "corrected_ocr_items.json"
OCR_PROCESSED_FILTERED_NAME = "02_filtered_ocr_overlays.json"
LEGACY_OCR_PROCESSED_CORRECTED_NAME = "ocr_processed_corrected.json"
SUBTITLE_REPEAT_IMAGE_WINDOW = 10
SUBTITLE_NEIGHBOR_IMAGE_GAP = 3
MIN_OVERLAY_SCORE = 0.9
GRAPHIC_SEQUENCE_GAP_SECONDS = 5
GRAPHIC_COUSIN_MIN_SIMILARITY = 0.62
GRAPHIC_COUSIN_MIN_TEXT_LENGTH = 6
ON_FOOTAGE_SEQUENCE_GAP_SECONDS = 1
OTHERS_PROGRESSION_IMAGE_GAP = 10
OVERLAY_KINDS = {"name", "lower_third", "title"}
IONIS_SCHOOL_TEXT = "IONIS SCHOOL OF TECHNOLOGY AND MANAGEMENT"
PLANETE_METIERS_TEXT = "PLANÈTE MÉTIERS"
LONG_BINARY_TOKEN = re.compile(r"\b[01]{10,}[A-Z]?\b")
DECIMAL_NOISE_TOKEN = re.compile(r"(?<!\w)\d+\.\.?\d+(?!\w)")
ALPHA_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def normalize_text(text):
    return re.sub(r"\W+", "", str(text).casefold())


def fold_text(text):
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\W+", "", without_marks)


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
    return normalized in OVERLAY_KINDS or normalized == "graphic"


def is_graphic_kind(kind):
    normalized = normalize_kind(kind)
    return normalized == "graphic"


def sanitize_graphic_overlay_text(text):
    normalized = " ".join(str(text or "").split())
    folded = fold_text(normalized)
    if fold_text(IONIS_SCHOOL_TEXT) in folded:
        if fold_text(PLANETE_METIERS_TEXT) not in folded:
            return None
        return f"{IONIS_SCHOOL_TEXT} — {PLANETE_METIERS_TEXT}"

    decimal_tokens = DECIMAL_NOISE_TOKEN.findall(normalized)
    meaningful_words = [
        word
        for word in ALPHA_WORD.findall(normalized)
        if word.casefold() != "psi"
    ]
    if (
        LONG_BINARY_TOKEN.search(normalized)
        or "\u25b2" in normalized
        or len(decimal_tokens) >= 3
        or (decimal_tokens and not meaningful_words)
    ):
        return None

    return re.sub(
        r"\blonis-STM\b",
        "Ionis-STM",
        normalized,
        flags=re.IGNORECASE,
    )


def overlay_label_key(item):
    if is_graphic_kind(item.get("kind")):
        return "graphic"
    return "on_footage"


def merge_question_parts(items):
    return list(items)


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


def text_extends_or_repeats(previous_text, candidate_text):
    previous_normalized = fold_text(previous_text)
    candidate_normalized = fold_text(candidate_text)
    if not previous_normalized or not candidate_normalized:
        return False
    if previous_normalized == candidate_normalized:
        return True
    if len(previous_normalized) < 2 or len(candidate_normalized) <= len(previous_normalized):
        return False
    return candidate_normalized.startswith(previous_normalized)


def edit_distance_at_most(left, right, limit=2):
    if left == right:
        return True
    if abs(len(left) - len(right)) > limit:
        return False

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return False
        previous = current
    return previous[-1] <= limit


def are_minor_ocr_variants(left_text, right_text, max_distance=2):
    left_folded = fold_text(left_text)
    right_folded = fold_text(right_text)
    if not left_folded or not right_folded:
        return False
    if left_folded == right_folded:
        return True
    if len(left_folded) < 6 or len(right_folded) < 6:
        return False
    return edit_distance_at_most(left_folded, right_folded, limit=max_distance)


def graphic_texts_are_cousins(left_text, right_text):
    if texts_are_progressive(left_text, right_text):
        return True
    if are_minor_ocr_variants(left_text, right_text):
        return True

    left_folded = fold_text(left_text)
    right_folded = fold_text(right_text)
    if (
        min(len(left_folded), len(right_folded))
        < GRAPHIC_COUSIN_MIN_TEXT_LENGTH
    ):
        return False
    return (
        SequenceMatcher(None, left_folded, right_folded).ratio()
        >= GRAPHIC_COUSIN_MIN_SIMILARITY
    )


def other_entry_quality(entry):
    text = str(entry.get("text", ""))
    normalized = fold_text(text)
    spaced_words = len(text.split())
    has_spacing = 1 if " " in text else 0
    has_separator = 1 if any(char in text for char in {"&", "-", "/"} ) else 0
    return (
        len(normalized),
        spaced_words,
        has_spacing,
        has_separator,
        -entry["second"],
    )


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


def graphic_cousin_components(image_groups, max_gap_seconds):
    texts = [joined_group_text(group) for group in image_groups]
    seconds = [
        min(float(item.get("second", 0)) for item in group)
        for group in image_groups
    ]
    remaining = set(range(len(image_groups)))
    components = []

    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component_indexes = [seed]
        pending = [seed]

        while pending:
            current = pending.pop()
            cousins = [
                candidate
                for candidate in sorted(remaining)
                if abs(seconds[candidate] - seconds[current]) < max_gap_seconds
                and graphic_texts_are_cousins(
                    texts[current],
                    texts[candidate],
                )
            ]
            for cousin in cousins:
                remaining.remove(cousin)
                component_indexes.append(cousin)
                pending.append(cousin)

        components.append([image_groups[index] for index in component_indexes])

    return components


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
        image_groups = sorted(
            by_image.values(),
            key=lambda group: min(float(item["second"]) for item in group),
        )
        supported_components = [
            component
            for component in graphic_cousin_components(
                image_groups,
                min_separator_seconds,
            )
            if len(component) >= 2
        ]
        selected_groups = [
            max(component, key=image_group_score)
            for component in supported_components
        ]
        selected_groups.sort(
            key=lambda group: min(float(item["second"]) for item in group)
        )
        for selected_items in selected_groups:
            collapsed.extend(selected_items)
        visual_group = []

    for item in items:
        if item.get("kind") != "graphic":
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
        if kind not in OVERLAY_KINDS:
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
    video_ocr_dir = existing_ocr_dir(video_path)
    corrected = video_ocr_dir / OCR_PROCESSED_CORRECTED_NAME
    legacy_corrected = video_ocr_dir / LEGACY_OCR_PROCESSED_CORRECTED_NAME
    if corrected.exists():
        return corrected
    if legacy_corrected.exists():
        return legacy_corrected
    return video_ocr_dir / OCR_PROCESSED_NAME


def filtered_ocr_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    return video_ocr_dir / OCR_PROCESSED_FILTERED_NAME


def enriched_ocr_source_path(video_path):
    return filtered_ocr_path(video_path)


def load_overlay_items(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = []
    seen_image_texts = set()
    for position, item in enumerate(payload.get("items", [])):
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
                "order": position,
            }
        )
    return items, payload


def load_groupable_items(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for position, item in enumerate(payload.get("items", [])):
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
                "order": position,
            }
        )
    return items, payload


def filter_overlay_items(items):
    filtered_items = merge_question_parts(sorted(items, key=lambda item: item["second"]))
    filtered_items = collapse_on_footage_progressions(filtered_items)
    filtered_items = remove_overlay_fragments(filtered_items)
    filtered_items = collapse_graphic_time_groups(filtered_items)
    filtered_items = merge_same_second_overlays(filtered_items)
    sanitized_items = []
    for item in filtered_items:
        if is_graphic_kind(item.get("kind")):
            sanitized_text = sanitize_graphic_overlay_text(item.get("text"))
            if sanitized_text is None:
                continue
            item = {**item, "text": sanitized_text}
        sanitized_items.append(item)
    filtered_items = remove_exact_overlay_duplicates(sanitized_items)
    return [
        item
        for item in filtered_items
        if not (
            is_graphic_kind(item.get("kind"))
            and "www" in str(item.get("text", "")).casefold()
        )
    ]


def item_sort_key(item):
    return (item["second"], item.get("order", float("inf")), item.get("image", ""), item["text"])


def image_order_map(items):
    order = {}
    for item in sorted(items, key=item_sort_key):
        image_name = item.get("image")
        if image_name and image_name not in order:
            order[image_name] = len(order)
    return order


def filter_subtitle_entries(entries, image_orders):
    sorted_entries = sorted(entries, key=item_sort_key)
    kept = []
    last_kept_index_by_text = {}

    for entry in sorted_entries:
        normalized = normalize_text(entry["text"])
        if not normalized:
            continue

        image_name = entry.get("image")
        image_index = image_orders.get(image_name)
        last_kept_position = last_kept_index_by_text.get(normalized)

        if last_kept_position is None:
            kept.append(entry)
            if image_index is not None:
                last_kept_index_by_text[normalized] = len(kept) - 1
            continue

        previous_entry = kept[last_kept_position]
        previous_image_index = image_orders.get(previous_entry.get("image"))
        if (
            image_index is not None
            and previous_image_index is not None
            and image_index - previous_image_index <= SUBTITLE_REPEAT_IMAGE_WINDOW
        ):
            if subtitle_entry_quality(entry) >= subtitle_entry_quality(previous_entry):
                kept[last_kept_position] = entry
            continue

        kept.append(entry)
        if image_index is not None:
            last_kept_index_by_text[normalized] = len(kept) - 1

    return kept


def subtitle_entry_quality(entry):
    text = str(entry.get("text", ""))
    normalized = fold_text(text)
    return (
        len(normalized),
        len(text),
        len(text.split()),
        entry["second"],
    )


def are_subtitle_neighbor_duplicates(left, right, image_orders):
    left_index = image_orders.get(left.get("image"))
    right_index = image_orders.get(right.get("image"))
    if left_index is None or right_index is None:
        if abs(right["second"] - left["second"]) > 1:
            return False
    elif not (0 < right_index - left_index <= SUBTITLE_NEIGHBOR_IMAGE_GAP):
        return False

    left_text = left.get("text", "")
    right_text = right.get("text", "")
    left_folded = fold_text(left_text)
    right_folded = fold_text(right_text)
    if not left_folded or not right_folded:
        return False
    if left_folded == right_folded:
        return True
    if are_minor_ocr_variants(left_text, right_text):
        return True
    shorter, longer = sorted((left_folded, right_folded), key=len)
    return len(shorter) >= 8 and shorter in longer


def filter_subtitle_neighbor_duplicates(entries, image_orders):
    if not entries:
        return entries

    sorted_entries = sorted(entries, key=item_sort_key)
    kept = []
    index = 0

    while index < len(sorted_entries):
        current = sorted_entries[index]
        best_entry = current
        next_index = index + 1

        while next_index < len(sorted_entries):
            candidate = sorted_entries[next_index]
            if not are_subtitle_neighbor_duplicates(best_entry, candidate, image_orders):
                break
            if subtitle_entry_quality(candidate) >= subtitle_entry_quality(best_entry):
                best_entry = candidate
            next_index += 1

        kept.append(best_entry)
        index = next_index

    return kept


def filter_other_progressive_entries(entries, image_orders, max_image_gap=OTHERS_PROGRESSION_IMAGE_GAP):
    if not entries:
        return entries

    sorted_entries = sorted(entries, key=item_sort_key)
    kept = []
    consumed = set()

    def within_gap(left, right):
        left_index = image_orders.get(left.get("image"))
        right_index = image_orders.get(right.get("image"))
        if left_index is not None and right_index is not None:
            return 0 < right_index - left_index <= max_image_gap
        return 0 < right["second"] - left["second"] <= max_image_gap

    for index, entry in enumerate(sorted_entries):
        if index in consumed:
            continue

        run_indexes = [index]
        run_changed = True
        while run_changed:
            run_changed = False
            for candidate_index in range(run_indexes[-1] + 1, len(sorted_entries)):
                if candidate_index in run_indexes:
                    continue
                candidate = sorted_entries[candidate_index]
                if any(
                    within_gap(sorted_entries[run_index], candidate)
                    and text_extends_or_repeats(sorted_entries[run_index]["text"], candidate["text"])
                    for run_index in run_indexes
                ):
                    run_indexes.append(candidate_index)
                    run_changed = True

        run = [sorted_entries[run_index] for run_index in run_indexes]
        if len(run) == 1:
            kept.append(run[0])
            consumed.add(index)
            continue

        best_entry = max(
            run,
            key=lambda candidate: (
                len(normalize_text(candidate["text"])),
                len(candidate["text"].split()),
                candidate["second"],
            ),
        )
        last_entry = run[-1]
        kept.append(best_entry)
        if last_entry is not best_entry:
            kept.append(last_entry)
        consumed.update(run_indexes)

    return sorted(kept, key=item_sort_key)


def filter_other_minor_variants(entries, image_orders, max_image_gap=OTHERS_PROGRESSION_IMAGE_GAP):
    if not entries:
        return entries

    sorted_entries = sorted(entries, key=item_sort_key)
    kept = []
    consumed = set()

    def within_gap(left, right):
        left_index = image_orders.get(left.get("image"))
        right_index = image_orders.get(right.get("image"))
        if left_index is not None and right_index is not None:
            return abs(right_index - left_index) <= max_image_gap
        return abs(right["second"] - left["second"]) <= max_image_gap

    for index, entry in enumerate(sorted_entries):
        if index in consumed:
            continue

        variant_indexes = [index]
        for candidate_index in range(index + 1, len(sorted_entries)):
            if candidate_index in consumed:
                continue
            candidate = sorted_entries[candidate_index]
            if not within_gap(entry, candidate):
                if candidate["second"] - entry["second"] > max_image_gap:
                    break
                continue
            if are_minor_ocr_variants(entry["text"], candidate["text"]):
                variant_indexes.append(candidate_index)
                continue
            if text_extends_or_repeats(entry["text"], candidate["text"]) or text_extends_or_repeats(candidate["text"], entry["text"]):
                continue

        best_entry = max((sorted_entries[i] for i in variant_indexes), key=other_entry_quality)
        kept.append(best_entry)
        consumed.update(variant_indexes)

    return sorted(kept, key=item_sort_key)


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
            entries = filter_subtitle_neighbor_duplicates(entries, image_orders)
        elif kind == "others":
            entries = filter_other_progressive_entries(entries, image_orders)
            entries = filter_other_minor_variants(entries, image_orders)
        values = {}
        details = {}
        timecode_counts = defaultdict(int)
        for entry in entries:
            timecode = format_timecode(entry["second"])
            text = entry["text"]
            if kind == "others":
                previous = values.get(timecode)
                if previous is None:
                    values[timecode] = text
                    details[timecode] = {
                        "timecode": timecode,
                        "text": text,
                        "texts": [text],
                        "image": entry.get("image"),
                        "images": [entry.get("image")],
                        "box": entry.get("box"),
                        "boxes": [entry.get("box")],
                    }
                else:
                    values[timecode] = f"{previous} / {text}"
                    detail = details[timecode]
                    detail["text"] = values[timecode]
                    detail.setdefault("texts", []).append(text)
                    detail.setdefault("images", []).append(entry.get("image"))
                    detail.setdefault("boxes", []).append(entry.get("box"))
                continue

            timecode_counts[timecode] += 1
            entry_key = timecode
            if timecode_counts[timecode] > 1:
                entry_key = f"{timecode}#{timecode_counts[timecode]}"
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
            "others_progression_image_gap": OTHERS_PROGRESSION_IMAGE_GAP,
        },
        "kinds": grouped_kinds,
        "kinds_details": grouped_kind_details,
        "source_item_count": len(source_payload.get("items", [])),
        "filtered_item_count": len(filtered_items),
    }
