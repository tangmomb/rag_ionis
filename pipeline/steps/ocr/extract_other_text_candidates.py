import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.support.json_io import read_json, write_json
from pipeline.support.ocr_filtering import filtered_ocr_path
from pipeline.support.paddle_ocr import box_bounds, image_files
from pipeline.support.paths import existing_images_dir, existing_ocr_dir, relative_to_video_dir


OUTPUT_DIRNAME = "other_text_review_candidates"
LEGACY_OUTPUT_DIRNAME = "ocr_processed_filtered_others_boxes"
MANIFEST_NAME = "review_candidates_manifest.json"
LEGACY_MANIFEST_NAME = "manifest.json"
BOX_PADDING_PX = 10
BOX_OUTLINE_COLOR = (255, 0, 0)
BOX_OUTLINE_WIDTH = 4


def output_dir(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / OUTPUT_DIRNAME
    legacy = video_ocr_dir / LEGACY_OUTPUT_DIRNAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def manifest_path(video_path):
    directory = output_dir(video_path)
    preferred = directory / MANIFEST_NAME
    legacy = directory / LEGACY_MANIFEST_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_others_entries(path):
    payload = read_json(path)
    details = payload.get("kinds_details", {}).get("others", {})
    entries = []
    for entry_id, item in sorted(details.items()):
        if not isinstance(item, dict):
            continue
        image_name = item.get("image")
        box = item.get("box")
        image_names = list(item.get("images", []))
        boxes = list(item.get("boxes", []))
        texts = list(item.get("texts", []))
        if not image_names and image_name:
            image_names = [image_name]
        if not boxes and box:
            boxes = [box]
        if not texts and item.get("text"):
            texts = [item.get("text", "")]
        annotations = []
        for annotation_index, annotation_box in enumerate(boxes):
            annotation_image = image_names[annotation_index] if annotation_index < len(image_names) else image_name
            annotation_text = texts[annotation_index] if annotation_index < len(texts) else item.get("text", "")
            if not annotation_image or not annotation_box:
                continue
            annotations.append(
                {
                    "image": annotation_image,
                    "box": annotation_box,
                    "text": annotation_text,
                }
            )
        if not annotations:
            continue
        entries.append(
            {
                "entry_id": entry_id,
                "timecode": item.get("timecode", entry_id),
                "text": item.get("text", ""),
                "image": image_name or annotations[0]["image"],
                "box": box or annotations[0]["box"],
                "annotations": annotations,
                "all_occurrences": list(item.get("all_occurrences", [])),
            }
        )
    return entries


def padded_box_bounds(poly, image_size):
    bounds = box_bounds(poly)
    if not bounds:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = bounds
    left = max(0, min(width, int(x1) - BOX_PADDING_PX))
    top = max(0, min(height, int(y1) - BOX_PADDING_PX))
    right = max(0, min(width, int(x2 + 0.999999) + BOX_PADDING_PX))
    bottom = max(0, min(height, int(y2 + 0.999999) + BOX_PADDING_PX))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def safe_stem(value):
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return cleaned.strip("_") or "item"


def annotate_image(image_path, boxes):
    with Image.open(image_path) as image:
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        wrote_box = False
        for box in boxes:
            bounds = padded_box_bounds(box, image.size)
            if bounds is None:
                continue
            draw.rectangle(bounds, outline=BOX_OUTLINE_COLOR, width=BOX_OUTLINE_WIDTH)
            wrote_box = True
        if not wrote_box:
            return None
        return annotated


def previous_image_name(image_name, image_sequence_names):
    if not image_name:
        return None
    try:
        index = image_sequence_names.index(image_name)
    except ValueError:
        return None
    if index <= 0:
        return None
    previous_name = image_sequence_names[index - 1]
    current_parts = Path(image_name).parts
    previous_parts = Path(previous_name).parts
    if current_parts[:-1] != previous_parts[:-1]:
        return None
    return previous_name


def extract_for_video(video_path, force=False):
    source = filtered_ocr_path(video_path)
    target_dir = output_dir(video_path)
    target_manifest = manifest_path(video_path)
    if target_manifest.exists() and not force:
        print(f"[skip] {target_manifest.name} existe deja")
        return target_manifest
    if not source.exists():
        print(f"[skip] OCR filtered introuvable: {source}")
        return None

    entries = load_others_entries(source)
    if force and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    images_dir = existing_images_dir(video_path)
    image_sequence_names = [path.relative_to(images_dir).as_posix() for path in image_files(images_dir)]
    manifest_items = []
    written = 0
    for index, entry in enumerate(entries, start=1):
        group_dir_name = f"{index:03d}__{safe_stem(entry['text'])}"
        group_dir = target_dir / group_dir_name
        group_dir.mkdir(parents=True, exist_ok=True)

        annotations = list(entry.get("annotations", []))
        occurrences = list(entry.get("all_occurrences", []))
        if not occurrences and entry.get("image") and entry.get("box"):
            occurrences = [
                {
                    "timecode": entry["timecode"],
                    "text": entry["text"],
                    "image": entry["image"],
                    "box": entry["box"],
                }
            ]

        if not annotations:
            continue
        first_annotation = annotations[0]
        annotation_image_name = first_annotation.get("image")
        grouped_annotations = [annotation for annotation in annotations if annotation.get("image") == annotation_image_name]
        grouped_boxes = [annotation.get("box") for annotation in grouped_annotations]
        grouped_texts = [annotation.get("text", "") for annotation in grouped_annotations]
        if not annotation_image_name or not grouped_boxes:
            continue

        occurrence_image_path = images_dir / annotation_image_name
        if not occurrence_image_path.exists():
            print(f"[skip] image occurrence introuvable: {occurrence_image_path}", flush=True)
            continue
        annotated = annotate_image(occurrence_image_path, grouped_boxes)
        if annotated is None:
            continue
        occurrence_output_name = (
            f"001__{safe_stem(Path(annotation_image_name).stem)}"
            f"__{safe_stem(entry['text'])}.png"
        )
        occurrence_output_path = group_dir / occurrence_output_name
        annotated.save(occurrence_output_path)

        previous_crop = None
        previous_annotation_image_name = previous_image_name(annotation_image_name, image_sequence_names)
        if previous_annotation_image_name:
            previous_image_path = images_dir / previous_annotation_image_name
            if previous_image_path.exists():
                previous_annotated = annotate_image(previous_image_path, grouped_boxes)
                if previous_annotated is not None:
                    previous_output_name = (
                        f"000__{safe_stem(Path(previous_annotation_image_name).stem)}"
                        f"__{safe_stem(entry['text'])}.png"
                    )
                    previous_output_path = group_dir / previous_output_name
                    previous_annotated.save(previous_output_path)
                    previous_crop = f"{group_dir_name}/{previous_output_name}"

        manifest_items.append(
            {
                "entry_id": entry["entry_id"],
                "timecode": entry["timecode"],
                "text": entry["text"],
                "image": entry["image"],
                "box": entry["box"],
                "texts": [annotation.get("text", "") for annotation in annotations],
                "images": [annotation.get("image") for annotation in annotations],
                "boxes": [annotation.get("box") for annotation in annotations],
                "group_dir": group_dir_name,
                "crop": f"{group_dir_name}/{occurrence_output_name}",
                "previous_crop": previous_crop,
                "selected_occurrence": {
                    "timecode": entry["timecode"],
                    "text": entry["text"],
                    "texts": grouped_texts,
                    "image": annotation_image_name,
                    "previous_image": previous_annotation_image_name,
                    "images": [annotation.get("image") for annotation in annotations],
                    "box": grouped_boxes[0],
                    "boxes": grouped_boxes,
                    "score": occurrences[0].get("score") if occurrences else None,
                    "annotation": f"{group_dir_name}/{occurrence_output_name}",
                    "previous_annotation": previous_crop,
                },
                "all_occurrence_count": len(occurrences),
            }
        )
        written += 1

    payload = {
        "source": relative_to_video_dir(source, video_path),
        "kind": "others",
        "output_dir": relative_to_video_dir(target_dir, video_path),
        "image_count": written,
        "all_occurrence_image_count": sum(item["all_occurrence_count"] for item in manifest_items),
        "rendering": {
            "mode": "full_image_with_red_box",
            "padding_px": BOX_PADDING_PX,
            "outline_color": list(BOX_OUTLINE_COLOR),
            "outline_width": BOX_OUTLINE_WIDTH,
            "includes_previous_image": True,
        },
        "items": manifest_items,
    }
    write_json(target_manifest, payload)
    print(f"[ok] {target_dir} ({written} image(s) annotee(s))", flush=True)
    return target_manifest
