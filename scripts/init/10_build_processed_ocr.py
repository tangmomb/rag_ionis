import argparse
import json
from pathlib import Path

from pipeline_analysis import analysed_infos_path, update_analysed_infos
from local_paddle_ocr import (
    box_bounds,
    box_geometry,
    collapse_answer_overlay_items,
    collapse_graphic_sequence_items,
    configure_stdio,
    deduplicate_items,
    filter_decor_items,
    graphic_kind_for_image,
    image_video_dirs,
    image_size,
    latest_video_dir,
    mark_last_graphic_sequence_as_outro,
    ocr_items_from_raw_result,
    refine_subtitle_kinds,
    seconds_from_image_name,
    subtitle_text_signal,
)
from pipeline_paths import existing_images_dir, existing_ocr_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DEFAULT_MIN_CONFIDENCE = 0.9
RAW_GROUPS = ("footage", "graphic", "mixture")
PROCESSED_NAME = "processed_ocr_items.json"
LEGACY_PROCESSED_NAME = "ocr_processed.json"


configure_stdio()


def raw_paths(ocr_dir):
    paths = {}
    for group_name in RAW_GROUPS:
        preferred = ocr_dir / f"raw_ocr_{group_name}_frames.json"
        legacy = ocr_dir / f"ocr_{group_name}.json"
        paths[group_name] = legacy if legacy.exists() and not preferred.exists() else preferred
    return paths


def processed_path(ocr_dir):
    preferred = ocr_dir / PROCESSED_NAME
    legacy = ocr_dir / LEGACY_PROCESSED_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sort_key(item):
    image_name = str(item.get("image", ""))
    parsed_second = seconds_from_image_name(Path(image_name).name)
    return (
        parsed_second if parsed_second is not None else float("inf"),
        image_name,
    )


def analysed_has_subtitles(video_path):
    path = analysed_infos_path(video_path)
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    value = payload.get("has_subtitles")
    return value if isinstance(value, bool) else None


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
        word_count, has_sentence_punctuation, _has_subtitle_signal = subtitle_text_signal(
            item.get("text", ""),
            geometry["relative_width"],
        )
        replacement = dict(item)
        replacement["kind"] = "others"
        sanitized.append(replacement)
    return sanitized


def normalize_output_kinds(items):
    normalized_items = []
    for item in items:
        replacement = dict(item)
        if graphic_kind_for_image(item.get("image")) == "graphic":
            replacement["kind"] = "graphic"
        elif item.get("kind") == "subtitle":
            replacement["kind"] = "subtitle"
        else:
            replacement["kind"] = "others"
        normalized_items.append(replacement)
    return normalized_items


def write_outputs(ocr_dir, result):
    json_path = processed_path(ocr_dir)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {len(result.get('items', []))} items -> {json_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transforme l'OCR brut en OCR processed sans relancer PaddleOCR."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier contenant outputs/images/. Defaut: dernier sous-dossier de downloads/youtube avec images.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent utilise si --video-dir est absent. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        help="Nombre maximum de videos a analyser.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="rec_score minimum pour garder une detection OCR. Defaut: valeur du JSON brut ou 0.9.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le JSON OCR processed meme s'il existe deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(image_video_dirs(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        images_dir = existing_images_dir(video_path)
        ocr_dir = existing_ocr_dir(video_path)
        has_subtitles = analysed_has_subtitles(video_path)
        ocr_dir.mkdir(parents=True, exist_ok=True)
        sources = raw_paths(ocr_dir)
        target = processed_path(ocr_dir)
        if target.exists() and not args.force:
            print(f"[skip] {target.name} existe deja")
            continue
        available_sources = {group_name: path for group_name, path in sources.items() if path.exists()}
        if not available_sources:
            print(f"[skip] OCR brut introuvable dans: {ocr_dir}")
            continue

        min_confidence = args.min_confidence
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
            print(f"[process {index}/{len(raw_items)}] {image_name}: {len(image_items)} texte(s)", flush=True)

        refined_items = refine_subtitle_kinds(items, images_dir)
        if has_subtitles is False:
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
        update_analysed_infos(
            video_path,
            "ocr_processed",
            {
                "status": "done",
                "sources": [relative_to_video_dir(ocr_dir / name, video_path) for name in source_names],
                "processed_file": relative_to_video_dir(target, video_path),
                "item_count": len(processed_items),
                "min_confidence": min_confidence,
            },
        )
        print(f"[done] {video_path.name}: {len(items)} items intermediaires", flush=True)
        done += 1

    print(f"{done} JSON OCR processed generes.")


if __name__ == "__main__":
    main()
