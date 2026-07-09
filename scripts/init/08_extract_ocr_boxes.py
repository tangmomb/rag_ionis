import argparse
import json
from pathlib import Path

from common.pipeline_analysis import update_analysed_infos
from common.local_paddle_ocr import (
    boxes_from_raw_result,
    configure_stdio,
    image_video_dirs,
    latest_video_dir,
    seconds_from_image_name,
)
from common.pipeline_paths import existing_ocr_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
RAW_GROUPS = ("footage", "graphic", "mixture")
LOCATION_NAME = "ocr_box_locations.json"
LEGACY_LOCATION_NAME = "ocr_location.json"


configure_stdio()


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
    return json.loads(path.read_text(encoding="utf-8"))


def sort_key(item):
    image_name = str(item.get("image", ""))
    parsed_second = seconds_from_image_name(Path(image_name).name)
    return (
        parsed_second if parsed_second is not None else float("inf"),
        image_name,
    )


def format_box(box):
    return json.dumps(box, ensure_ascii=False)


def format_result(result):
    lines = ["{"]
    lines.append(f'  "sources": {json.dumps(result.get("sources", []), ensure_ascii=False)},')
    lines.append('  "items": [')
    items = result.get("items", [])
    for item_index, item in enumerate(items):
        suffix = "," if item_index < len(items) - 1 else ""
        lines.append("    {")
        lines.append(f'      "image": {json.dumps(item.get("image", ""), ensure_ascii=False)},')
        lines.append('      "boxes": [')
        boxes = item.get("boxes", [])
        for box_index, box in enumerate(boxes):
            box_suffix = "," if box_index < len(boxes) - 1 else ""
            lines.append(f"        {format_box(box)}{box_suffix}")
        lines.append("      ]")
        lines.append(f"    }}{suffix}")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_outputs(ocr_dir, result):
    json_path = location_path(ocr_dir)
    json_path.write_text(format_result(result), encoding="utf-8")
    print(f"[write] location -> {json_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrait les emplacements OCR depuis les JSON bruts et produit un ocr_box_locations.json unique."
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
        "--force",
        action="store_true",
        help="Regenere le JSON OCR location meme s'il existe deja.",
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
        ocr_dir = existing_ocr_dir(video_path)
        ocr_dir.mkdir(parents=True, exist_ok=True)
        sources = raw_paths(ocr_dir)
        target = location_path(ocr_dir)
        if target.exists() and not args.force:
            print(f"[skip] {target.name} existe deja")
            continue
        available_sources = {group_name: path for group_name, path in sources.items() if path.exists()}
        if not available_sources:
            print(f"[skip] OCR brut introuvable dans: {ocr_dir}")
            continue

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
                boxes = boxes_from_raw_result(raw_item.get("raw"))
                total_boxes += len(boxes)
                total_images += 1
                box_items.append(
                    {
                        "image": image_name,
                        "boxes": boxes,
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
                "items": box_items,
            },
        )
        update_analysed_infos(
            video_path,
            "ocr_boxes",
            {
                "status": "done",
                "sources": [relative_to_video_dir(ocr_dir / name, video_path) for name in source_names],
                "boxes_file": relative_to_video_dir(target, video_path),
                "image_count": total_images,
                "box_count": total_boxes,
            },
        )
        print(f"[done] {video_path.name}: {total_boxes} box(es)", flush=True)
        done += 1

    print(f"{done} JSON OCR location generes.")


if __name__ == "__main__":
    main()
