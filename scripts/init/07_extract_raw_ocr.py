import argparse
import json
from pathlib import Path

from pipeline_analysis import update_analysed_infos
from local_paddle_ocr import (
    LocalPaddleOCR,
    configure_stdio,
    image_files,
    image_video_dirs,
    latest_video_dir,
)
from pipeline_paths import existing_images_dir, ocr_dir as output_ocr_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DEFAULT_MIN_CONFIDENCE = 0.9
IMAGE_GROUPS = ("footage", "graphic", "mixture")


configure_stdio()


def raw_ocr_name(group_name):
    return f"raw_ocr_{group_name}_frames.json"


def existing_raw_ocr_path(ocr_dir, group_name):
    preferred = ocr_dir / raw_ocr_name(group_name)
    legacy = ocr_dir / f"ocr_{group_name}.json"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def write_group_raw_outputs(ocr_dir, group_name, raw_result):
    json_path = ocr_dir / raw_ocr_name(group_name)
    json_path.write_text(json.dumps(raw_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] raw {group_name} -> {json_path}", flush=True)
    return json_path


def base_payload(args, items):
    return {
        "device": args.device,
        "lang": args.lang,
        "min_confidence": args.min_confidence,
        "items": items,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR local brut des images contenant du texte avec PaddleOCR."
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
        "--device",
        default="gpu:0",
        help="Device PaddleOCR, par exemple gpu:0 ou cpu. Defaut: gpu:0",
    )
    parser.add_argument(
        "--lang",
        default="fr",
        help="Langue OCR PaddleOCR. Defaut: fr",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"rec_score minimum pour garder une detection OCR. Defaut: {DEFAULT_MIN_CONFIDENCE}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le JSON OCR brut meme s'il existe deja.",
    )
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--detail", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--images-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--batch-api", action="store_true", help=argparse.SUPPRESS)
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
    print(f"OCR local: PaddleOCR ({args.device}, lang={args.lang})", flush=True)
    ocr = LocalPaddleOCR(device=args.device, lang=args.lang, min_confidence=args.min_confidence)

    for video_path in videos:
        images_dir = existing_images_dir(video_path)
        ocr_dir = output_ocr_dir(video_path)
        ocr_dir.mkdir(parents=True, exist_ok=True)
        group_output_paths = {
            group_name: existing_raw_ocr_path(ocr_dir, group_name)
            for group_name in IMAGE_GROUPS
        }
        if all(path.exists() for path in group_output_paths.values()) and not args.force:
            print(f"[skip] OCR brut deja genere pour {video_path.name}")
            continue

        images = image_files(images_dir)
        if not images:
            print(f"[skip] {video_path.name}: aucune image", flush=True)
            empty_payload = base_payload(args, [])
            for group_name in IMAGE_GROUPS:
                write_group_raw_outputs(ocr_dir, group_name, empty_payload)
            continue

        print(f"[analyse] {video_path.name}: {len(images)} images", flush=True)
        raw_items_by_group = {group_name: [] for group_name in IMAGE_GROUPS}
        for index, image_path in enumerate(images, start=1):
            image_name = image_path.relative_to(images_dir).as_posix()
            raw_result = ocr.recognize_raw(image_path)
            item = {
                "image": image_name,
                "raw": raw_result,
            }
            group_name = Path(image_name).parts[0] if Path(image_name).parts else ""
            if group_name in raw_items_by_group:
                raw_items_by_group[group_name].append(item)
            print(f"[ocr {index}/{len(images)}] {image_name}: brut capture", flush=True)

        raw_files = {}
        for group_name in IMAGE_GROUPS:
            group_path = write_group_raw_outputs(
                ocr_dir,
                group_name,
                base_payload(args, raw_items_by_group[group_name]),
            )
            raw_files[group_name] = relative_to_video_dir(group_path, video_path)
        update_analysed_infos(
            video_path,
            "extract_raw_ocr",
            {
                "status": "done",
                "raw_files": raw_files,
                "image_count": sum(len(items) for items in raw_items_by_group.values()),
                "image_counts": {
                    group_name: len(items)
                    for group_name, items in raw_items_by_group.items()
                },
                "device": args.device,
                "lang": args.lang,
                "min_confidence": args.min_confidence,
            },
        )
        print(
            f"[done] {video_path.name}: {sum(len(items) for items in raw_items_by_group.values())} images OCR brutes",
            flush=True,
        )


if __name__ == "__main__":
    main()
