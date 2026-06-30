import argparse
import json
from pathlib import Path

from local_paddle_ocr import (
    LocalPaddleOCR,
    configure_stdio,
    deduplicate_items,
    filter_decor_items,
    image_files,
    image_video_dirs,
    latest_video_dir,
    ocr_items_for_image,
    refine_subtitle_kinds,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DEFAULT_MIN_CONFIDENCE = 0.9


configure_stdio()


def write_outputs(transcript_dir, video_id, result):
    json_path = transcript_dir / f"{video_id}_ocr_processed.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {len(result.get('items', []))} items -> {json_path}", flush=True)


def write_raw_outputs(transcript_dir, video_id, raw_result):
    json_path = transcript_dir / f"{video_id}_ocr_brut.json"
    json_path.write_text(json.dumps(raw_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] raw -> {json_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR local des images contenant du texte avec PaddleOCR."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier contenant images/. Defaut: dernier sous-dossier de downloads/youtube avec images/.",
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
        help="Regenere le JSON OCR meme s'il existe deja.",
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
        images_dir = video_path / "images"
        transcript_dir = video_path / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        output_path = transcript_dir / f"{video_path.name}_ocr_processed.json"
        if output_path.exists() and not args.force:
            print(f"[skip] {output_path.name} existe deja")
            continue

        images = image_files(images_dir)
        if not images:
            print(f"[skip] {video_path.name}: aucune image", flush=True)
            write_outputs(transcript_dir, video_path.name, {"items": []})
            continue

        print(f"[analyse] {video_path.name}: {len(images)} images", flush=True)
        items = []
        raw_items = []
        for index, image_path in enumerate(images, start=1):
            raw_result = ocr.recognize_raw(image_path)
            raw_items.append(
                {
                    "image": image_path.name,
                    "raw": raw_result,
                }
            )
            image_items = ocr_items_for_image(ocr, image_path)
            items.extend(image_items)
            print(f"[ocr {index}/{len(images)}] {image_path.name}: {len(image_items)} texte(s)", flush=True)

        write_raw_outputs(transcript_dir, video_path.name, {"items": raw_items})
        refined_items = refine_subtitle_kinds(items, images_dir)
        processed_items = deduplicate_items(filter_decor_items(refined_items, images_dir))
        write_outputs(transcript_dir, video_path.name, {"items": processed_items})
        print(f"[done] {video_path.name}: {len(items)} items bruts", flush=True)


if __name__ == "__main__":
    main()
