import argparse
import shutil
from pathlib import Path

from local_paddle_ocr import (
    LocalPaddleOCR,
    configure_stdio,
    image_files,
    image_video_dirs,
    latest_video_dir,
    ocr_items_for_image,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")


configure_stdio()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtre localement les images qui contiennent du texte avec PaddleOCR."
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
        default=0.45,
        help="Score minimum pour garder une detection OCR. Defaut: 0.45",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le dossier images/with_text meme s'il existe deja.",
    )
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--detail", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--batch-api", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def write_filtered_images(images_dir, images, force=False):
    output_dir = images_dir / "with_text"
    if output_dir.exists() and force:
        for existing in output_dir.iterdir():
            if existing.is_file():
                existing.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for path in images:
        target = output_dir / path.name
        if target.exists() and not force:
            continue
        shutil.copy2(path, target)
        copied += 1
    print(f"[write] {copied} images -> {output_dir}", flush=True)


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
        images = image_files(images_dir)
        if not images:
            print(f"[skip] {video_path.name}: aucune image", flush=True)
            continue

        print(f"[analyse] {video_path.name}: {len(images)} images", flush=True)
        kept = []
        for index, image_path in enumerate(images, start=1):
            items = ocr_items_for_image(ocr, image_path)
            if items:
                kept.append(image_path)
            print(f"[ocr {index}/{len(images)}] {image_path.name}: {len(items)} texte(s)", flush=True)

        write_filtered_images(images_dir, kept, force=args.force)
        print(f"[done] {video_path.name}: {len(kept)}/{len(images)} images gardees", flush=True)


if __name__ == "__main__":
    main()
