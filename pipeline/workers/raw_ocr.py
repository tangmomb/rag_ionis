from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Worker interne pour l'extraction OCR isolee.",
    )
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("PADDLEOCR_BATCH_SIZE", "8")),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Importer le moteur seulement dans ce processus. En particulier, ne pas
    # importer PyTorch avant Paddle sur Windows.
    from pipeline.steps.inspection.extract_raw_ocr import extract_for_video

    extract_for_video(
        args.video_path,
        device=args.device,
        lang=args.lang,
        min_confidence=args.min_confidence,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
