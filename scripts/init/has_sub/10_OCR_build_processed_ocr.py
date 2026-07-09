import os
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
INIT_DIR = CURRENT_DIR.parent
if str(INIT_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_DIR))
os.environ["PIPELINE_TRANSCRIPTS_DIR_NAME"] = "transcripts_ocr"

from common.used_by_hs10_ns10_processed_ocr_builder import add_common_args, process_matching_videos  # noqa: E402
import argparse  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build processed OCR pour les videos avec has_subtitles=true."
    )
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    process_matching_videos(args, expected_has_subtitles=True)


if __name__ == "__main__":
    main()
