import os
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
INIT_DIR = CURRENT_DIR.parent
if str(INIT_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_DIR))
os.environ["PIPELINE_TRANSCRIPTS_DIR_NAME"] = "transcripts_whisper"

from common.used_by_hs18_ns20_create_plain_transcript import main as shared_main  # noqa: E402


def main():
    shared_main()


if __name__ == "__main__":
    main()
