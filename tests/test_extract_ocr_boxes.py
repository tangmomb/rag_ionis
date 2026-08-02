from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.steps.inspection import extract_ocr_boxes


class ExtractOcrBoxesTests(unittest.TestCase):
    def test_location_output_preserves_scores_for_subtitle_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "video.mp4"
            video.touch()
            ocr_dir = root / "ocr"
            raw_dir = ocr_dir / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "raw_ocr_footage_frames.json").write_text(
                json.dumps(
                    {
                        "min_confidence": 0.9,
                        "items": [
                            {
                                "image": "footage/00_00.jpg",
                                "raw": {
                                    "rec_polys": [
                                        [[0, 0], [10, 0], [10, 5], [0, 5]],
                                        [[20, 0], [30, 0], [30, 5], [20, 5]],
                                    ],
                                    "rec_texts": ["faible", "fort"],
                                    "rec_scores": [0.42, 0.96],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                extract_ocr_boxes,
                "existing_ocr_dir",
                return_value=ocr_dir,
            ):
                target = extract_ocr_boxes.extract_for_video(video, force=True)

            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["min_confidence"], 0.9)
        self.assertEqual(payload["items"][0]["texts"], ["faible", "fort"])
        self.assertEqual(payload["items"][0]["scores"], [0.42, 0.96])
