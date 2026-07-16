from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.inspection import infer_video_type


class InferVideoTypeTests(unittest.TestCase):
    def motion_manifest(self):
        return {
            "class_counts": {
                "footage": 1,
                "graphic": 9,
                "mixture": 0,
            },
            "items": [
                {"pred_label": "footage"},
                {"pred_label": "graphic"},
            ],
        }

    def test_motion_design_requires_video_shorter_than_three_minutes(self):
        self.assertEqual(
            infer_video_type.infer_video_type_from_manifest(
                self.motion_manifest(),
                duration_seconds=179,
            ),
            "motion_design",
        )

    def test_exactly_three_minutes_is_not_motion_design(self):
        self.assertEqual(
            infer_video_type.infer_video_type_from_manifest(
                self.motion_manifest(),
                duration_seconds=180,
            ),
            "video_recording",
        )

    def test_unknown_duration_is_not_motion_design(self):
        self.assertEqual(
            infer_video_type.infer_video_type_from_manifest(
                self.motion_manifest(),
                duration_seconds=None,
            ),
            "video_recording",
        )

    def test_long_video_without_any_footage_is_not_motion_design(self):
        manifest = {
            "class_counts": {"footage": 0, "graphic": 10, "mixture": 0},
            "items": [{"pred_label": "graphic"}],
        }

        self.assertEqual(
            infer_video_type.infer_video_type_from_manifest(
                manifest,
                duration_seconds=181,
            ),
            "video_recording",
        )

    def test_duration_can_be_read_from_generated_video_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory)
            metadata_dir = video_dir / "metadata"
            metadata_dir.mkdir()
            video = video_dir / "video.mp4"
            video.touch()
            (metadata_dir / "video_manifest.json").write_text(
                json.dumps({"video": {"duration_seconds": 91.02}}),
                encoding="utf-8",
            )

            duration = infer_video_type.video_duration_seconds(video)

        self.assertEqual(duration, 91.02)


if __name__ == "__main__":
    unittest.main()
