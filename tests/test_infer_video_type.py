from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
INIT_DIR = ROOT_DIR / "scripts" / "init"
if str(INIT_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_DIR))

MODULE_PATH = INIT_DIR / "06_infer_video_type.py"
SPEC = importlib.util.spec_from_file_location("infer_video_type_step", MODULE_PATH)
infer_video_type = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer_video_type)


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


if __name__ == "__main__":
    unittest.main()
