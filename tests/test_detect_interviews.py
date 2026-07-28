import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.steps.inspection.detect_interviews import (
    InterviewDetectionOptions,
    bridge_transient_gaps,
    classify_pair,
    dominant_visual_cluster,
)


def create_frame(
    path,
    *,
    scene="interview",
    subtitle=False,
    hand_offset=0,
):
    image = Image.new(
        "RGB",
        (320, 180),
        (226, 222, 207) if scene == "interview" else (36, 72, 128),
    )
    draw = ImageDraw.Draw(image)
    if scene == "interview":
        for x in range(20, 320, 70):
            draw.rectangle(
                (x, 15, x + 28, 35),
                outline=(90, 125, 105),
                width=3,
            )
        draw.ellipse((120, 30, 200, 115), fill=(186, 137, 105))
        draw.rectangle((105, 105, 215, 180), fill=(115, 102, 88))
        draw.line(
            (150, 120, 95 + hand_offset, 150),
            fill=(186, 137, 105),
            width=14,
        )
        draw.line(
            (170, 120, 235 - hand_offset, 145),
            fill=(186, 137, 105),
            width=14,
        )
    else:
        for offset in range(-180, 320, 35):
            draw.line(
                (offset, 0, offset + 180, 180),
                fill=(235, 190, 55),
                width=12,
            )
    if subtitle:
        draw.rectangle((45, 150, 275, 179), fill=(8, 8, 8))
        draw.rectangle((65, 157, 250, 162), fill=(235, 235, 235))
        draw.rectangle((80, 168, 235, 173), fill=(235, 235, 235))
    image.save(path)
    return path


class InterviewDetectionTests(unittest.TestCase):
    def test_ignores_bottom_subtitles_and_interview_motion(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before = create_frame(root / "before.jpg")
            after = create_frame(
                root / "after.jpg",
                subtitle=True,
                hand_offset=18,
            )

            decision = classify_pair(
                before,
                after,
                InterviewDetectionOptions(),
                {},
                {},
            )

        self.assertTrue(decision["is_similar"])

    def test_keeps_a_real_scene_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            interview = create_frame(root / "interview.jpg")
            graphic = create_frame(root / "graphic.jpg", scene="graphic")

            decision = classify_pair(
                interview,
                graphic,
                InterviewDetectionOptions(),
                {},
                {},
            )

        self.assertFalse(decision["is_similar"])

    def test_bridges_a_gap_only_when_temporal_context_matches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = [
                create_frame(
                    root / f"frame_{index}.jpg",
                    hand_offset=index,
                )
                for index in range(5)
            ]
            pairs = [
                {"is_similar": True},
                {"is_similar": False},
                {"is_similar": True},
                {"is_similar": True},
            ]

            bridged = bridge_transient_gaps(
                pairs,
                frames,
                InterviewDetectionOptions(),
                {},
                {},
            )

        self.assertTrue(bridged[1]["is_similar"])
        self.assertTrue(bridged[1]["bridge_gap"])
        self.assertIn("bridge_context", bridged[1])

    def test_does_not_bridge_a_real_cut(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = [
                create_frame(root / "old_0.jpg"),
                create_frame(root / "old_1.jpg"),
                create_frame(root / "new_0.jpg", scene="graphic"),
                create_frame(root / "new_1.jpg", scene="graphic"),
                create_frame(root / "new_2.jpg", scene="graphic"),
            ]
            pairs = [
                {"is_similar": True},
                {"is_similar": False},
                {"is_similar": True},
                {"is_similar": True},
            ]

            bridged = bridge_transient_gaps(
                pairs,
                frames,
                InterviewDetectionOptions(),
                {},
                {},
            )

        self.assertFalse(bridged[1]["is_similar"])
        self.assertNotIn("bridge_gap", bridged[1])

    def test_dominant_cluster_accepts_seventy_percent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = [
                create_frame(
                    root / f"interview_{index}.jpg",
                    hand_offset=index,
                    subtitle=index % 2 == 0,
                )
                for index in range(7)
            ]
            frames.extend(
                create_frame(
                    root / f"graphic_{index}.jpg",
                    scene="graphic",
                )
                for index in range(3)
            )

            cluster = dominant_visual_cluster(
                frames,
                InterviewDetectionOptions(),
            )

        self.assertGreaterEqual(cluster["frame_ratio"], 0.70)

    def test_dominant_cluster_rejects_sixty_percent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = [
                create_frame(root / f"interview_{index}.jpg")
                for index in range(6)
            ]
            frames.extend(
                create_frame(
                    root / f"graphic_{index}.jpg",
                    scene="graphic",
                )
                for index in range(4)
            )

            cluster = dominant_visual_cluster(
                frames,
                InterviewDetectionOptions(),
            )

        self.assertLess(cluster["frame_ratio"], 0.70)


if __name__ == "__main__":
    unittest.main()
