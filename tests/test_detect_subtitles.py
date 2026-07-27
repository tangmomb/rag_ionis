import unittest

from pipeline.steps.inspection.detect_subtitles import analyze_subtitle_anchor
from pipeline.support.paddle_ocr import box_text_pairs_from_raw_result


def subtitle_entry(second, text, *, cx=0.82, cy=0.96):
    return {
        "image": f"footage/{second:05.1f}.jpg",
        "second": second,
        "key": text,
        "geometry": {
            "cx": cx,
            "cy": cy,
            "relative_width": 0.32,
            "relative_height": 0.04,
        },
    }


class SubtitleDetectionTests(unittest.TestCase):
    def test_detects_changing_text_at_any_stable_position(self):
        entries = [
            subtitle_entry(second / 2, f"subtitle-{second // 4}")
            for second in range(25)
        ]

        result = analyze_subtitle_anchor(entries)

        self.assertTrue(result["has_subtitles"])
        self.assertAlmostEqual(result["anchor"]["cx"], 0.82)
        self.assertAlmostEqual(result["anchor"]["cy"], 0.96)
        self.assertGreaterEqual(result["text_variant_count"], 3)

    def test_rejects_static_text_at_a_stable_position(self):
        entries = [
            subtitle_entry(second / 2, "static-logo")
            for second in range(25)
        ]

        result = analyze_subtitle_anchor(entries)

        self.assertFalse(result["has_subtitles"])
        self.assertEqual(result["reason"], "not_enough_anchor_candidates")

    def test_preserves_text_alignment_with_detected_boxes(self):
        raw = {
            "rec_polys": [
                [[0, 0], [10, 0], [10, 5], [0, 5]],
                [[20, 0], [30, 0], [30, 5], [20, 5]],
            ],
            "rec_texts": ["premier", "second"],
        }

        pairs = box_text_pairs_from_raw_result(raw)

        self.assertEqual([text for _box, text in pairs], ["premier", "second"])
        self.assertEqual(len(pairs), 2)


if __name__ == "__main__":
    unittest.main()
