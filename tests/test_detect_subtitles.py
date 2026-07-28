import unittest

from pipeline.steps.inspection.detect_subtitles import analyze_subtitle_anchor
from pipeline.support.paddle_ocr import box_text_pairs_from_raw_result


def subtitle_entry(
    second,
    text,
    *,
    cx=0.82,
    cy=0.96,
    relative_width=0.32,
):
    return {
        "image": f"footage/{second:05.1f}.jpg",
        "second": second,
        "key": text,
        "geometry": {
            "cx": cx,
            "cy": cy,
            "relative_width": relative_width,
            "relative_height": 0.04,
        },
    }


class SubtitleDetectionTests(unittest.TestCase):
    def test_detects_changing_text_at_any_stable_position(self):
        texts = (
            "Bonjour tout le monde.",
            "Nous découvrons le campus.",
            "Voici le prochain projet.",
            "Merci pour votre attention.",
        )
        entries = [
            subtitle_entry(second / 2, texts[min(second // 6, len(texts) - 1)])
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

    def test_rejects_near_duplicate_ocr_as_text_variation(self):
        noisy_logo = (
            "L'ÉCOLE",
            "LECOLE",
            "L ECOLE",
            "LECOLF",
        )
        entries = [
            subtitle_entry(second / 2, noisy_logo[second % len(noisy_logo)])
            for second in range(25)
        ]

        result = analyze_subtitle_anchor(entries)

        self.assertFalse(result["has_subtitles"])
        self.assertEqual(result["reason"], "not_enough_anchor_candidates")

    def test_rejects_stable_decorative_group_with_ocr_noise(self):
        noisy_texts = (
            ("LECOLE", "LECOLF"),
            ("DE LA DOURLE", "DE LA DOUBLE"),
            ("COMIPETENCE", "COMPETENCE"),
        )
        entries = []
        for frame in range(25):
            for row, variants in enumerate(noisy_texts):
                entries.append(
                    subtitle_entry(
                        frame / 2,
                        variants[frame % len(variants)],
                        cx=0.17,
                        cy=0.24 + row * 0.025,
                    )
                )

        result = analyze_subtitle_anchor(entries)

        self.assertFalse(result["has_subtitles"])
        self.assertEqual(result["reason"], "not_enough_anchor_candidates")

    def test_rejects_short_gibberish_as_text_variation(self):
        gibberish = ("1W", "2", "A", "2S", "W", "2W", "3 W D")
        entries = [
            subtitle_entry(
                second / 2,
                gibberish[second % len(gibberish)],
                cx=0.14,
                cy=0.41,
            )
            for second in range(25)
        ]

        result = analyze_subtitle_anchor(entries)

        self.assertFalse(result["has_subtitles"])
        self.assertEqual(result["reason"], "not_enough_anchor_candidates")

    def test_tries_next_anchor_after_dominant_candidate_fails(self):
        changing_texts = (
            "Bonjour à toutes les personnes.",
            "Nous présentons le nouveau campus.",
            "Voici les projets des étudiants.",
            "Merci beaucoup pour votre attention.",
        )
        entries = []
        for frame in range(25):
            second = frame / 2
            for duplicate in range(5):
                entries.append(
                    subtitle_entry(
                        second,
                        "STATIC LOGO",
                        cx=0.20 + duplicate * 0.001,
                        cy=0.20,
                    )
                )
            text = changing_texts[min(frame // 6, 3)]
            entries.append(
                subtitle_entry(second, text, cx=0.05, cy=0.20)
            )
            entries.append(
                subtitle_entry(second, text, cx=0.35, cy=0.20)
            )
            entries.append(
                subtitle_entry(second, text, cx=0.82, cy=0.96)
            )

        result = analyze_subtitle_anchor(entries)

        self.assertTrue(result["has_subtitles"])
        self.assertGreater(result["candidate_rank"], 1)
        self.assertAlmostEqual(result["anchor"]["cx"], 0.82)
        self.assertAlmostEqual(result["anchor"]["cy"], 0.96)

    def test_prefers_wide_subtitle_signal_over_narrow_valid_overlay(self):
        texts = (
            "Bonjour à toutes les personnes.",
            "Nous présentons le nouveau campus.",
            "Voici les projets des étudiants.",
            "Merci beaucoup pour votre attention.",
        )
        entries = []
        for frame in range(25):
            second = frame / 2
            text = texts[min(frame // 6, 3)]
            for duplicate in range(5):
                entries.append(
                    subtitle_entry(
                        second,
                        text,
                        cx=0.25 + duplicate * 0.001,
                        cy=0.40,
                        relative_width=0.08,
                    )
                )
            entries.append(
                subtitle_entry(
                    second,
                    text,
                    cx=0.50,
                    cy=0.94,
                    relative_width=0.55,
                )
            )

        result = analyze_subtitle_anchor(entries)

        self.assertTrue(result["has_subtitles"])
        self.assertEqual(result["valid_candidate_count"], 2)
        self.assertAlmostEqual(result["anchor"]["cx"], 0.50)
        self.assertAlmostEqual(
            result["anchor"]["median_relative_width"],
            0.55,
        )

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
