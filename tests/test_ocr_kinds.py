import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.steps.ocr.build_processed_ocr import normalize_output_kinds
from pipeline.support.paddle_ocr import (
    classify_text,
    compact_text_key,
    filter_decor_items,
    records_from_raw_result,
    static_decor_keys,
    text_key,
)


class OcrKindTests(unittest.TestCase):
    def static_entry(self, image_index, text):
        return {
            "item": {
                "image": f"footage/frame_{image_index:02d}.jpg",
                "text": text,
                "kind": "others",
            },
            "key": text_key(text),
            "compact_key": compact_text_key(text),
            "geometry": {
                "cx": 0.5,
                "cy": 0.8,
            },
        }

    def test_classify_text_only_emits_canonical_non_graphic_kinds(self) -> None:
        image_size = (1000, 1000)

        cases = [
            ("IONIS", [10, 10, 150, 60]),
            ("Alice Martin", [10, 850, 250, 900]),
            ("Une question ?", [200, 300, 800, 360]),
        ]

        for text, box in cases:
            with self.subTest(text=text):
                self.assertEqual(classify_text(text, box, image_size), "others")

    def test_classify_text_keeps_subtitle_detection(self) -> None:
        box = [200, 850, 800, 900]

        self.assertEqual(
            classify_text("Ceci est une phrase de sous-titre.", box, (1000, 1000)),
            "subtitle",
        )

    def test_output_normalization_accepts_legacy_kinds(self) -> None:
        items = [
            {"image": "footage/00_01.png", "kind": kind}
            for kind in ("logo", "lower_third", "title", "name", "other", "others")
        ]

        normalized = normalize_output_kinds(items)

        self.assertEqual({item["kind"] for item in normalized}, {"others"})

    def test_image_count_rule_keeps_the_ocr_confidence_threshold(self) -> None:
        records = records_from_raw_result(
            {
                "rec_texts": ["Détection faible", "Détection fiable"],
                "rec_scores": [0.8999, 0.9],
                "rec_polys": [
                    [[0, 0], [10, 0], [10, 10], [0, 10]],
                    [[0, 20], [10, 20], [10, 30], [0, 30]],
                ],
            },
            min_confidence=0.9,
        )

        self.assertEqual(
            [record["text"] for record in records],
            ["Détection fiable"],
        )

    def test_static_text_on_twenty_distinct_images_is_kept(self) -> None:
        entries = [
            self.static_entry(
                image_index,
                "Ionis-STM promo 2015"
                if image_index == 0
                else "lonis-STM promo 2015",
            )
            for image_index in range(20)
        ]
        self.assertEqual(static_decor_keys(entries), set())

    def test_static_text_on_twenty_one_distinct_images_is_removed(self) -> None:
        entries = [
            self.static_entry(
                image_index,
                "Ionis-STM promo 2015"
                if image_index == 0
                else "lonis-STM promo 2015",
            )
            for image_index in range(21)
        ]
        self.assertEqual(len(static_decor_keys(entries)), 21)

    def test_separate_runs_are_accumulated_across_the_video(self) -> None:
        entries = [
            self.static_entry(image_index, "Ionis-STM promo 2015")
            for image_index in (*range(9), *range(20, 29), *range(40, 49))
        ]

        self.assertEqual(len(static_decor_keys(entries)), 27)

    @patch("pipeline.support.paddle_ocr.image_size", return_value=(1280, 720))
    def test_small_text_grouped_with_readable_overlay_is_kept(self, _image_size) -> None:
        items = [
            {
                "image": "footage/00_17_500.jpg",
                "text": "THIERRY VOISIN",
                "kind": "others",
                "second": 17.5,
                "box": [[93, 565], [372, 565], [372, 595], [93, 595]],
            },
            {
                "image": "footage/00_17_500.jpg",
                "text": "Senior Vice-President",
                "kind": "others",
                "second": 17.5,
                "box": [[94, 605], [308, 605], [308, 626], [94, 626]],
            },
        ]

        filtered = filter_decor_items(items, Path("images"))

        self.assertEqual(
            [item["text"] for item in filtered],
            ["THIERRY VOISIN", "Senior Vice-President"],
        )

    @patch("pipeline.support.paddle_ocr.image_size", return_value=(1280, 720))
    def test_small_isolated_text_is_still_removed(self, _image_size) -> None:
        items = [
            {
                "image": "footage/00_17_500.jpg",
                "text": "Senior Vice-President",
                "kind": "others",
                "second": 17.5,
                "box": [[94, 605], [308, 605], [308, 626], [94, 626]],
            },
        ]

        self.assertEqual(filter_decor_items(items, Path("images")), [])


if __name__ == "__main__":
    unittest.main()
