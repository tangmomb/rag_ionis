import unittest

from pipeline.steps.ocr.build_processed_ocr import normalize_output_kinds
from pipeline.support.paddle_ocr import classify_text


class OcrKindTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
