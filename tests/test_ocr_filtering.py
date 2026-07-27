import unittest

from pipeline.support.ocr_filtering import collapse_graphic_time_groups


def graphic_items(second, image, *texts):
    return [
        {
            "kind": "graphic",
            "second": second,
            "image": image,
            "text": text,
        }
        for text in texts
    ]


class OcrFilteringTests(unittest.TestCase):
    def test_prefers_repeated_graphic_text_over_longer_orphan(self):
        items = [
            *graphic_items(
                64.5,
                "graphic/01_04_500.jpg",
                "EFFICA",
                "ÉNERG",
                "CONSOI",
                "BIODIVER",
                "&RSE",
            ),
            *graphic_items(
                65.0,
                "graphic/01_05.jpg",
                "IONIS-STM",
                "EN DEUX",
            ),
            *graphic_items(
                65.5,
                "graphic/01_05_500.jpg",
                "IONIS-STM",
                "EN DEUX MOTS ?",
            ),
            *graphic_items(
                66.0,
                "graphic/01_06.jpg",
                "IONIS-STM",
                "EN DEUX MOTS ?",
            ),
        ]

        result = collapse_graphic_time_groups(items)

        self.assertEqual(
            [item["text"] for item in result],
            ["IONIS-STM", "EN DEUX MOTS ?"],
        )
        self.assertEqual(
            {item["image"] for item in result},
            {"graphic/01_05_500.jpg"},
        )

    def test_keeps_distinct_supported_cards_inside_same_time_window(self):
        items = [
            *graphic_items(1.0, "graphic/00_01.jpg", "PREMIER CARTON"),
            *graphic_items(1.5, "graphic/00_01_500.jpg", "PREMIER CARTON"),
            *graphic_items(2.0, "graphic/00_02.jpg", "SECOND CARTON"),
            *graphic_items(2.5, "graphic/00_02_500.jpg", "SECOND CARTON"),
        ]

        result = collapse_graphic_time_groups(items)

        self.assertEqual(
            [item["text"] for item in result],
            ["PREMIER CARTON", "SECOND CARTON"],
        )

    def test_accepts_partial_or_slightly_different_ocr_as_a_cousin(self):
        items = [
            *graphic_items(5.0, "graphic/00_05.jpg", "IONIS-5TM EN"),
            *graphic_items(
                5.5,
                "graphic/00_05_500.jpg",
                "IONIS-STM EN DEUX MOTS ?",
            ),
        ]

        result = collapse_graphic_time_groups(items)

        self.assertEqual(
            [item["text"] for item in result],
            ["IONIS-STM EN DEUX MOTS ?"],
        )

    def test_discards_graphic_text_without_a_cousin(self):
        items = graphic_items(8.0, "graphic/00_08.jpg", "TEXTE ISOLE")

        self.assertEqual(collapse_graphic_time_groups(items), [])


if __name__ == "__main__":
    unittest.main()
