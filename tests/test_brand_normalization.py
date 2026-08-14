import unittest

from pipeline.support.brand_normalization import normalize_ionis_stm_text


class BrandNormalizationTests(unittest.TestCase):
    def test_all_ionis_stm_variants_share_one_normalization(self):
        variants = (
            "Ionis-STM",
            "Ionis STM",
            "Onis-STM",
            "Onis STM",
            "L'Ionis STM",
            "L’Onis-STM",
            "L Ionis STM",
            "Lonis STM",
            "Yonis STM",
            "Yaunis-STM",
            "Unisystem",
            "UNISSTM",
            "IonisSTM",
            "UNICEF-CM",
            "UNICEF-TM",
            "IONIS ASTM",
        )

        normalized, matched_sources = normalize_ionis_stm_text(
            " | ".join(variants)
        )

        self.assertEqual(
            normalized,
            " | ".join(["Ionis-STM"] * len(variants)),
        )
        self.assertEqual(matched_sources, list(variants))

    def test_normalization_does_not_match_a_word_prefix(self):
        normalized, matched_sources = normalize_ionis_stm_text(
            "ionis stmlab unisysteme lUnisystem"
        )

        self.assertEqual(
            normalized,
            "ionis stmlab unisysteme lUnisystem",
        )
        self.assertEqual(matched_sources, [])


if __name__ == "__main__":
    unittest.main()
