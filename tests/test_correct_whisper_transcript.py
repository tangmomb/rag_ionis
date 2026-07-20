from collections import defaultdict
import unittest

from pipeline.steps.transcripts import correct_whisper_transcript as correction


class IonisStmCorrectionTests(unittest.TestCase):
    def test_known_ionis_stm_variants_are_normalized(self):
        corrections = defaultdict(set)
        result = correction.correct_text(
            "IONIS STM, yonis stm, l'ionis stm, ionis stm et yaunis stm.",
            {}, {}, {}, {}, correction.correction_settings("balanced"), corrections,
        )

        self.assertEqual(
            result,
            "Ionis-STM, Ionis-STM, Ionis-STM, Ionis-STM et Ionis-STM.",
        )
        self.assertEqual(corrections["l'ionis stm"], {"Ionis-STM"})

    def test_ionis_stm_normalization_does_not_match_a_word_prefix(self):
        corrections = defaultdict(set)

        result = correction.normalize_ionis_stm("ionis stmlab", corrections)

        self.assertEqual(result, "ionis stmlab")
        self.assertFalse(corrections)
