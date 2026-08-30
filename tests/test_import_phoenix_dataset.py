from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.import_phoenix_dataset import load_examples


class ImportPhoenixDatasetTests(unittest.TestCase):
    def test_load_examples_maps_labels_to_expected_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "cases.csv"
            csv_path.write_text(
                "question,difficulty,expected_action,shadow_status\n"
                "Question ?,easy,clarify,ambiguous_question\n",
                encoding="utf-8",
            )

            examples = load_examples(csv_path)

        self.assertEqual(examples[0]["input"], {"question": "Question ?"})
        self.assertEqual(
            examples[0]["output"],
            {"action": "clarify", "shadow_status": "ambiguous_question"},
        )
        self.assertEqual(examples[0]["metadata"], {"difficulty": "easy"})

    def test_load_examples_rejects_unknown_shadow_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "cases.csv"
            csv_path.write_text(
                "question,difficulty,expected_action,shadow_status\n"
                "Question ?,easy,answer,unknown\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "shadow_status invalide"):
                load_examples(csv_path)


if __name__ == "__main__":
    unittest.main()
