from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.import_phoenix_dataset import google_sheet_csv_url, load_examples


class ImportPhoenixDatasetTests(unittest.TestCase):
    def test_load_examples_maps_labels_to_expected_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "cases.csv"
            csv_path.write_text(
                "question,difficulty,expected_action,shadow_verdict,shadow_issue\n"
                "Question ?,easy,clarify,acceptable,ambiguous_question\n",
                encoding="utf-8",
            )

            examples = load_examples(csv_path)

        self.assertEqual(examples[0]["input"], {"question": "Question ?"})
        self.assertEqual(
            examples[0]["output"],
            {
                "action": "clarify",
                "shadow_verdict": "acceptable",
                "shadow_issue": "ambiguous_question",
            },
        )
        self.assertEqual(examples[0]["metadata"], {"difficulty": "easy"})

    def test_load_examples_rejects_unknown_shadow_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "cases.csv"
            csv_path.write_text(
                "question,difficulty,expected_action,shadow_verdict,shadow_issue\n"
                "Question ?,easy,answer,acceptable,unknown\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "shadow_issue invalide"):
                load_examples(csv_path)

    def test_load_examples_maps_golden_dataset_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "golden_dataset.csv"
            csv_path.write_text(
                "question,expected_execution_route,youtube_video_ids,relevant_chunk_ids\n"
                "Question ?,vector_search,\"[\"\"video-1\"\"]\",\"[12,13]\"\n",
                encoding="utf-8",
            )

            examples = load_examples(csv_path)

        self.assertEqual(examples[0]["input"], {"question": "Question ?"})
        self.assertEqual(
            examples[0]["output"],
            {
                "expected_execution_route": "vector_search",
                "youtube_video_ids": ["video-1"],
                "relevant_chunk_ids": [12, 13],
            },
        )

    def test_google_sheet_csv_url_preserves_gid(self) -> None:
        self.assertEqual(
            google_sheet_csv_url(
                "https://docs.google.com/spreadsheets/d/sheet-id/edit?gid=123"
            ),
            "https://docs.google.com/spreadsheets/d/sheet-id/export?format=csv&gid=123",
        )


if __name__ == "__main__":
    unittest.main()
