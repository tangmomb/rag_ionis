from __future__ import annotations

import unittest
from unittest.mock import patch

from interface.backend import golden_recall


class GoldenRecallTests(unittest.TestCase):
    def test_google_sheet_csv_url_preserves_gid(self) -> None:
        self.assertEqual(
            golden_recall.google_sheet_csv_url(
                "https://docs.google.com/spreadsheets/d/sheet-id/edit?gid=123"
            ),
            "https://docs.google.com/spreadsheets/d/sheet-id/export?format=csv&gid=123",
        )

    def test_evaluate_uses_returned_video_and_chunk_sources(self) -> None:
        cases = golden_recall.parse_golden_cases(
            "question,youtube_video_ids,relevant_chunk_ids\n"
            'Question,"[""video-1""]","[12,13]"\n'
        )

        class Source:
            def __init__(self, chunk_id: int, video_url: str) -> None:
                self.chunk_id = chunk_id
                self.video_url = video_url

        class Response:
            # ``sources`` intentionally differs: Phoenix evaluates this raw
            # retrieval list, not citations selected for the final answer.
            sources = []
            retrieval = {
                "retrieved_sources": [
                    {"chunk_id": 12, "video_url": "https://www.youtube.com/watch?v=video-1"},
                    {"chunk_id": 99, "video_url": "https://www.youtube.com/watch?v=other"},
                ]
            }

        with patch.object(golden_recall, "run_rag", return_value=Response()):
            result = golden_recall.evaluate(cases)

        self.assertEqual(result["video_recall"], 1.0)
        self.assertEqual(result["chunk_recall"], 0.5)
        self.assertEqual(result["video_recall_micro"], 1.0)
        self.assertEqual(result["case_count"], 1)

    def test_main_fails_when_global_recall_threshold_is_missed(self) -> None:
        with (
            patch.object(golden_recall, "load_golden_cases", return_value=[]),
            patch.object(
                golden_recall,
                "evaluate",
                return_value={
                    "failed_cases": [],
                    "video_recall": 0.89,
                    "chunk_recall": 1.0,
                },
            ),
        ):
            exit_code = golden_recall.main(
                ["--min-recall", "0.90"]
            )

        self.assertEqual(exit_code, 1)

    def test_evaluate_averages_recall_per_question_like_phoenix(self) -> None:
        cases = golden_recall.parse_golden_cases(
            "question,youtube_video_ids,relevant_chunk_ids\n"
            'Q1,"[""a"",""b""]",[]\n'
            'Q2,"[""c""]",[]\n'
        )

        class Response:
            def __init__(self, video_ids: list[str]) -> None:
                self.retrieval = {
                    "retrieved_sources": [
                        {"video_url": f"https://www.youtube.com/watch?v={video_id}"}
                        for video_id in video_ids
                    ]
                }

        with patch.object(
            golden_recall,
            "run_rag",
            side_effect=[Response(["a"]), Response(["c"])],
        ):
            result = golden_recall.evaluate(cases)

        self.assertEqual(result["video_recall"], 0.75)
        self.assertEqual(result["video_recall_micro"], 2 / 3)

    def test_evaluate_waits_between_questions(self) -> None:
        cases = golden_recall.parse_golden_cases(
            "question,youtube_video_ids,relevant_chunk_ids\n"
            'Q1,"[""a""]",[]\n'
            'Q2,"[""b""]",[]\n'
        )

        class Response:
            retrieval = {"retrieved_sources": []}

        with (
            patch.object(golden_recall, "run_rag", return_value=Response()),
            patch.object(golden_recall.time, "sleep") as sleep,
        ):
            golden_recall.evaluate(cases, request_delay_seconds=10.0)

        sleep.assert_called_once_with(10.0)


if __name__ == "__main__":
    unittest.main()
