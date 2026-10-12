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

    def test_evaluate_uses_raw_retrieval_sources(self) -> None:
        cases = golden_recall.parse_golden_cases(
            "question,youtube_video_ids,relevant_chunk_ids\n"
            'Question,"[""video-1""]","[12,13]"\n'
        )
        retrieved = [
            {"chunk_id": 12, "video_url": "https://www.youtube.com/watch?v=video-1"},
            {"chunk_id": 99, "video_url": "https://www.youtube.com/watch?v=other"},
        ]

        with patch.object(
            golden_recall, "orchestrate_request", return_value=("", retrieved, {})
        ):
            result = golden_recall.evaluate(cases)

        self.assertEqual(result["video_recall"], 1.0)
        self.assertEqual(result["chunk_recall"], 0.5)
        self.assertEqual(result["video_recall_micro"], 1.0)
        self.assertEqual(result["case_count"], 1)

    def test_evaluate_skips_questions_without_expected_sources(self) -> None:
        cases = golden_recall.parse_golden_cases(
            "question,youtube_video_ids,relevant_chunk_ids,expected_action\n"
            'Question,"[""video-1""]",[],\n'
            "Question piège,[],[],abstain\n"
        )

        with patch.object(
            golden_recall,
            "orchestrate_request",
            return_value=("", [{"chunk_id": 1, "video_url": "https://youtu.be/video-1"}], {}),
        ) as orchestrate:
            result = golden_recall.evaluate(cases)

        orchestrate.assert_called_once()
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["measured_case_count"], 1)
        self.assertEqual(result["video_recall"], 1.0)

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

        def retrieval(video_ids: list[str]):
            sources = [
                {"video_url": f"https://www.youtube.com/watch?v={video_id}"}
                for video_id in video_ids
            ]
            return "", sources, {}

        with patch.object(
            golden_recall,
            "orchestrate_request",
            side_effect=[retrieval(["a"]), retrieval(["c"])],
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

        with (
            patch.object(golden_recall, "orchestrate_request", return_value=("", [], {})),
            patch.object(golden_recall.time, "sleep") as sleep,
        ):
            golden_recall.evaluate(cases, request_delay_seconds=10.0)

        sleep.assert_called_once_with(10.0)


if __name__ == "__main__":
    unittest.main()
