from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from utils.app_phoenix_replay.app import app


TRACE_ID = "c4667ce228864d1ee8f3b8016205b236"


class PhoenixReplayAppTests(unittest.TestCase):
    def test_index_exposes_replay_interface(self) -> None:
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Phoenix Replay", response.text)
        self.assertIn("Rejouer le dernier tour", response.text)

    def test_inspect_runs_without_writes(self) -> None:
        result = {
            "status": "dry_run",
            "original_trace_id": TRACE_ID,
            "target_question": "Question",
            "history": [],
            "history_message_count": 0,
        }
        with patch(
            "utils.app_phoenix_replay.app.run_replay",
            return_value=result,
        ) as run_replay:
            response = TestClient(app).post(
                "/api/inspect",
                json={"trace_id": TRACE_ID, "mode": "exact-context"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result)
        self.assertTrue(run_replay.call_args.args[0].dry_run)
        self.assertFalse(run_replay.call_args.kwargs["shutdown_after"])

    def test_replay_runs_pipeline_mode(self) -> None:
        result = {
            "status": "replayed",
            "replay_trace_id": "new-trace",
            "replay_conversation_id": 444,
        }
        with patch(
            "utils.app_phoenix_replay.app.run_replay",
            return_value=result,
        ) as run_replay:
            response = TestClient(app).post(
                "/api/replay",
                json={"trace_id": TRACE_ID},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result)
        self.assertFalse(run_replay.call_args.args[0].dry_run)

    def test_invalid_trace_id_is_rejected(self) -> None:
        response = TestClient(app).post(
            "/api/inspect",
            json={"trace_id": "not-a-trace"},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
