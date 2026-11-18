from __future__ import annotations

import json
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from interface import rate_limit
from interface.app import app
from interface.backend import api
from interface.backend.request_context import (
    REQUEST_LOGGER_NAME,
    current_request_details,
    current_request_id,
    record_request_details,
)
from interface.backend.schemas import RagRequest, RagResponse


def fake_response() -> RagResponse:
    return RagResponse(
        conversation_id=12,
        message_id=34,
        answer="Réponse",
        action="answer",
        sources=[],
        retrieval={},
    )


class RunRagDetailsTest(unittest.TestCase):
    def run_rag_with_details(self, **patches) -> dict:
        details: dict = {}
        tokens = (current_request_id.set("req-1"), current_request_details.set(details))
        try:
            with mock.patch.object(api, "current_trace_id", return_value="f" * 32), mock.patch.object(
                api, "record_trace_score_annotations"
            ), mock.patch.object(api, "execute_rag", **patches):
                try:
                    api.run_rag(RagRequest(question="Bonjour"))
                except RuntimeError:
                    pass
        finally:
            current_request_details.reset(tokens[1])
            current_request_id.reset(tokens[0])
        return details

    def test_answer_fills_the_request_log_details(self) -> None:
        details = self.run_rag_with_details(return_value=fake_response())

        self.assertEqual(details["rag_event"], "rag_answered")
        self.assertEqual(details["trace_id"], "f" * 32)
        self.assertEqual((details["conversation_id"], details["message_id"]), (12, 34))
        self.assertIn("rag_completed_ms", details)

    def test_failure_fills_trace_and_error(self) -> None:
        details = self.run_rag_with_details(side_effect=RuntimeError("Mistral indisponible"))

        self.assertEqual(details["rag_event"], "rag_failed")
        self.assertEqual(details["trace_id"], "f" * 32)
        self.assertIn("Mistral indisponible", details["error"])
        # Must not override the HTTP line's own duration_ms.
        self.assertNotIn("duration_ms", details)

    def test_outside_an_http_request_details_get_their_own_line(self) -> None:
        with self.assertLogs(REQUEST_LOGGER_NAME, level="INFO") as logs:
            record_request_details("rag_answered", trace_id="t1")

        self.assertEqual(logs.records[0].getMessage(), "rag_answered")
        self.assertEqual(logs.records[0].trace_id, "t1")


class SingleLogLineTest(unittest.TestCase):
    def test_streamed_question_gives_one_line_with_trace_and_real_duration(self) -> None:
        seen: list[str | None] = []

        def slow_answer(payload, **_kwargs):
            seen.append(current_request_id.get())
            time.sleep(0.2)
            record_request_details("rag_answered", trace_id="a" * 32, conversation_id=7, message_id=8)
            return fake_response()

        with (
            mock.patch.object(rate_limit, "enabled", return_value=False),
            mock.patch.object(api, "run_rag", side_effect=slow_answer),
            self.assertLogs(REQUEST_LOGGER_NAME, level="INFO") as logs,
        ):
            response = TestClient(app).post(
                "/api/rag/stream",
                json={"question": "Bonjour"},
                headers={"X-Request-ID": "stream-req-42"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-ID"], "stream-req-42")
        self.assertEqual(seen, ["stream-req-42"])
        lines = [record for record in logs.records if record.getMessage() == "http_request"]
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertEqual(line.request_id, "stream-req-42")
        self.assertEqual(line.trace_id, "a" * 32)
        self.assertEqual((line.conversation_id, line.message_id), (7, 8))
        self.assertEqual(line.status, 200)
        # Logged after the stream ended, so it includes the 200 ms of generation.
        self.assertGreaterEqual(line.duration_ms, 200)

    def test_rejected_question_is_logged_once_without_trace(self) -> None:
        rejection = rate_limit.Rejection(429, "Trop de questions.", 5)
        with (
            mock.patch.object(rate_limit, "enabled", return_value=True),
            mock.patch.object(rate_limit, "check_request", return_value=rejection),
            self.assertLogs(REQUEST_LOGGER_NAME, level="INFO") as logs,
        ):
            TestClient(app).post("/api/rag/stream", json={"question": "Bonjour"})

        lines = [record for record in logs.records if record.getMessage() == "http_request"]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].rate_limited)
        self.assertFalse(hasattr(lines[0], "trace_id"))

    def test_log_line_is_valid_json(self) -> None:
        from interface.request_logging import JsonFormatter

        with (
            mock.patch.object(rate_limit, "enabled", return_value=False),
            self.assertLogs(REQUEST_LOGGER_NAME, level="INFO") as logs,
        ):
            TestClient(app).get("/api/llm-models")

        payload = json.loads(JsonFormatter().format(logs.records[-1]))
        self.assertEqual(payload["event"], "http_request")
        self.assertEqual(payload["path"], "/api/llm-models")


if __name__ == "__main__":
    unittest.main()
