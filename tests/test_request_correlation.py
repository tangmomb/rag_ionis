from __future__ import annotations

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from interface import rate_limit
from interface.app import app
from interface.backend import api
from interface.backend.request_context import REQUEST_LOGGER_NAME, current_request_id
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


class RunRagCorrelationTest(unittest.TestCase):
    def run_with_request_id(self, request_id: str, **patches):
        token = current_request_id.set(request_id)
        try:
            with mock.patch.object(api, "current_trace_id", return_value="f" * 32), mock.patch.object(
                api, "record_trace_score_annotations"
            ), mock.patch.object(api, "execute_rag", **patches):
                with self.assertLogs(REQUEST_LOGGER_NAME, level="INFO") as logs:
                    try:
                        api.run_rag(RagRequest(question="Bonjour"))
                    except RuntimeError:
                        pass
        finally:
            current_request_id.reset(token)
        return logs.records

    def test_answer_logs_request_id_with_trace_and_message(self) -> None:
        records = self.run_with_request_id("req-1", return_value=fake_response())

        record = records[-1]
        self.assertEqual(record.getMessage(), "rag_answered")
        self.assertEqual(record.request_id, "req-1")
        self.assertEqual(record.trace_id, "f" * 32)
        self.assertEqual((record.conversation_id, record.message_id), (12, 34))
        self.assertIn("rag_completed_ms", record.__dict__)

    def test_failure_logs_trace_then_reraises(self) -> None:
        records = self.run_with_request_id("req-2", side_effect=RuntimeError("Mistral indisponible"))

        record = records[-1]
        self.assertEqual(record.getMessage(), "rag_failed")
        self.assertEqual(record.request_id, "req-2")
        self.assertEqual(record.trace_id, "f" * 32)
        self.assertIn("Mistral indisponible", record.error)

    def test_failure_is_still_raised(self) -> None:
        token = current_request_id.set("req-3")
        try:
            with mock.patch.object(api, "execute_rag", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError), self.assertLogs(REQUEST_LOGGER_NAME):
                    api.run_rag(RagRequest(question="Bonjour"))
        finally:
            current_request_id.reset(token)


class StreamContextTest(unittest.TestCase):
    def test_streaming_worker_sees_the_http_request_id(self) -> None:
        seen: list[str | None] = []

        def capture(payload, **_kwargs):
            seen.append(current_request_id.get())
            return fake_response()

        with (
            mock.patch.object(rate_limit, "enabled", return_value=False),
            mock.patch.object(api, "run_rag", side_effect=capture),
        ):
            response = TestClient(app).post(
                "/api/rag/stream",
                json={"question": "Bonjour"},
                headers={"X-Request-ID": "stream-req-42"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-ID"], "stream-req-42")
        self.assertEqual(seen, ["stream-req-42"])


if __name__ == "__main__":
    unittest.main()
