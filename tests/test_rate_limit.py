from __future__ import annotations

import json
import logging
import os
import unittest
from datetime import datetime, timezone
from unittest import mock

from fastapi.testclient import TestClient

from interface import rate_limit
from interface.app import app
from interface.request_logging import JsonFormatter

LIMITS = rate_limit.Limits(per_minute=10, per_day=50, global_per_day=300)
NOW = datetime(2026, 11, 18, 22, 59, 40, tzinfo=timezone.utc)  # 23:59:40 à Paris


class DecisionTest(unittest.TestCase):
    def windows(self):
        return rate_limit.windows_at(NOW)

    def test_requests_within_every_limit_pass(self) -> None:
        counts = {"minute": 10, "day": 50, "global": 300}

        self.assertIsNone(rate_limit.evaluate(counts, LIMITS, self.windows()))

    def test_burst_is_rejected_until_the_next_minute(self) -> None:
        rejection = rate_limit.evaluate({"minute": 11, "day": 11, "global": 11}, LIMITS, self.windows())

        self.assertEqual(rejection.status_code, 429)
        self.assertEqual(rejection.retry_after_seconds, 20)

    def test_daily_quota_is_rejected_until_paris_midnight(self) -> None:
        rejection = rate_limit.evaluate({"minute": 1, "day": 51, "global": 60}, LIMITS, self.windows())

        self.assertEqual(rejection.status_code, 429)
        self.assertEqual(rejection.retry_after_seconds, 20)
        self.assertIn("50 questions par jour", rejection.detail)

    def test_global_budget_wins_and_returns_503(self) -> None:
        rejection = rate_limit.evaluate({"minute": 11, "day": 51, "global": 301}, LIMITS, self.windows())

        self.assertEqual(rejection.status_code, 503)
        self.assertIn("budget", rejection.detail)


class ConfigurationTest(unittest.TestCase):
    def test_budget_is_converted_into_questions(self) -> None:
        with mock.patch.dict(os.environ, {"RAG_DAILY_BUDGET_USD": "2", "RAG_COST_PER_QUESTION_USD": "0.0065"}):
            self.assertEqual(rate_limit.limits_from_env().global_per_day, 307)

    def test_invalid_values_fall_back_to_defaults(self) -> None:
        with mock.patch.dict(os.environ, {"RAG_RATE_LIMIT_PER_MINUTE": "abc", "RAG_RATE_LIMIT_PER_DAY": "-3"}):
            limits = rate_limit.limits_from_env()

        self.assertEqual((limits.per_minute, limits.per_day), (10, 50))

    def test_enabled_only_in_production_unless_overridden(self) -> None:
        with mock.patch.dict(os.environ, {"RAG_IONIS_ENV": "production", "RAG_RATE_LIMIT_ENABLED": ""}):
            self.assertTrue(rate_limit.enabled())
        with mock.patch.dict(os.environ, {"RAG_IONIS_ENV": "staging", "RAG_RATE_LIMIT_ENABLED": ""}):
            self.assertFalse(rate_limit.enabled())
        with mock.patch.dict(os.environ, {"RAG_IONIS_ENV": "production", "RAG_RATE_LIMIT_ENABLED": "false"}):
            self.assertFalse(rate_limit.enabled())

    def test_ip_is_never_stored_in_clear(self) -> None:
        key = rate_limit.hashed_client("203.0.113.7")

        self.assertNotIn("203.0.113.7", key)
        self.assertEqual(key, rate_limit.hashed_client("203.0.113.7"))
        self.assertNotEqual(key, rate_limit.hashed_client("203.0.113.8"))

    def test_buckets_use_paris_day_and_utc_minute(self) -> None:
        windows = rate_limit.windows_at(NOW)
        buckets = rate_limit.buckets_for("abc", windows)

        self.assertEqual(buckets["global"], ("global-day", datetime(2026, 11, 17, 23, 0, tzinfo=timezone.utc)))
        self.assertEqual(buckets["minute"][1], datetime(2026, 11, 18, 22, 59, tzinfo=timezone.utc))
        self.assertTrue(buckets["day"][0].startswith("ip-day:"))

    def test_paid_paths(self) -> None:
        for path in ("/api/rag", "/api/rag/stream", "/api/rag/messages/12/expand"):
            self.assertTrue(rate_limit.PAID_PATH_PATTERN.match(path), path)
        for path in ("/api/rag/messages/12/feedback", "/api/llm-models", "/health"):
            self.assertFalse(rate_limit.PAID_PATH_PATTERN.match(path), path)

    def test_counter_failure_lets_the_request_through(self) -> None:
        with mock.patch(
            "interface.backend.database.connect_database", side_effect=RuntimeError("db down")
        ):
            self.assertIsNone(rate_limit.check_request("203.0.113.7", now=NOW))


class MiddlewareTest(unittest.TestCase):
    def test_rejected_question_returns_detail_and_retry_after(self) -> None:
        rejection = rate_limit.Rejection(429, "Trop de questions.", 12)
        with (
            mock.patch.object(rate_limit, "enabled", return_value=True),
            mock.patch.object(rate_limit, "check_request", return_value=rejection) as check,
        ):
            response = TestClient(app).post("/api/rag/stream", json={"question": "Bonjour"})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {"detail": "Trop de questions."})
        self.assertEqual(response.headers["Retry-After"], "12")
        self.assertTrue(response.headers["X-Request-ID"])
        check.assert_called_once()

    def test_free_routes_are_not_counted(self) -> None:
        with (
            mock.patch.object(rate_limit, "enabled", return_value=True),
            mock.patch.object(rate_limit, "check_request") as check,
        ):
            response = TestClient(app).get("/api/llm-models")

        check.assert_not_called()
        self.assertTrue(response.headers["X-Request-ID"])

    def test_disabled_outside_production(self) -> None:
        with (
            mock.patch.object(rate_limit, "enabled", return_value=False),
            mock.patch.object(rate_limit, "check_request") as check,
            mock.patch("interface.backend.api.run_rag", side_effect=RuntimeError("stop")),
        ):
            TestClient(app, raise_server_exceptions=False).post("/api/rag", json={"question": "Bonjour"})

        check.assert_not_called()


class JsonLogTest(unittest.TestCase):
    def test_log_line_is_json_with_extra_fields(self) -> None:
        record = logging.makeLogRecord(
            {"name": "rag_ionis.http", "levelname": "INFO", "msg": "http_request", "status": 429, "request_id": "r1"}
        )

        payload = json.loads(JsonFormatter().format(record))

        self.assertEqual(payload["event"], "http_request")
        self.assertEqual(payload["status"], 429)
        self.assertEqual(payload["request_id"], "r1")
        self.assertIn("ts", payload)


if __name__ == "__main__":
    unittest.main()
