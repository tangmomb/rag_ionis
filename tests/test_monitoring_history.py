from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest import mock

from fastapi.testclient import TestClient

from interface.app import app
from interface.monitoring import history

AT = datetime(2026, 11, 18, 14, 18, tzinfo=timezone.utc)


class HistoryShapeTest(unittest.TestCase):
    def test_rows_become_recall_and_faithfulness_series(self) -> None:
        rows = [
            (AT, "recall_gate", "ok", {"video_recall": 0.8667, "chunk_recall": 0.8, "image_tag": "06aab61abcdef"}),
            (AT, "recall", "anomaly", {"video_recall": 0.74}),
            (AT, "quality", "ok", {"day": "2026-11-17", "canary_faithfulness": 1.0, "canary_relevance": 0.9}),
            (AT, "restore", "ok", {"total_seconds": 16}),
        ]

        result = history.history_from_rows(rows)

        self.assertEqual([point["video_recall"] for point in result["recall"]], [0.8667, 0.74])
        self.assertEqual(result["recall"][0]["commit"], "06aab61")
        self.assertEqual(result["recall"][1]["kind"], "recall")
        self.assertEqual(result["faithfulness"][0]["canary_faithfulness"], 1.0)
        self.assertEqual(result["thresholds"]["recall"], 0.8)

    def test_failed_reports_and_missing_scores_are_skipped(self) -> None:
        rows = [
            (AT, "recall", "failed", {"exit_code": 1}),
            (AT, "recall_gate", "ok", {"exit_code": 0}),
            (AT, "quality", "ok", {"day": "2026-11-17", "canary_faithfulness": None}),
            (AT, "quality", "failed", {"exit_code": 1}),
        ]

        result = history.history_from_rows(rows)

        self.assertEqual((result["recall"], result["faithfulness"]), ([], []))

    def test_public_traffic_series_includes_days_without_a_canary_score(self) -> None:
        rows = [
            (AT, "quality", "ok", {"canary_faithfulness": 1, "traffic_faithfulness": None}),
            (AT, "quality", "anomaly", {
                "day": "2026-11-17", "traffic_faithfulness": 0, "traffic_judged": 2,
                "traffic_relevance": 0.5,
            }),
            (AT, "quality", "failed", {"traffic_faithfulness": 0.8}),
        ]

        result = history.history_from_rows(rows)

        self.assertEqual(len(result["faithfulness"]), 1)
        self.assertEqual(result["traffic_faithfulness"], [{
            "at": AT.isoformat(), "day": "2026-11-17", "status": "anomaly",
            "traffic_faithfulness": 0, "traffic_judged": 2, "traffic_relevance": 0.5,
        }])

    def test_no_message_or_user_data_is_exposed(self) -> None:
        rows = [(AT, "quality", "ok", {"canary_faithfulness": 1.0, "anomalies": ["x"], "flagged": ["question"]})]

        point = history.history_from_rows(rows)["faithfulness"][0]

        self.assertNotIn("anomalies", point)
        self.assertNotIn("flagged", point)


class HistoryEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        history._cache.update(at=0.0, value=None)

    def test_endpoint_serves_and_caches_history(self) -> None:
        payload = history.history_from_rows([(AT, "recall", "ok", {"video_recall": 0.87})])
        with mock.patch.object(history, "load_history", return_value=payload) as load:
            client = TestClient(app)
            first = client.get("/api/monitoring/history").json()
            client.get("/api/monitoring/history")

        self.assertEqual(first["recall"][0]["video_recall"], 0.87)
        load.assert_called_once()

    def test_other_sites_can_read_history_without_credentials(self) -> None:
        payload = history.history_from_rows([])
        with mock.patch.object(history, "load_history", return_value=payload):
            response = TestClient(app).get(
                "/api/monitoring/history",
                headers={"Origin": "https://portfolio-ia.tanguym.fr"},
            )

        self.assertEqual(response.headers["access-control-allow-origin"], "*")

    def test_paid_routes_keep_strict_cors(self) -> None:
        response = TestClient(app).get(
            "/api/llm-models", headers={"Origin": "https://portfolio-ia.tanguym.fr"}
        )

        self.assertNotEqual(response.headers.get("access-control-allow-origin"), "*")

    def test_unavailable_database_returns_empty_series(self) -> None:
        with mock.patch.object(history, "load_history", side_effect=RuntimeError("db down")):
            response = TestClient(app).get("/api/monitoring/history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recall"], [])
        self.assertIn("error", response.json())


if __name__ == "__main__":
    unittest.main()
