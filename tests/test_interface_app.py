from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from interface.app import RagRequest, app
from interface.backend import retrieval
from interface.backend.schemas import ExecutionPlan


class InterfaceAppTests(unittest.TestCase):
    def test_bm25_sources_expose_chunk_hierarchy(self) -> None:
        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params):
                self.sql = sql
                self.params = params

            def fetchall(self):
                return [
                    (
                        10,
                        "Video",
                        "https://example.test/video",
                        None,
                        2,
                        "section",
                        1,
                        "Contenu de section",
                        ["Alice"],
                        0.75,
                    )
                ]

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def cursor(self):
                return Cursor()

        query = ExecutionPlan(
            raw_question="Question",
            query_text="Question",
            query_text_bm25="Question",
        )

        with patch.object(retrieval, "connect_database", return_value=Connection()):
            chunks, _ = retrieval.fetch_bm25_chunks(query, candidate_chunk_ids=None)

        self.assertEqual(chunks[0]["chunk_level"], "section")
        self.assertEqual(chunks[0]["chunk_parent_id"], 1)

    def test_public_routes_are_preserved(self) -> None:
        client = TestClient(app)

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()["paths"]),
            {
                "/",
                "/api/rag",
                "/api/video-thumbnails",
                "/health",
                "/styles.css",
                "/version",
            },
        )

    def test_request_schema_remains_available_from_app(self) -> None:
        payload = RagRequest(question="Bonjour")

        self.assertTrue(payload.useSql)
        self.assertTrue(payload.useRerank)
        self.assertEqual(payload.topK, 40)
        self.assertEqual(payload.finalK, 5)


if __name__ == "__main__":
    unittest.main()
