from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from interface.app import RagRequest, app
from interface.backend import generation, retrieval
from interface.backend.schemas import ExecutionPlan


class InterfaceAppTests(unittest.TestCase):
    def test_bm25_search_is_limited_to_detail_chunks(self) -> None:
        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params):
                if "c.chunk_level = 'detail'" not in sql:
                    raise AssertionError("Le filtre detail est absent.")
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
                        "detail",
                        1,
                        "Contenu detail",
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

        self.assertEqual(chunks[0]["chunk_level"], "detail")
        self.assertEqual(chunks[0]["chunk_parent_id"], 1)

    def test_detail_results_are_expanded_with_section_and_global_context(self) -> None:
        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params):
                if "section.id = detail.chunk_parent_id" not in sql:
                    raise AssertionError("La jointure vers la section est absente.")
                if "global_chunk.id = section.chunk_parent_id" not in sql:
                    raise AssertionError("La jointure vers le global est absente.")
                self.params = params

            def fetchall(self):
                return [
                    (
                        10,
                        20,
                        2,
                        "Resume de section",
                        30,
                        1,
                        "Resume global",
                    )
                ]

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def cursor(self):
                return Cursor()

        chunks = [
            {
                "chunk_id": 10,
                "chunk_level": "detail",
                "chunk_index": 7,
                "text": "Contenu detail",
            }
        ]
        with patch.object(retrieval, "connect_database", return_value=Connection()):
            expanded, trace = retrieval.expand_detail_context(chunks)

        self.assertEqual(
            expanded[0]["section_context"],
            {
                "chunk_id": 20,
                "chunk_index": 2,
                "text": "Resume de section",
            },
        )
        self.assertEqual(
            expanded[0]["global_context"],
            {
                "chunk_id": 30,
                "chunk_index": 1,
                "text": "Resume global",
            },
        )
        self.assertEqual(trace["expanded_count"], 1)

    def test_generation_context_includes_hierarchical_parents(self) -> None:
        text = generation.source_context_text(
            {
                "text": "Contenu detail",
                "section_context": {"text": "Resume de section"},
                "global_context": {"text": "Resume global"},
            }
        )

        self.assertIn("Contenu detail", text)
        self.assertIn("Resume de section", text)
        self.assertIn("Resume global", text)

    def test_public_routes_are_preserved(self) -> None:
        client = TestClient(app)

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()["paths"]),
            {
                "/",
                "/api/rag",
                "/api/rag/stream",
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
