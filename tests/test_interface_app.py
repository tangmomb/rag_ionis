from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from interface.app import RagRequest, app


class InterfaceAppTests(unittest.TestCase):
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
