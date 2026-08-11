from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from interface.backend.llm_providers import LLMResponse
from utils.app_llm_tester import app as llm_tester


class LlmTesterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(llm_tester.app)

    def test_index_and_static_assets_are_served(self) -> None:
        index = self.client.get("/")
        stylesheet = self.client.get("/static/styles.css")
        script = self.client.get("/static/app.js")

        self.assertEqual(index.status_code, 200)
        self.assertIn("Response tester", index.text)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertEqual(script.status_code, 200)

    def test_config_reports_key_presence_without_exposing_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MISTRAL_API_KEY": "",
            },
            clear=False,
        ):
            response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        providers = {item["id"]: item for item in payload["providers"]}
        self.assertEqual(set(providers), {"mistral"})
        self.assertFalse(providers["mistral"]["configured"])

    def test_missing_key_is_reported_before_network_call(self) -> None:
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": ""}, clear=False),
            patch.object(llm_tester, "create_llm_response") as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "mistral",
                    "model": "mistral-large-latest",
                    "message": "Bonjour",
                    "max_output_tokens": 100,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("MISTRAL_API_KEY", response.text)
        create.assert_not_called()

    def test_mistral_returns_extracted_text_and_raw_payload(self) -> None:
        raw = {
            "choices": [
                {"message": {"content": "Bonjour Mistral"}}
            ]
        }
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="mistral",
                    model="mistral-large-latest",
                    output_text="Bonjour Mistral",
                    raw_payload=raw,
                ),
            ),
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "mistral",
                    "model": "mistral-large-latest",
                    "message": "Bonjour",
                    "max_output_tokens": 200,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Bonjour Mistral")
        self.assertEqual(response.json()["response"], raw)

    def test_other_providers_are_rejected(self) -> None:
        response = self.client.post(
            "/api/generate",
            json={
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "message": "Bonjour",
                "max_output_tokens": 200,
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
