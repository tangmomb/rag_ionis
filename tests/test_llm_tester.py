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
                "OPENAI_API_KEY": "openai-secret",
                "MISTRAL_API_KEY": "",
                "GOOGLE_API_KEY": "google-secret",
                "GEMINI_API_KEY": "",
            },
            clear=False,
        ):
            response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        providers = {item["id"]: item for item in payload["providers"]}
        self.assertTrue(providers["openai"]["configured"])
        self.assertFalse(providers["mistral"]["configured"])
        self.assertTrue(providers["google"]["configured"])
        self.assertNotIn("openai-secret", response.text)
        self.assertNotIn("google-secret", response.text)

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

    def test_openai_returns_extracted_text_and_raw_payload(self) -> None:
        raw = {
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Bonjour OpenAI"}
                    ],
                }
            ],
        }
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="openai",
                    model="gpt-5.6-sol",
                    output_text="Bonjour OpenAI",
                    raw_payload=raw,
                ),
            ) as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "openai",
                    "model": "gpt-5.6-sol",
                    "message": "Bonjour",
                    "max_output_tokens": 200,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Bonjour OpenAI")
        self.assertEqual(response.json()["response"], raw)
        self.assertEqual(
            create.call_args.kwargs["max_output_tokens"],
            200,
        )
        self.assertNotIn("secret", str(create.call_args.kwargs))

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

    def test_google_returns_extracted_text_and_raw_payload(self) -> None:
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "raisonnement", "thought": True},
                            {"text": "Bonjour Google"},
                        ]
                    }
                }
            ]
        }
        with (
            patch.dict(
                os.environ,
                {"GOOGLE_API_KEY": "secret", "GEMINI_API_KEY": ""},
                clear=False,
            ),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="google",
                    model="gemini-3.6-flash",
                    output_text="Bonjour Google",
                    raw_payload=raw,
                ),
            ) as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "google",
                    "model": "models/gemini-3.6-flash",
                    "message": "Bonjour",
                    "max_output_tokens": 200,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Bonjour Google")
        self.assertEqual(response.json()["response"], raw)
        self.assertEqual(create.call_args.kwargs["provider"], "google")
        self.assertEqual(
            create.call_args.kwargs["model"],
            "models/gemini-3.6-flash",
        )


if __name__ == "__main__":
    unittest.main()
