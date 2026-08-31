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
                "OPENAI_API_KEY": "",
                "GOOGLE_API_KEY": "",
                "GEMINI_API_KEY": "",
            },
            clear=False,
        ):
            response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        providers = {item["id"]: item for item in payload["providers"]}
        self.assertEqual(set(providers), {"openai", "mistral", "google"})
        self.assertFalse(providers["mistral"]["configured"])
        self.assertFalse(providers["openai"]["configured"])
        self.assertFalse(providers["google"]["configured"])
        self.assertEqual(providers["google"]["label"], "Gemini")
        self.assertIn("gpt-5.6-sol", providers["openai"]["models"])
        self.assertEqual(
            providers["openai"]["generationControls"]["verbosity"],
            ["low", "medium", "high"],
        )
        self.assertIn("zai-glm-5-2", providers["mistral"]["models"])
        self.assertEqual(
            [region["id"] for region in providers["mistral"]["regions"]],
            ["global", "eu", "us"],
        )
        self.assertEqual(
            [region["id"] for region in providers["openai"]["regions"]],
            ["global", "eu", "us"],
        )
        self.assertIn("gemini-3.6-flash", providers["google"]["models"])

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

    def test_openai_and_google_providers_are_accepted(self) -> None:
        for provider, model, key_name in (
            ("openai", "gpt-5.6-sol", "OPENAI_API_KEY"),
            ("google", "gemini-3.6-flash", "GOOGLE_API_KEY"),
        ):
            with (
                patch.dict(os.environ, {key_name: "secret"}, clear=False),
                patch.object(
                    llm_tester,
                    "create_llm_response",
                    return_value=LLMResponse(
                        provider=provider,
                        model=model,
                        output_text="Bonjour",
                        raw_payload={"content": "Bonjour"},
                    ),
                ) as create,
            ):
                response = self.client.post(
                    "/api/generate",
                    json={
                        "provider": provider,
                        "model": model,
                        "message": "Bonjour",
                        "max_output_tokens": 200,
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["provider"], provider)
            create.assert_called_once()

    def test_unknown_provider_is_rejected(self) -> None:
        response = self.client.post(
            "/api/generate",
            json={
                "provider": "unknown",
                "model": "unknown-model",
                "message": "Bonjour",
                "max_output_tokens": 200,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_generation_options_are_forwarded(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="openai",
                    model="gpt-5.6-luna",
                    output_text="Bonjour",
                    raw_payload={},
                ),
            ) as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "openai",
                    "model": "gpt-5.6-luna",
                    "message": "Bonjour",
                    "reasoning_effort": "none",
                    "verbosity": "low",
                    "openai_service_tier": "default",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "none")
        self.assertEqual(create.call_args.kwargs["verbosity"], "low")
        self.assertEqual(create.call_args.kwargs["openai_service_tier"], "default")
        self.assertNotIn("response_schema", create.call_args.kwargs)

    def test_system_message_is_sent_before_user_message(self) -> None:
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="mistral",
                    model="mistral-large-latest",
                    output_text="Bonjour",
                    raw_payload={},
                ),
            ) as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "mistral",
                    "model": "mistral-large-latest",
                    "system_message": "Réponds en français.",
                    "message": "Bonjour",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            create.call_args.kwargs["input"],
            [
                {"role": "system", "content": "Réponds en français."},
                {"role": "user", "content": "Bonjour"},
            ],
        )

    def test_gemini_thinking_budget_is_forwarded(self) -> None:
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="google",
                    model="gemini-2.5-flash",
                    output_text="Bonjour",
                    raw_payload={},
                ),
            ) as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "google",
                    "model": "gemini-2.5-flash",
                    "message": "Bonjour",
                    "thinking_budget": 1024,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(create.call_args.kwargs["thinking_budget"], 1024)

    def test_mistral_region_is_forwarded_as_base_url(self) -> None:
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="mistral",
                    model="mistral-large-latest",
                    output_text="Bonjour",
                    raw_payload={},
                ),
            ) as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "mistral",
                    "model": "mistral-large-latest",
                    "message": "Bonjour",
                    "mistral_region": "eu",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            create.call_args.kwargs["mistral_base_url"],
            "https://api.eu.mistral.ai/v1",
        )

    def test_openai_region_is_forwarded_as_base_url(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="openai",
                    model="gpt-5.6-sol",
                    output_text="Bonjour",
                    raw_payload={},
                ),
            ) as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "openai",
                    "model": "gpt-5.6-sol",
                    "message": "Bonjour",
                    "openai_region": "eu",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            create.call_args.kwargs["openai_base_url"],
            "https://eu.api.openai.com/v1",
        )

    def test_openai_fast_tier_is_forwarded(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_tester,
                "create_llm_response",
                return_value=LLMResponse(
                    provider="openai",
                    model="gpt-5.6-sol",
                    output_text="Bonjour",
                    raw_payload={},
                ),
            ) as create,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "provider": "openai",
                    "model": "gpt-5.6-sol",
                    "message": "Bonjour",
                    "openai_service_tier": "fast",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(create.call_args.kwargs["openai_service_tier"], "fast")


if __name__ == "__main__":
    unittest.main()
