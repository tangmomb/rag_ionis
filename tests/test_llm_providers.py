from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from interface.backend import llm_providers
from interface.backend.utilities import serialize_openai_response


class LlmProviderTests(unittest.TestCase):
    def http_response(self, payload, *, status_code=200):
        response = Mock()
        response.ok = 200 <= status_code < 300
        response.status_code = status_code
        response.json.return_value = payload
        response.text = json.dumps(payload)
        return response

    def test_provider_is_inferred_from_model_identifier(self) -> None:
        self.assertEqual(
            llm_providers.provider_for_model("gpt-5.6-terra"),
            "openai",
        )
        self.assertEqual(
            llm_providers.provider_for_model("mistral-medium-latest"),
            "mistral",
        )
        self.assertEqual(
            llm_providers.provider_for_model("gemini-3.6-flash"),
            "google",
        )

    def test_unknown_model_requires_an_explicit_supported_provider(self) -> None:
        with self.assertRaisesRegex(
            llm_providers.LLMProviderError,
            "Fournisseur impossible",
        ):
            llm_providers.provider_for_model("custom-model")

    def test_openai_uses_responses_api_and_normalizes_the_sdk_response(self) -> None:
        raw = {
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "OpenAI"}
                    ],
                }
            ],
        }
        sdk_response = SimpleNamespace(
            output_text="OpenAI",
            model_dump=lambda mode: raw,
        )
        client = Mock()
        client.responses.create.return_value = sdk_response
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=False),
            patch.object(llm_providers, "OpenAI", return_value=client),
        ):
            response = llm_providers.create_llm_response(
                model="gpt-5.6-sol",
                input="Bonjour",
                max_output_tokens=200,
                store=False,
            )

        self.assertEqual(response.output_text, "OpenAI")
        self.assertEqual(response.raw_payload, raw)
        request = client.responses.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5.6-sol")
        self.assertEqual(
            request["input"],
            [{"role": "user", "content": "Bonjour"}],
        )
        self.assertEqual(request["max_output_tokens"], 200)
        self.assertFalse(request["store"])

    def test_mistral_translates_normalized_messages_to_chat_completions(self) -> None:
        raw = {
            "choices": [
                {"message": {"content": "Mistral"}}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        }
        trace = MagicMock()
        trace_context = MagicMock()
        trace_context.__enter__.return_value = trace
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_providers.requests,
                "post",
                return_value=self.http_response(raw),
            ) as post,
            patch.object(
                llm_providers,
                "trace_operation",
                return_value=trace_context,
            ) as trace_operation,
        ):
            response = llm_providers.create_llm_response(
                model="mistral-medium-latest",
                input=[
                    {"role": "system", "content": "Réponds brièvement."},
                    {"role": "user", "content": "Bonjour"},
                ],
                max_output_tokens=300,
            )

        self.assertEqual(response.output_text, "Mistral")
        self.assertEqual(
            post.call_args.args[0],
            "https://api.mistral.ai/v1/chat/completions",
        )
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["model"], "mistral-medium-latest")
        self.assertEqual(request["max_tokens"], 300)
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(
            trace_operation.call_args.args[0],
            "MistralChatCompletion",
        )
        self.assertEqual(trace_operation.call_args.kwargs["kind"], "LLM")
        self.assertEqual(
            trace_operation.call_args.kwargs["attributes"]["llm.provider"],
            "mistral",
        )
        trace.set_output.assert_called_once_with(raw)
        trace.set_attribute.assert_has_calls(
            [
                call("llm.token_count.prompt", 10),
                call("llm.token_count.completion", 3),
                call("llm.token_count.total", 13),
            ]
        )

    def test_google_separates_system_instruction_from_contents(self) -> None:
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "pensée", "thought": True},
                            {"text": "Google"},
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 4,
                "totalTokenCount": 15,
            },
        }
        trace = MagicMock()
        trace_context = MagicMock()
        trace_context.__enter__.return_value = trace
        with (
            patch.dict(
                os.environ,
                {"GOOGLE_API_KEY": "secret", "GEMINI_API_KEY": ""},
                clear=False,
            ),
            patch.object(
                llm_providers.requests,
                "post",
                return_value=self.http_response(raw),
            ) as post,
            patch.object(
                llm_providers,
                "trace_operation",
                return_value=trace_context,
            ) as trace_operation,
        ):
            response = llm_providers.create_llm_response(
                model="gemini-3.6-flash",
                input=[
                    {"role": "system", "content": "Réponds brièvement."},
                    {"role": "user", "content": "Bonjour"},
                ],
                max_output_tokens=400,
            )

        self.assertEqual(response.output_text, "Google")
        self.assertIn(
            "/models/gemini-3.6-flash:generateContent",
            post.call_args.args[0],
        )
        request = post.call_args.kwargs["json"]
        self.assertEqual(
            request["systemInstruction"]["parts"],
            [{"text": "Réponds brièvement."}],
        )
        self.assertEqual(request["contents"][0]["role"], "user")
        self.assertEqual(
            request["generationConfig"]["maxOutputTokens"],
            400,
        )
        self.assertEqual(
            trace_operation.call_args.args[0],
            "GoogleGenerateContent",
        )
        self.assertEqual(trace_operation.call_args.kwargs["kind"], "LLM")
        self.assertEqual(
            trace_operation.call_args.kwargs["attributes"]["llm.provider"],
            "google",
        )
        trace.set_output.assert_called_once_with(raw)
        trace.set_attribute.assert_has_calls(
            [
                call("llm.token_count.prompt", 11),
                call("llm.token_count.completion", 4),
                call("llm.token_count.total", 15),
            ]
        )

    def test_normalized_response_is_serializable_for_phoenix_traces(self) -> None:
        response = llm_providers.LLMResponse(
            provider="mistral",
            model="mistral-small-latest",
            output_text="Réponse",
            raw_payload={"id": "abc", "usage": {"total_tokens": 12}},
        )

        serialized = serialize_openai_response(response)

        self.assertEqual(
            json.loads(serialized),
            {"id": "abc", "usage": {"total_tokens": 12}},
        )


if __name__ == "__main__":
    unittest.main()
