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
            llm_providers.provider_for_model("zai-glm-5-2"),
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

    def test_structured_schema_has_a_function_name_for_langchain(self) -> None:
        schema = llm_providers.langchain_response_schema({"type": "object"})

        self.assertEqual(schema["title"], "rag_response")
        self.assertEqual(schema["type"], "object")

    def test_manual_llm_span_exposes_phoenix_message_cards_and_token_usage(self) -> None:
        operation = MagicMock()
        trace_context = MagicMock()
        trace_context.__enter__.return_value = operation
        response = SimpleNamespace(
            content="Réponse claire",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 4,
                "total_tokens": 16,
            },
        )
        messages = [
            {"role": "system", "content": "Sois bref."},
            {"role": "user", "content": "Bonjour"},
        ]

        with patch.object(
            llm_providers,
            "trace_operation",
            return_value=trace_context,
        ) as trace:
            result = llm_providers.invoke_langchain_model(
                "openai",
                "gpt-5.6-terra",
                messages,
                lambda: response,
                invocation_parameters={"model": "gpt-5.6-terra"},
            )

        self.assertIs(result, response)
        trace.assert_called_once_with(
            "gpt-5.6-terra",
            kind="LLM",
            input_value=messages,
            attributes={
                "llm.provider": "openai",
                "llm.system": "openai",
                "llm.model_name": "gpt-5.6-terra",
                "llm.invocation_parameters": {"model": "gpt-5.6-terra"},
            },
        )
        attributes = {
            item.args[0]: item.args[1]
            for item in operation.set_attribute.call_args_list
        }
        self.assertEqual(
            attributes["llm.input_messages.0.message.content"],
            "Sois bref.",
        )
        self.assertEqual(
            attributes["llm.output_messages.0.message.content"],
            "Réponse claire",
        )
        self.assertEqual(attributes["llm.token_count.prompt"], 12)
        self.assertEqual(attributes["llm.token_count.completion"], 4)
        self.assertEqual(attributes["llm.token_count.total"], 16)

    def test_mistral_span_exposes_the_chat_completion_message(self) -> None:
        operation = MagicMock()
        trace_context = MagicMock()
        trace_context.__enter__.return_value = operation
        response = MagicMock()
        response.model_dump.return_value = {
            "choices": [{"message": {"content": "Réponse Mistral"}}]
        }

        with patch.object(
            llm_providers,
            "trace_operation",
            return_value=trace_context,
        ):
            llm_providers.invoke_langchain_model(
                "mistral",
                "zai-glm-5-2",
                [{"role": "user", "content": "Bonjour"}],
                lambda: response,
            )

        attributes = {
            item.args[0]: item.args[1]
            for item in operation.set_attribute.call_args_list
        }
        self.assertEqual(
            attributes["llm.output_messages.0.message.content"],
            "Réponse Mistral",
        )

    def test_traced_invocation_parameters_exclude_credentials_and_timeouts(self) -> None:
        parameters = llm_providers.traced_invocation_parameters(
            {
                "model": "gpt-5.6-terra",
                "api_key": "secret",
                "timeout": 300,
                "max_completion_tokens": 500,
            }
        )

        self.assertEqual(
            parameters,
            {"model": "gpt-5.6-terra", "max_completion_tokens": 500},
        )

    def test_every_llm_invocation_gets_its_own_structured_phoenix_attributes(self) -> None:
        operations = [MagicMock(), MagicMock()]
        contexts = []
        for operation in operations:
            context = MagicMock()
            context.__enter__.return_value = operation
            contexts.append(context)
        responses = [
            SimpleNamespace(content="Première réponse", usage_metadata={}),
            SimpleNamespace(content="Deuxième réponse", usage_metadata={}),
        ]

        with patch.object(
            llm_providers,
            "trace_operation",
            side_effect=contexts,
        ) as trace:
            for response in responses:
                llm_providers.invoke_langchain_model(
                    "openai",
                    "gpt-5.6-terra",
                    [{"role": "user", "content": "Question"}],
                    lambda response=response: response,
                )

        self.assertEqual(trace.call_count, 2)
        for operation, expected_output in zip(
            operations,
            ("Première réponse", "Deuxième réponse"),
        ):
            attributes = {
                item.args[0]: item.args[1]
                for item in operation.set_attribute.call_args_list
            }
            self.assertEqual(
                attributes["llm.input_messages.0.message.content"],
                "Question",
            )
            self.assertEqual(
                attributes["llm.output_messages.0.message.content"],
                expected_output,
            )

    def test_openai_uses_langchain_chat_model_and_normalizes_response(self) -> None:
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
        sdk_response = SimpleNamespace(content="OpenAI", model_dump=lambda mode: raw)
        client = Mock()
        client.invoke.return_value = sdk_response
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "secret",
                    "OPENAI_SERVICE_TIER": "",
                },
                clear=False,
            ),
            patch.object(llm_providers, "ChatOpenAI", return_value=client) as chat_openai,
        ):
            response = llm_providers.create_llm_response(
                model="gpt-5.6-sol",
                input="Bonjour",
                max_output_tokens=200,
                store=False,
            )

        self.assertEqual(response.output_text, "OpenAI")
        self.assertEqual(response.raw_payload, raw)
        request = chat_openai.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5.6-sol")
        self.assertEqual(
            client.invoke.call_args.args[0],
            [{"role": "user", "content": "Bonjour"}],
        )
        self.assertEqual(request["max_completion_tokens"], 200)
        self.assertFalse(request["store"])
        self.assertEqual(request["service_tier"], "fast")

    def test_openai_forwards_reasoning_and_verbosity(self) -> None:
        sdk_response = SimpleNamespace(content="OpenAI", model_dump=lambda mode: {})
        client = Mock()
        client.invoke.return_value = sdk_response
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=False),
            patch.object(llm_providers, "ChatOpenAI", return_value=client) as chat_openai,
        ):
            llm_providers.create_llm_response(
                model="gpt-5.6-luna",
                input="Bonjour",
                reasoning_effort="none",
                verbosity="low",
            )

        request = chat_openai.call_args.kwargs
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(request["verbosity"], "low")
        self.assertTrue(request["use_responses_api"])

    def test_openai_uses_configured_regional_base_url(self) -> None:
        sdk_response = SimpleNamespace(content="OpenAI", model_dump=lambda mode: {})
        client = Mock()
        client.invoke.return_value = sdk_response
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=False),
            patch.object(llm_providers, "ChatOpenAI", return_value=client) as chat_openai,
        ):
            llm_providers.create_llm_response(
                model="gpt-5.6-sol",
                input="Bonjour",
                openai_base_url="https://eu.api.openai.com/v1",
            )

        self.assertEqual(chat_openai.call_args.kwargs["base_url"], "https://eu.api.openai.com/v1")

    def test_openai_request_tier_overrides_environment_tier(self) -> None:
        sdk_response = SimpleNamespace(content="OpenAI", model_dump=lambda mode: {})
        client = Mock()
        client.invoke.return_value = sdk_response
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "secret", "OPENAI_SERVICE_TIER": "default"},
                clear=False,
            ),
            patch.object(llm_providers, "ChatOpenAI", return_value=client) as chat_openai,
        ):
            llm_providers.create_llm_response(
                model="gpt-5.6-sol",
                input="Bonjour",
                openai_service_tier="fast",
            )

        self.assertEqual(chat_openai.call_args.kwargs["service_tier"], "fast")

    def test_openai_uses_configured_fast_service_tier(self) -> None:
        sdk_response = SimpleNamespace(content="OpenAI", model_dump=lambda mode: {"service_tier": "priority"})
        client = Mock()
        client.invoke.return_value = sdk_response
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "secret",
                    "OPENAI_SERVICE_TIER": "fast",
                },
                clear=False,
            ),
            patch.object(llm_providers, "ChatOpenAI", return_value=client) as chat_openai,
        ):
            llm_providers.create_llm_response(
                model="gpt-5.6-sol",
                input="Bonjour",
            )

        request = chat_openai.call_args.kwargs
        self.assertEqual(request["service_tier"], "fast")

    def test_mistral_uses_official_chat_sdk(self) -> None:
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
        sdk_response = SimpleNamespace(model_dump=lambda mode: raw)
        client = Mock()
        client.chat.complete.return_value = sdk_response
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_providers,
                "Mistral",
                return_value=client,
            ) as mistral,
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
        mistral.assert_called_once_with(
            api_key="secret",
            timeout_ms=llm_providers.REQUEST_TIMEOUT_SECONDS * 1000,
        )
        request = client.chat.complete.call_args.kwargs
        self.assertEqual(request["model"], "mistral-medium-latest")
        self.assertEqual(request["max_tokens"], 300)
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(response.raw_payload, raw)

    def test_mistral_runtime_limits_are_configurable(self) -> None:
        sdk_response = SimpleNamespace(model_dump=lambda mode: {})
        client = Mock()
        client.chat.complete.return_value = sdk_response
        with (
            patch.dict(
                os.environ,
                {
                    "MISTRAL_API_KEY": "secret",
                    llm_providers.LLM_REQUEST_TIMEOUT_ENV: "45",
                    llm_providers.LLM_MAX_RETRIES_ENV: "0",
                },
                clear=False,
            ),
            patch.object(
                llm_providers,
                "Mistral",
                return_value=client,
            ) as mistral,
        ):
            llm_providers.create_llm_response(
                model="mistral-medium-latest",
                input=[{"role": "user", "content": "Bonjour"}],
            )

        self.assertEqual(mistral.call_args.kwargs["timeout_ms"], 45000)

    def test_mistral_does_not_set_an_inference_region(self) -> None:
        sdk_response = SimpleNamespace(model_dump=lambda mode: {})
        client = Mock()
        client.chat.complete.return_value = sdk_response
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "secret"}, clear=False),
            patch.object(llm_providers, "Mistral", return_value=client) as mistral,
        ):
            llm_providers.create_llm_response(
                model="mistral-large-latest",
                input="Bonjour",
            )

        self.assertNotIn("base_url", mistral.call_args.kwargs)

    def test_mistral_rejects_missing_structured_result(self) -> None:
        raw = SimpleNamespace(model_dump=lambda mode: {"choices": [{"message": {"content": "texte non structure"}}]})
        client = Mock()
        client.chat.complete.return_value = raw
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "secret"}, clear=False),
            patch.object(llm_providers, "Mistral", return_value=client),
        ):
            with self.assertRaisesRegex(
                llm_providers.LLMProviderError,
                "objet JSON structuré attendu",
            ):
                llm_providers.create_llm_response(
                    model="mistral-medium-latest",
                    input="Bonjour",
                    response_schema={"type": "object"},
                )

    def test_google_uses_google_genai_sdk(self) -> None:
        raw = {"id": "google-response"}
        sdk_response = SimpleNamespace(text="Google", model_dump=lambda mode: raw)
        client = Mock()
        client.models.generate_content.return_value = sdk_response
        operation = MagicMock()
        trace_context = MagicMock()
        trace_context.__enter__.return_value = operation
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "secret", "GEMINI_API_KEY": ""}, clear=False),
            patch.object(llm_providers.genai, "Client", return_value=client) as google_client,
            patch.object(
                llm_providers,
                "trace_operation",
                return_value=trace_context,
            ),
        ):
            response = llm_providers.create_llm_response(
                model="gemini-3.5-flash-lite",
                input=[
                    {"role": "system", "content": "Réponds brièvement."},
                    {"role": "user", "content": "Bonjour"},
                ],
                max_output_tokens=400,
            )

        self.assertEqual(response.output_text, "Google")
        self.assertEqual(google_client.call_args.kwargs["api_key"], "secret")
        self.assertEqual(
            google_client.call_args.kwargs["http_options"].timeout,
            300_000,
        )
        self.assertEqual(
            google_client.call_args.kwargs["http_options"].retry_options.attempts,
            5,
        )
        request = client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-3.5-flash-lite")
        self.assertEqual(request["config"].max_output_tokens, 400)
        self.assertEqual(request["config"].system_instruction, "Réponds brièvement.")
        self.assertEqual(request["config"].service_tier, "priority")
        self.assertEqual(request["config"].thinking_config.thinking_level.value, "LOW")
        attributes = {
            item.args[0]: item.args[1]
            for item in operation.set_attribute.call_args_list
        }
        self.assertEqual(attributes["google.thinking_level.requested"], "low")

    def test_google_forwards_thinking_budget(self) -> None:
        sdk_response = SimpleNamespace(text="Google", model_dump=lambda mode: {})
        client = Mock()
        client.models.generate_content.return_value = sdk_response
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "secret"}, clear=False),
            patch.object(llm_providers.genai, "Client", return_value=client),
        ):
            llm_providers.create_llm_response(
                model="gemini-3.6-flash",
                input="Bonjour",
                thinking_budget=1024,
            )

        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.thinking_config.thinking_budget, 1024)

    def test_gemini_38_flash_uses_server_default_thinking(self) -> None:
        sdk_response = SimpleNamespace(text="Google", model_dump=lambda mode: {})
        client = Mock()
        client.models.generate_content.return_value = sdk_response
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "secret"}, clear=False),
            patch.object(
                llm_providers.genai,
                "Client",
                return_value=client,
            ),
        ):
            llm_providers.create_llm_response(
                model="gemini-3.8-flash",
                input="Bonjour",
            )

        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.thinking_config.thinking_level.value, "LOW")
        self.assertEqual(config.service_tier, "priority")

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
