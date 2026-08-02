from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.speakers import validate_speakers as speaker_validation
from pipeline.options import PipelineOptions


class FakeResponses:
    def __init__(self, output_texts):
        self.output_texts = iter(output_texts)
        self.calls = []

    def create(self, **body):
        self.calls.append(body)
        response = next(self.output_texts)
        if isinstance(response, dict):
            return SimpleNamespace(
                output_text=response.get("output_text", ""),
                output=[],
                status=response.get("status", "completed"),
                incomplete_details=response.get("incomplete_details"),
                error=response.get("error"),
            )
        return SimpleNamespace(
            output_text=response,
            output=[],
            status="completed",
            incomplete_details=None,
            error=None,
        )


class FakeChatCompletions:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.calls = []

    def create(self, **body):
        self.calls.append(body)
        content, finish_reason = next(self.choices)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )


class ValidateChunkSpeakersTests(unittest.TestCase):
    def test_speaker_validation_uses_luna_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                PipelineOptions().speaker_validation_model,
                "gpt-5.6-luna",
            )

    def test_request_uses_strict_schema_without_reasoning_override(self) -> None:
        speakers = ["Lou-Anne Corveddu", "Ionis-STM"]
        candidates = [
            {"name": "Lou-Anne Corveddu", "methods": ["ocr_lower_third"]},
            {"name": "Ionis-STM", "methods": ["ocr_lower_third"]},
        ]

        body, request_log = speaker_validation.build_response_request(
            "gpt-5.6-luna",
            speakers,
            "Lou-Anne Corvedu présente son métier",
            candidates,
            ["Lou-Anne Corveddu", "Responsable marketing"],
            2,
        )

        schema = body["text"]["format"]["schema"]
        self.assertNotIn("reasoning", body)
        self.assertNotIn("reasoning", request_log)
        self.assertTrue(body["text"]["format"]["strict"])
        speaker_item = schema["properties"]["speakers"]["items"]
        self.assertEqual(speaker_item["type"], "object")
        self.assertEqual(speaker_item["required"], ["speaker", "title"])
        self.assertFalse(speaker_item["additionalProperties"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(request_log["text"], body["text"])
        self.assertEqual(body["max_output_tokens"], 2048)
        self.assertEqual(request_log["max_output_tokens"], 2048)
        prompt = body["input"][1]["content"]
        self.assertIn('"video_title"', prompt)
        self.assertIn("Lou-Anne Corvedu présente son métier", prompt)
        self.assertNotIn('"methods"', prompt)
        self.assertNotIn("ocr_lower_third", prompt)
        self.assertIn("ocr_detected_texts", prompt)
        self.assertNotIn("expected_speaker_count", prompt)
        self.assertNotIn("speaker(s) distinct(s)", prompt)
        self.assertIn("Lou-Anne Corveddu", prompt)
        self.assertIn("ajoute-les", prompt)
        self.assertIn("Recherche et extrais explicitement", prompt)
        self.assertIn("nom de l'entreprise", prompt)
        self.assertIn("N'omets pas l'entreprise", prompt)
        self.assertIn("CTO - Mappy.com", prompt)
        self.assertLess(
            prompt.index('"video_title"'),
            prompt.index('"transcript_excerpt"'),
        )
        self.assertLess(
            prompt.index('"ocr_detected_texts"'),
            prompt.index('"candidates"'),
        )

    def test_request_sends_transcript_excerpt_to_luna(self) -> None:
        excerpt = "Bonjour, je suis Alice Martin, directrice." * 20
        body, _request_log = speaker_validation.build_response_request(
            "gpt-5.6-luna",
            [],
            transcript_excerpt=excerpt,
        )

        prompt = body["input"][1]["content"]
        self.assertIn('"transcript_excerpt"', prompt)
        self.assertIn(excerpt, prompt)
        self.assertLess(
            prompt.index('"transcript_excerpt"'),
            prompt.index('"ocr_detected_texts"'),
        )

    def test_short_title_name_corrects_first_name_without_dropping_surname(self) -> None:
        valid = speaker_validation.preserve_candidate_name_parts(
            ["Matthieu"],
            [
                {
                    "name": "Mathieu Dumontier",
                    "methods": ["transcript_je_m_appelle"],
                }
            ],
            "Portrait de Matthieu",
        )

        self.assertEqual(valid, ["Matthieu Dumontier"])

    def test_title_spelling_corrects_complete_candidate_without_dropping_surname(self) -> None:
        valid = speaker_validation.preserve_candidate_name_parts(
            ["Mathieu Dumontier"],
            [
                {
                    "name": "Mathieu Dumontier",
                    "methods": ["transcript_je_m_appelle"],
                }
            ],
            "Apporter ma pierre à l'édifice – Matthieu, Responsable Affaires",
        )

        self.assertEqual(valid, ["Matthieu Dumontier"])

    def test_complete_model_name_is_kept(self) -> None:
        valid = speaker_validation.preserve_candidate_name_parts(
            ["Lou-Anne Corvedu"],
            [{"name": "Lou-Anne Corveddu", "methods": ["ocr_lower_third"]}],
        )

        self.assertEqual(valid, ["Lou-Anne Corvedu"])

    def test_unrelated_short_name_is_not_expanded(self) -> None:
        valid = speaker_validation.preserve_candidate_name_parts(
            ["Alice"],
            [{"name": "Mathieu Dumontier", "methods": ["transcript_je_m_appelle"]}],
        )

        self.assertEqual(valid, ["Alice"])

    def test_parser_accepts_structured_and_legacy_answers(self) -> None:
        speakers = ["Lou-Anne Corveddu", "Ionis-STM"]

        structured = speaker_validation.parse_valid_speakers(
            '{"valid_speakers":["Lou-Anne Corveddu"]}'
        )
        legacy = speaker_validation.parse_valid_speakers('["Lou-Anne Corveddu"]')

        self.assertEqual(structured, ["Lou-Anne Corveddu"])
        self.assertEqual(legacy, ["Lou-Anne Corveddu"])

    def test_parser_keeps_speaker_titles(self) -> None:
        details = speaker_validation.parse_speaker_details(
            '{"speakers":[{"speaker":"Cyril Morcrette",'
            '"title":"Country Manager France-Benelux-Switzerland - Desigual"}]}'
        )

        self.assertEqual(
            details,
            [
                {
                    "speaker": "Cyril Morcrette",
                    "title": (
                        "Country Manager France-Benelux-Switzerland - Desigual"
                    ),
                }
            ],
        )

    def test_model_output_is_kept_without_matching_the_original_candidate(self) -> None:
        valid = speaker_validation.parse_valid_speakers(
            '{"valid_speakers":["Loucif Ouyahia"]}'
        )

        self.assertEqual(valid, ["Loucif Ouyahia"])

    def test_model_output_is_not_cleaned_or_deduplicated(self) -> None:
        valid = speaker_validation.parse_valid_speakers(
            '{"valid_speakers":["  Loucif   Ouyahia  ","  Loucif   Ouyahia  ",""]}'
        )

        self.assertEqual(valid, ["  Loucif   Ouyahia  ", "  Loucif   Ouyahia  ", ""])

    def test_empty_response_is_retried(self) -> None:
        responses = FakeResponses(["", '{"valid_speakers":["Lou-Anne Corveddu"]}'])
        client = SimpleNamespace(responses=responses)

        answer, request_log = speaker_validation.ask_gpt(client, "gpt-5-nano", ["Lou-Anne Corveddu"])

        self.assertEqual(answer, '{"valid_speakers":["Lou-Anne Corveddu"]}')
        self.assertEqual(request_log["attempt_count"], 2)
        self.assertEqual(len(responses.calls), 2)

    def test_empty_response_after_all_attempts_has_clear_error(self) -> None:
        responses = FakeResponses(["", "", ""])
        client = SimpleNamespace(responses=responses)

        with self.assertRaisesRegex(RuntimeError, "aucun JSON complet apres 3 tentatives"):
            speaker_validation.ask_gpt(client, "gpt-5-nano", ["Lou-Anne Corveddu"])

    def test_truncated_json_is_retried(self) -> None:
        responses = FakeResponses(
            [
                '{"speakers":[{"speaker":"Gilles Babinet","title":"Digital Champion',
                '{"speakers":[{"speaker":"Gilles Babinet","title":"Digital Champion"}]}',
            ]
        )
        client = SimpleNamespace(responses=responses)

        answer, request_log = speaker_validation.ask_gpt(
            client,
            "gpt-5-nano",
            ["Gilles Babinet"],
        )

        self.assertIn('"Digital Champion"}]}', answer)
        self.assertEqual(request_log["attempt_count"], 2)
        self.assertEqual(len(responses.calls), 2)

    def test_incomplete_response_status_is_retried(self) -> None:
        responses = FakeResponses(
            [
                {
                    "output_text": '{"speakers":[]}',
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
                '{"speakers":[]}',
            ]
        )
        client = SimpleNamespace(responses=responses)

        answer, request_log = speaker_validation.ask_gpt(
            client,
            "gpt-5-nano",
            [],
        )

        self.assertEqual(answer, '{"speakers":[]}')
        self.assertEqual(request_log["attempt_count"], 2)
        self.assertEqual(len(responses.calls), 2)

    def test_invalid_json_after_all_attempts_has_clear_error(self) -> None:
        responses = FakeResponses(["{", "{", "{"])
        client = SimpleNamespace(responses=responses)

        with self.assertRaisesRegex(
            RuntimeError,
            "aucun JSON complet apres 3 tentatives",
        ):
            speaker_validation.ask_gpt(
                client,
                "gpt-5-nano",
                ["Gilles Babinet"],
            )

    def test_chat_completion_length_finish_is_retried(self) -> None:
        completions = FakeChatCompletions(
            [
                (
                    '{"speakers":[{"speaker":"Gilles Babinet"',
                    "length",
                ),
                (
                    '{"speakers":[{"speaker":"Gilles Babinet","title":""}]}',
                    "stop",
                ),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        answer, request_log = speaker_validation.ask_gpt(
            client,
            "gpt-5-nano",
            ["Gilles Babinet"],
        )

        self.assertIn('"Gilles Babinet"', answer)
        self.assertEqual(request_log["attempt_count"], 2)
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(
            completions.calls[0]["max_completion_tokens"],
            2048,
        )

    def test_empty_parser_input_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Reponse OpenAI vide"):
            speaker_validation.parse_valid_speakers("")


if __name__ == "__main__":
    unittest.main()
