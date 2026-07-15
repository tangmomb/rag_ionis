from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
INIT_DIR = ROOT_DIR / "scripts" / "init"
if str(INIT_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_DIR))

from common import used_by_hs22_ns22_validate_chunk_speakers as speaker_validation


class FakeResponses:
    def __init__(self, output_texts):
        self.output_texts = iter(output_texts)
        self.calls = []

    def create(self, **body):
        self.calls.append(body)
        return SimpleNamespace(
            output_text=next(self.output_texts),
            output=[],
            status="completed",
            incomplete_details=None,
            error=None,
        )


class ValidateChunkSpeakersTests(unittest.TestCase):
    def test_request_uses_strict_schema_and_minimal_reasoning_for_gpt5(self) -> None:
        speakers = ["Lou-Anne Corveddu", "Ionis-STM"]

        body, request_log = speaker_validation.build_response_request("gpt-5-nano", speakers)

        schema = body["text"]["format"]["schema"]
        self.assertEqual(body["reasoning"], {"effort": "minimal"})
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(schema["properties"]["valid_speakers"]["items"]["enum"], speakers)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(request_log["text"], body["text"])

    def test_parser_accepts_structured_and_legacy_answers(self) -> None:
        speakers = ["Lou-Anne Corveddu", "Ionis-STM"]

        structured = speaker_validation.parse_valid_speakers(
            '{"valid_speakers":["Lou-Anne Corveddu"]}', speakers
        )
        legacy = speaker_validation.parse_valid_speakers('["Lou-Anne Corveddu"]', speakers)

        self.assertEqual(structured, ["Lou-Anne Corveddu"])
        self.assertEqual(legacy, ["Lou-Anne Corveddu"])

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

        with self.assertRaisesRegex(RuntimeError, "aucun texte apres 3 tentatives"):
            speaker_validation.ask_gpt(client, "gpt-5-nano", ["Lou-Anne Corveddu"])

    def test_empty_parser_input_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Reponse OpenAI vide"):
            speaker_validation.parse_valid_speakers("", ["Lou-Anne Corveddu"])


if __name__ == "__main__":
    unittest.main()
