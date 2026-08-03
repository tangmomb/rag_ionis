from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from interface.backend import api
from interface.backend.generation import (
    generate_answer,
    generate_person_clarification_answer,
    parse_answer_output,
)
from interface.backend.schemas import RagRequest


class _Responses:
    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(self.outputs.pop(0), ensure_ascii=False)
        )


class _Client:
    def __init__(self, outputs: list[dict]) -> None:
        self.responses = _Responses(outputs)


def _source() -> dict:
    return {
        "chunk_id": 7,
        "video_title": "Vidéo test",
        "video_url": "https://example.test/video",
        "thumbnail_medium_url": None,
        "chunk_index": 1,
        "chunk_level": "detail",
        "chunk_parent_id": None,
        "bm25_score": None,
        "vector_score": None,
        "rrf_score": 0.8,
        "cohere_relevance_score": 0.9,
        "rank_sources": {},
        "text": "La vidéo a été publiée le 12 avril 2022.",
        "persons": [],
        "person_details": [],
        "transcript": None,
        "video_description": None,
        "video_type": "interview",
        "section_context": None,
        "global_context": None,
    }


class AnswerActionTests(unittest.TestCase):
    def test_generator_output_exposes_answer_and_action(self) -> None:
        trace: dict[str, str] = {}

        answer = parse_answer_output(
            '{"answer":"Peux-tu préciser la vidéo ?","action":"clarify"}',
            trace,
        )

        self.assertEqual(answer, "Peux-tu préciser la vidéo ?")
        self.assertEqual(trace["action"], "clarify")

    def test_invalid_or_missing_action_falls_back_to_abstention(self) -> None:
        trace: dict[str, str] = {}

        answer = parse_answer_output('{"answer":"Réponse non qualifiée"}', trace)

        self.assertEqual(answer, "Réponse non qualifiée")
        self.assertEqual(trace["action"], "abstain")

    def test_no_source_is_still_submitted_to_the_answer_model(self) -> None:
        client = _Client(
            [{"answer": "De quelle vidéo parles-tu ?", "action": "clarify"}]
        )
        trace: dict[str, str] = {}

        answer = generate_answer(
            client,
            "Quelle est sa date de publication ?",
            "mistral-medium-latest",
            [],
            trace,
        )

        self.assertEqual(answer, "De quelle vidéo parles-tu ?")
        self.assertEqual(trace["action"], "clarify")
        self.assertEqual(len(client.responses.calls), 1)
        self.assertIn(
            "Aucune source exploitable",
            client.responses.calls[0]["input"][1]["content"],
        )

    def test_ambiguous_person_candidates_are_given_to_the_answer_model(self) -> None:
        client = _Client(
            [{"answer": "Parles-tu d'Alice Martin ou d'Alice Durand ?", "action": "clarify"}]
        )
        trace: dict[str, str] = {}

        answer = generate_person_clarification_answer(
            client,
            "Que fait Alice ?",
            "mistral-medium-latest",
            {
                "ambiguous": True,
                "suggestions": ["Alice Martin", "Alice Durand"],
            },
            trace,
        )

        self.assertEqual(answer, "Parles-tu d'Alice Martin ou d'Alice Durand ?")
        self.assertEqual(trace["action"], "clarify")
        prompt = client.responses.calls[0]["input"][1]["content"]
        self.assertIn("Alice Martin", prompt)
        self.assertIn("Alice Durand", prompt)

    def test_execute_rag_uses_the_action_from_the_generation_call(self) -> None:
        retrieval = {
            "route": "rag",
            "retrieval_mode": "rag+structured_sql",
            "contextual_question": "Quand la vidéo a-t-elle été publiée ?",
            "answer_model": "mistral-medium-latest",
        }

        def generate(*args, **kwargs):
            trace = args[5]
            trace["action"] = "answer"
            return "Le 12 avril 2022. [S1]"

        with (
            patch.object(api, "orchestrate_request", return_value=("", [_source()], retrieval)),
            patch.object(api, "get_llm_client", return_value=object()),
            patch.object(api, "generate_final_answer", side_effect=generate) as generator,
            patch.object(api, "store_chat_message", return_value=(3, 9)),
        ):
            response = api.execute_rag(RagRequest(question="Date de publication ?"))

        self.assertEqual(response.action, "answer")
        self.assertEqual(response.answer, "Le 12 avril 2022.")
        self.assertEqual(len(response.sources), 1)
        self.assertNotIn("source_evaluation", response.retrieval)
        self.assertNotIn("answer_evaluation", response.retrieval)
        generator.assert_called_once()

    def test_precomputed_direct_answer_keeps_answer_action(self) -> None:
        retrieval = {
            "route": "direct",
            "retrieval_mode": "direct",
            "contextual_question": "Bonjour",
            "answer_model": None,
        }
        with (
            patch.object(
                api,
                "orchestrate_request",
                return_value=("Bonjour !", [], retrieval),
            ),
            patch.object(api, "get_llm_client", return_value=None),
            patch.object(api, "generate_final_answer") as generator,
            patch.object(api, "store_chat_message", return_value=(3, 10)),
        ):
            response = api.execute_rag(RagRequest(question="Bonjour"))

        self.assertEqual(response.action, "answer")
        self.assertEqual(response.answer, "Bonjour !")
        generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
