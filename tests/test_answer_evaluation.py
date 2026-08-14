from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from interface.backend import api
from interface.backend.answer_judge import (
    ANALYTICS_VIDEO_METADATA_CORRECTION,
    ANSWER_JUDGE_RESPONSE_SCHEMA,
    judge_final_answer,
)
from interface.backend.generation import (
    generate_answer,
    generate_person_clarification_answer,
    parse_answer_output,
)
from interface.backend.schemas import ExecutionPlan, RagRequest


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
    def test_answer_judge_is_limited_to_risky_answers(self) -> None:
        self.assertFalse(api.should_run_answer_judge("direct", "answer", {}))
        self.assertFalse(api.should_run_answer_judge("rag", "answer", {}))
        self.assertTrue(api.should_run_answer_judge("rag", "clarify", {}))
        self.assertTrue(api.should_run_answer_judge("rag", "abstain", {}))
        self.assertTrue(
            api.should_run_answer_judge(
                "rag",
                "answer",
                {"sql_sub_intent": "analytics"},
            )
        )
        self.assertFalse(api.should_run_answer_judge("multi_source", "answer", {}))
        self.assertTrue(
            api.should_run_answer_judge(
                "rag",
                "answer",
                {"direct_lookup": {"result_count": 0}},
            )
        )

    def test_answer_judge_requests_sql_retry_with_strict_schema(self) -> None:
        client = _Client(
            [
                {
                    "valid": False,
                    "retry_stage": "sql",
                    "reason": "La requête utilise une intersection.",
                    "correction": "Utiliser l'union des vidéos des deux personnes.",
                }
            ]
        )

        verdict = judge_final_answer(
            client,
            "mistral-medium-latest",
            "Laquelle a le plus de vues ?",
            {
                "route": "multi_source",
                "sql_sub_intent": "analytics",
                "execution_plan": {"persons": ["Loucif", "Sophie Ollivier"]},
            },
            [],
            "Précise les vidéos.",
            "clarify",
        )

        self.assertFalse(verdict["valid"])
        self.assertEqual(verdict["retry_stage"], "sql")
        self.assertEqual(
            client.responses.calls[0]["response_schema"],
            ANSWER_JUDGE_RESPONSE_SCHEMA,
        )

    def test_answer_judge_retries_analytics_without_video_metadata(self) -> None:
        client = _Client([])
        source = _source()
        source["video_title"] = "Résultat analytique"
        source["video_url"] = ""

        verdict = judge_final_answer(
            client,
            "mistral-medium-latest",
            "Laquelle a le plus de vues ?",
            {"sql_sub_intent": "analytics"},
            [source],
            "Loucif a le plus de vues. [S1]",
            "answer",
        )

        self.assertFalse(verdict["valid"])
        self.assertEqual(verdict["retry_stage"], "sql")
        self.assertEqual(verdict["correction"], ANALYTICS_VIDEO_METADATA_CORRECTION)
        self.assertEqual(client.responses.calls, [])

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

    def test_execute_rag_keeps_cited_sources_for_clarification(self) -> None:
        retrieval = {
            "route": "rag",
            "retrieval_mode": "rag+structured_sql",
            "contextual_question": "Je cherche la video de Sophie",
            "answer_model": "mistral-medium-latest",
        }
        source = _source()
        source["thumbnail_medium_url"] = "https://example.test/thumbnail.jpg"

        def generate(*args, **kwargs):
            trace = args[5]
            trace["action"] = "clarify"
            return "Parles-tu de cette Sophie ? [S1]"

        with (
            patch.object(api, "orchestrate_request", return_value=("", [source], retrieval)),
            patch.object(api, "get_llm_client", return_value=object()),
            patch.object(api, "generate_final_answer", side_effect=generate),
            patch.object(api, "store_chat_message", return_value=(3, 11)),
        ):
            response = api.execute_rag(RagRequest(question="Video de Sophie ?"))

        self.assertEqual(response.action, "clarify")
        self.assertEqual(response.answer, "Parles-tu de cette Sophie ?")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(
            response.sources[0].thumbnail_medium_url,
            "https://example.test/thumbnail.jpg",
        )
        self.assertEqual(response.retrieval["answer_source_indexes"], [1])

    def test_execute_rag_retries_sql_once_when_judge_detects_intersection(self) -> None:
        execution_plan = ExecutionPlan(
            route="multi_source",
            sql_sub_intent="analytics",
            raw_question="Loucif et Sophie Ollivier ?",
            query_text="Quelle vidéo entre Loucif et Sophie Ollivier a le plus de vues ?",
            query_text_bm25="Loucif Sophie vues",
            persons=["Loucif", "Sophie Ollivier"],
            sql_main_source=True,
            top_k=None,
            final_k=None,
        )
        retrieval = {
            "route": "multi_source",
            "retrieval_mode": "multi_source",
            "contextual_question": execution_plan.query_text,
            "answer_model": "mistral-medium-latest",
            "planner_model": "mistral-medium-latest",
            "sql_sub_intent": "analytics",
            "execution_plan": execution_plan.model_dump(),
            "resolved_persons": ["Loucif Ouyahia", "Sophie Ollivier"],
            "resolved_companies": [],
            "multi_source_actions": [],
        }
        corrected_source = _source()
        corrected_source["chunk_id"] = 40
        corrected_source["video_title"] = "Sophie Ollivier"
        corrected_source["text"] = "view_count: 127"
        generated_answers = iter(
            [
                ("Précise les vidéos concernées.", "clarify"),
                ("Sophie Ollivier a le plus de vues. [S1]", "answer"),
            ]
        )

        def generate(*args, **kwargs):
            answer, action = next(generated_answers)
            args[5]["action"] = action
            if action == "answer":
                self.assertIn("union", kwargs["judge_feedback"])
            return answer

        judge_verdicts = [
            {
                "status": "completed",
                "valid": False,
                "retry_stage": "sql",
                "reason": "Le SQL cherche une vidéo commune.",
                "correction": "Utiliser l'union des vidéos des deux personnes.",
            },
            {
                "status": "completed",
                "valid": True,
                "retry_stage": "none",
                "reason": "",
                "correction": "",
            },
        ]

        with (
            patch.object(api, "orchestrate_request", return_value=("", [], retrieval)),
            patch.object(api, "get_llm_client", return_value=object()),
            patch.object(api, "generate_final_answer", side_effect=generate) as generator,
            patch.object(api, "judge_final_answer", side_effect=judge_verdicts) as judge,
            patch.object(
                api,
                "run_analytics_text_to_sql",
                return_value=(
                    [corrected_source],
                    {
                        "status": "executed",
                        "sql": "SELECT ... WHERE speaker_a OR speaker_b",
                        "params": ["Loucif", "Sophie Ollivier"],
                        "result_count": 1,
                    },
                ),
            ) as sql_retry,
            patch.object(api, "store_chat_message", return_value=(213, 277)),
        ):
            response = api.execute_rag(
                RagRequest(question="Loucif et Sophie Ollivier ?", conversationId=213)
            )

        self.assertEqual(response.action, "answer")
        self.assertEqual(response.answer, "Sophie Ollivier a le plus de vues.")
        self.assertEqual(response.sources[0].chunk_id, 40)
        self.assertTrue(response.retrieval["answer_judge"]["retry_performed"])
        self.assertTrue(response.retrieval["answer_judge"]["final"]["valid"])
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(judge.call_count, 2)
        sql_retry.assert_called_once()

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
