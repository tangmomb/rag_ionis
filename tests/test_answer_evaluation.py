from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.runtime import Runtime

from interface.backend import api
from interface.backend import orchestration_graph
from interface.backend.answer_evaluation import evaluate_answer_shadow
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


class RagResponseGraphTests(unittest.TestCase):
    def test_studio_runtime_can_fall_back_without_explicit_context(self) -> None:
        runtime = Runtime(context=None)
        self.assertIsInstance(
            api._runtime_context(runtime),
            api.RagResponseContext,
        )
        client = object()
        with patch.object(
            orchestration_graph.services,
            "get_llm_client",
            return_value=client,
        ):
            self.assertIs(orchestration_graph._runtime_client(runtime), client)

    def test_graph_exposes_the_existing_response_pipeline_steps(self) -> None:
        graph = api.RAG_RESPONSE_GRAPH.get_graph()

        self.assertTrue(
            {
                "orchestrate",
                "generate",
                "accept_precomputed",
                "evaluate",
                "correct",
                "retry_retrieval",
                "expand_retrieval",
                "finalize",
                "persist",
            }.issubset(graph.nodes)
        )

    def test_response_state_is_checkpoint_serializable(self) -> None:
        self.assertNotIn("answer_client", api.RagResponseState.__annotations__)
        serializer = JsonPlusSerializer()
        state = {
            "payload": RagRequest(question="Question").model_dump(),
            "answer": "Réponse",
            "sources": [],
            "retrieval": {},
            "answer_trace": {"action": "answer"},
        }

        restored = serializer.loads_typed(serializer.dumps_typed(state))

        self.assertEqual(restored["payload"]["question"], "Question")
        self.assertEqual(restored["answer_trace"]["action"], "answer")

    def test_shadow_evaluation_records_a_diagnostic_without_mutating_the_answer(self) -> None:
        state = {
            "payload": RagRequest(question="Question").model_dump(),
            "answer": "Réponse inchangée",
            "sources": [_source()],
            "retrieval": {
                "route": "rag",
                "answer_model": "mistral-medium-latest",
                "contextual_question": "Question",
            },
            "answer_trace": {"action": "answer"},
        }
        diagnostic = {
            "enabled": True,
            "mode": "shadow",
            "status": "acceptable",
            "reason": "Réponse étayée",
        }
        with (
            patch.object(api, "shadow_evaluation_enabled", return_value=True),
            patch.object(
                api,
                "evaluate_answer_shadow",
                return_value=diagnostic,
            ) as evaluator,
        ):
            sink: dict = {}
            result = api._evaluate_response(
                state,
                Runtime(
                    context=api.RagResponseContext(
                        answer_client=object(),
                        shadow_evaluation_model_override="judge-model",
                        shadow_evaluation_sink=sink,
                    )
                ),
            )

        self.assertEqual(result["shadow_evaluation"], diagnostic)
        self.assertFalse(result["correction_requested"])
        self.assertEqual(sink, diagnostic)
        self.assertEqual(state["answer"], "Réponse inchangée")
        evaluator.assert_called_once()
        self.assertEqual(evaluator.call_args.args[1], "judge-model")

    def test_shadow_evaluation_error_does_not_escape_the_node(self) -> None:
        state = {
            "payload": RagRequest(question="Question").model_dump(),
            "answer": "Réponse inchangée",
            "sources": [_source()],
            "retrieval": {
                "route": "rag",
                "answer_model": "mistral-medium-latest",
            },
            "answer_trace": {"action": "answer"},
        }
        with (
            patch.object(api, "shadow_evaluation_enabled", return_value=True),
            patch.object(
                api,
                "evaluate_answer_shadow",
                side_effect=RuntimeError("échec évaluateur"),
            ),
        ):
            result = api._evaluate_response(
                state,
                Runtime(context=api.RagResponseContext(answer_client=object())),
            )

        self.assertEqual(result["shadow_evaluation"]["status"], "error")
        self.assertEqual(state["answer"], "Réponse inchangée")

    def test_direct_answer_is_not_submitted_to_shadow_evaluation(self) -> None:
        state = {
            "payload": RagRequest(question="Bonjour").model_dump(),
            "answer": "Bonjour !",
            "sources": [],
            "retrieval": {"route": "direct", "answer_model": None},
            "answer_trace": {"action": "answer"},
        }
        with (
            patch.object(api, "shadow_evaluation_enabled", return_value=True),
            patch.object(api, "evaluate_answer_shadow") as evaluator,
        ):
            result = api._evaluate_response(
                state,
                Runtime(context=api.RagResponseContext(answer_client=object())),
            )

        self.assertEqual(result["shadow_evaluation"]["status"], "not_applicable")
        evaluator.assert_not_called()

    def test_generation_abstention_requests_at_most_one_correction(self) -> None:
        state = {
            "payload": RagRequest(question="Question").model_dump(),
            "answer": "Réponse initiale",
            "sources": [_source()],
            "retrieval": {"route": "rag", "answer_model": "answer-model"},
            "answer_trace": {"action": "abstain"},
            "correction_count": 0,
        }
        runtime = Runtime(
            context=api.RagResponseContext(
                answer_client=object(),
                shadow_evaluation_enabled_override=False,
                correction_loop_enabled_override=True,
            )
        )
        first = api._evaluate_response(state, runtime)
        second = api._evaluate_response(
            {**state, "correction_count": 1},
            runtime,
        )

        self.assertTrue(first["correction_requested"])
        self.assertFalse(second["correction_requested"])
        self.assertEqual(
            api._post_evaluation_route({**state, **first}), "expand_retrieval"
        )
        self.assertEqual(api._post_evaluation_route(second), "finalize")

    def test_correction_route_depends_on_evaluation_issue(self) -> None:
        base_state = {"correction_requested": True}

        self.assertEqual(
            api._post_evaluation_route(
                {
                    **base_state,
                    "shadow_evaluation": {"issue": "unsupported_answer"},
                }
            ),
            "correct",
        )
        self.assertEqual(
            api._post_evaluation_route(
                {
                    **base_state,
                    "shadow_evaluation": {"issue": "bad_retrieval"},
                }
            ),
            "retry_retrieval",
        )
        self.assertEqual(
            api._post_evaluation_route(
                {
                    **base_state,
                    "shadow_evaluation": {"issue": "insufficient_sources"},
                }
            ),
            "expand_retrieval",
        )

    def test_insufficient_sources_expands_retrieval_limits(self) -> None:
        state = {
            "payload": RagRequest(
                question="Question",
                topK=20,
                finalK=5,
                plannerPrompt="Prompt planner",
            ).model_dump(),
            "answer": "Réponse initiale",
            "sources": [_source()],
            "retrieval": {"route": "rag", "answer_model": "answer-model"},
            "answer_trace": {
                "action": "abstain",
                "retry_query": "date de publication de la vidéo mentionnée",
            },
            "shadow_evaluation": {
                "verdict": "needs_correction",
                "issue": "insufficient_sources",
                "reason": "Il manque une source.",
            },
        }

        def orchestrate(payload):
            self.assertEqual(payload.topK, api.MAX_TOP_K)
            self.assertEqual(payload.finalK, api.MAX_FINAL_K)
            self.assertIn("plus large", payload.plannerPrompt)
            self.assertIn("date de publication de la vidéo mentionnée", payload.plannerPrompt)
            return "", [_source(), {**_source(), "chunk_id": 8}], {
                "route": "rag",
                "answer_model": "answer-model",
            }

        with patch.object(api, "orchestrate_request", side_effect=orchestrate):
            result = api._expand_retrieval(state)

        self.assertEqual(result["correction_count"], 1)
        self.assertEqual(len(result["sources"]), 2)
        correction = result["retrieval"]["correction"]
        self.assertEqual(correction["strategy"], "expand_retrieval")
        self.assertTrue(correction["succeeded"])
        self.assertEqual(api._post_retrieval_route(result), "generate")

    def test_bad_retrieval_uses_a_precise_new_planner_request(self) -> None:
        state = {
            "payload": RagRequest(
                question="Question",
                topK=10,
                finalK=3,
            ).model_dump(),
            "answer": "Réponse initiale",
            "sources": [_source()],
            "retrieval": {"route": "rag", "answer_model": "answer-model"},
            "answer_trace": {"action": "answer"},
            "shadow_evaluation": {
                "verdict": "needs_correction",
                "issue": "bad_retrieval",
                "reason": "Sources hors sujet.",
            },
        }

        def orchestrate(payload):
            self.assertEqual(payload.topK, 20)
            self.assertEqual(payload.finalK, 8)
            self.assertIn("plus précise", payload.plannerPrompt)
            return "", [_source()], {
                "route": "rag",
                "answer_model": "answer-model",
            }

        with patch.object(api, "orchestrate_request", side_effect=orchestrate):
            result = api._retry_retrieval(state)

        self.assertEqual(
            result["retrieval"]["correction"]["strategy"],
            "retry_retrieval",
        )
        self.assertEqual(api._post_retrieval_route(result), "generate")

    def test_correction_regenerates_with_same_sources_and_records_attempt(self) -> None:
        state = {
            "payload": RagRequest(
                question="Question",
                answerPrompt="Prompt initial",
            ).model_dump(),
            "answer": "Réponse initiale",
            "sources": [_source()],
            "retrieval": {
                "route": "rag",
                "answer_model": "answer-model",
                "contextual_question": "Question autonome",
            },
            "answer_trace": {"action": "answer"},
            "shadow_evaluation": {
                "verdict": "needs_correction",
                "issue": "unsupported_answer",
                "reason": "Une affirmation dépasse les sources.",
                "suggested_correction": "Supprimer cette affirmation.",
            },
        }
        runtime = Runtime(context=api.RagResponseContext(answer_client=object()))

        def generate(_client, question, model, retrieval, sources, trace, prompt):
            self.assertEqual(question, "Question autonome")
            self.assertEqual(model, "answer-model")
            self.assertEqual(sources, state["sources"])
            self.assertIn("Réponse initiale", prompt)
            self.assertIn("unsupported_answer", prompt)
            trace.update({"action": "answer", "source_indexes": [1]})
            return "Réponse corrigée"

        with patch.object(api, "generate_final_answer", side_effect=generate):
            result = api._correct_response(state, runtime)

        self.assertEqual(result["answer"], "Réponse corrigée")
        self.assertEqual(result["correction_count"], 1)
        self.assertTrue(result["retrieval"]["correction"]["succeeded"])
        self.assertEqual(len(result["shadow_evaluation_history"]), 1)

    def test_failed_correction_keeps_the_initial_answer(self) -> None:
        state = {
            "payload": RagRequest(question="Question").model_dump(),
            "answer": "Réponse initiale",
            "sources": [_source()],
            "retrieval": {"route": "rag", "answer_model": "answer-model"},
            "answer_trace": {"action": "answer"},
            "shadow_evaluation": {
                "verdict": "needs_correction",
                "issue": "unsupported_answer",
            },
        }
        runtime = Runtime(context=api.RagResponseContext(answer_client=object()))
        with patch.object(
            api,
            "generate_final_answer",
            side_effect=RuntimeError("échec correction"),
        ):
            result = api._correct_response(state, runtime)

        self.assertEqual(result["answer"], "Réponse initiale")
        self.assertFalse(result["retrieval"]["correction"]["succeeded"])

    def test_finalize_removes_postgresql_nul_characters(self) -> None:
        result = api._finalize_response(
            {
                "answer": "Réponse\x00 corrigée",
                "sources": [],
                "retrieval": {},
                "answer_trace": {"action": "answer"},
            }
        )

        self.assertEqual(result["answer"], "Réponse corrigée")

    def test_graph_runs_one_correction_then_finalizes(self) -> None:
        diagnostics = [
            {
                "verdict": "needs_correction",
                "issue": "unsupported_answer",
                "status": "unsupported_answer",
                "reason": "Réponse non étayée.",
                "suggested_correction": "Retirer l'affirmation.",
            },
            {
                "verdict": "acceptable",
                "issue": "none",
                "status": "acceptable",
                "reason": "Réponse corrigée.",
            },
        ]

        def generate(_client, _question, _model, _retrieval, _sources, trace, _prompt):
            trace.update({"action": "answer", "source_indexes": [1]})
            return "Réponse initiale" if generator.call_count == 1 else "Réponse corrigée"

        with (
            patch.object(
                api,
                "orchestrate_request",
                return_value=(
                    "",
                    [_source()],
                    {
                        "route": "rag",
                        "answer_model": "answer-model",
                        "contextual_question": "Question",
                    },
                ),
            ),
            patch.object(api, "get_llm_client", return_value=object()),
            patch.object(api, "generate_final_answer", side_effect=generate) as generator,
            patch.object(api, "evaluate_answer_shadow", side_effect=diagnostics) as evaluator,
            patch.object(
                api,
                "select_answer_sources",
                side_effect=lambda answer, sources, _indexes: (answer, sources),
            ),
            patch.object(api, "store_chat_message", return_value=(3, 9)),
            patch.object(api, "remember_conversation_turn", return_value={}),
            patch.object(api, "current_trace_id", return_value="trace-1"),
            patch.object(api, "telemetry_status", return_value={"project": "test"}),
        ):
            result = api.RAG_RESPONSE_GRAPH.invoke(
                {
                    "payload": RagRequest(
                        question="Question",
                        conversationId=3,
                    ).model_dump()
                },
                context=api.RagResponseContext(
                    shadow_evaluation_enabled_override=True,
                    correction_loop_enabled_override=True,
                ),
            )

        self.assertEqual(result["answer"], "Réponse corrigée")
        self.assertEqual(result["correction_count"], 1)
        self.assertEqual(result["shadow_evaluation"]["verdict"], "acceptable")
        self.assertEqual(len(result["retrieval"]["shadow_evaluation_history"]), 1)
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(evaluator.call_count, 2)

    def test_shadow_evaluator_uses_a_strict_structured_response(self) -> None:
        client = _Client(
            [
                {
                    "verdict": "acceptable",
                    "issue": "none",
                    "reason": "Réponse étayée",
                    "retrieval_quality": 0.9,
                    "answer_grounded": True,
                    "suggested_correction": None,
                }
            ]
        )

        result = evaluate_answer_shadow(
            client,
            "mistral-medium-latest",
            "Question",
            "Réponse",
            "answer",
            [_source()],
        )

        self.assertEqual(result["status"], "acceptable")
        self.assertEqual(result["verdict"], "acceptable")
        self.assertEqual(result["issue"], "none")
        self.assertTrue(result["answer_grounded"])
        self.assertEqual(len(client.responses.calls), 1)
        self.assertIn("response_schema", client.responses.calls[0])

    def test_correct_clarification_is_acceptable_despite_ambiguity(self) -> None:
        client = _Client(
            [
                {
                    "verdict": "acceptable",
                    "issue": "ambiguous_question",
                    "reason": "La réponse demande la précision nécessaire.",
                    "retrieval_quality": 1.0,
                    "answer_grounded": True,
                    "suggested_correction": None,
                }
            ]
        )

        result = evaluate_answer_shadow(
            client,
            "judge-model",
            "Je cherche la vidéo de Camille",
            "De quelle Camille parlez-vous ?",
            "abstain",
            [],
        )

        self.assertEqual(result["verdict"], "acceptable")
        self.assertEqual(result["issue"], "ambiguous_question")
        self.assertEqual(result["status"], "acceptable")


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
        trace: dict[str, object] = {}

        answer = parse_answer_output(
            ('{"answer":"Peux-tu préciser la vidéo ?","action":"abstain",'
             '"source_indexes":[],"retry_query":"identifier la vidéo concernée"}'),
            trace,
        )

        self.assertEqual(answer, "Peux-tu préciser la vidéo ?")
        self.assertEqual(trace["action"], "abstain")
        self.assertEqual(trace["source_indexes"], [])
        self.assertEqual(trace["retry_query"], "identifier la vidéo concernée")

    def test_invalid_or_missing_action_falls_back_to_abstention(self) -> None:
        trace: dict[str, object] = {}

        answer = parse_answer_output('{"answer":"Réponse non qualifiée"}', trace)

        self.assertEqual(answer, "Réponse non qualifiée")
        self.assertEqual(trace["action"], "abstain")

    def test_no_source_is_still_submitted_to_the_answer_model(self) -> None:
        client = _Client(
            [{"answer": "De quelle vidéo parles-tu ?", "action": "abstain"}]
        )
        trace: dict[str, object] = {}

        answer = generate_answer(
            client,
            "Quelle est sa date de publication ?",
            "mistral-medium-latest",
            [],
            trace,
        )

        self.assertEqual(answer, "De quelle vidéo parles-tu ?")
        self.assertEqual(trace["action"], "abstain")
        self.assertEqual(len(client.responses.calls), 1)
        self.assertIn(
            "Aucune source exploitable",
            client.responses.calls[0]["input"][1]["content"],
        )

    def test_ambiguous_person_candidates_are_given_to_the_answer_model(self) -> None:
        client = _Client(
            [{"answer": "Parles-tu d'Alice Martin ou d'Alice Durand ?", "action": "abstain"}]
        )
        trace: dict[str, object] = {}

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
        self.assertEqual(trace["action"], "abstain")
        prompt = client.responses.calls[0]["input"][1]["content"]
        self.assertIn("Alice Martin", prompt)
        self.assertIn("Alice Durand", prompt)

    def test_execute_rag_uses_the_action_from_the_generation_call(self) -> None:
        retrieval = {
            "route": "rag",
            "retrieval_mode": "rag+sql",
            "contextual_question": "Quand la vidéo a-t-elle été publiée ?",
            "answer_model": "mistral-medium-latest",
        }

        def generate(*args, **kwargs):
            trace = args[5]
            trace["action"] = "answer"
            trace["source_indexes"] = [1]
            return "Le 12 avril 2022."

        with (
            patch.object(api, "create_conversation", return_value=3),
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

    def test_execute_rag_hides_sources_for_abstention(self) -> None:
        retrieval = {
            "route": "rag",
            "retrieval_mode": "rag+sql",
            "contextual_question": "Je cherche la video de Sophie",
            "answer_model": "mistral-medium-latest",
        }
        source = _source()
        source["thumbnail_medium_url"] = "https://example.test/thumbnail.jpg"

        def generate(*args, **kwargs):
            trace = args[5]
            trace["action"] = "abstain"
            trace["source_indexes"] = [1]
            return "Parles-tu de cette Sophie ?"

        with (
            patch.object(api, "create_conversation", return_value=3),
            patch.object(api, "orchestrate_request", return_value=("", [source], retrieval)),
            patch.object(api, "get_llm_client", return_value=object()),
            patch.object(api, "generate_final_answer", side_effect=generate),
            patch.object(api, "store_chat_message", return_value=(3, 11)),
        ):
            response = api.execute_rag(RagRequest(question="Video de Sophie ?"))

        self.assertEqual(response.action, "abstain")
        self.assertEqual(response.answer, "Parles-tu de cette Sophie ?")
        self.assertEqual(response.sources, [])
        self.assertEqual(response.retrieval["answer_source_indexes"], [])

    def test_precomputed_direct_answer_keeps_answer_action(self) -> None:
        retrieval = {
            "route": "direct",
            "retrieval_mode": "direct",
            "contextual_question": "Bonjour",
            "answer_model": None,
        }
        with (
            patch.object(api, "create_conversation", return_value=3),
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
