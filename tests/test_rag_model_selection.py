from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from interface.backend import orchestration, orchestration_graph, planner
from interface.backend.generation import (
    DEFAULT_ANSWER_PROMPT_TEMPLATE,
    render_answer_system_prompt,
)
from interface.backend.schemas import ExecutionPlan, PlannerPlan, RagRequest
from interface.backend.utilities import normalize_model_name


class _CapturingResponses:
    def __init__(self, output: dict) -> None:
        self.output = output
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.output))


class _Client:
    def __init__(self, output: dict) -> None:
        self.responses = _CapturingResponses(output)


class RagModelSelectionTests(unittest.TestCase):
    def test_planner_llm_can_only_request_analytics_sql(self) -> None:
        properties = planner.PLANNER_RESPONSE_SCHEMA["properties"]
        self.assertEqual(properties["analytics"]["type"], "boolean")
        self.assertNotIn("sql_sub_intent", properties)

    def test_planner_boolean_preserves_entities_and_execution_intent(self) -> None:
        for analytics, scope, expected in ((False, None, None), (True, "specific", "analytics")):
            with self.subTest(analytics=analytics):
                client = _Client({
                    "route": "search", "analytics": analytics,
                    "analytics_scope": scope, "query_text": "Fadila",
                    "query_text_bm25": "Fadila", "persons": ["Fadila"],
                })
                plan, _, _, verified = planner.run_planner("Fadila", client)
                self.assertTrue(verified)
                self.assertEqual(plan.sql_sub_intent, expected)
                self.assertEqual(plan.persons, ["Fadila"])
                self.assertEqual(plan.query_text_bm25, "Fadila")

    def test_planner_rejects_non_boolean_analytics(self) -> None:
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value):
                client = _Client({"route": "search", "analytics": value, "query_text": "Fadila"})
                plan, _, _, verified = planner.run_planner("Fadila", client)
                self.assertFalse(verified)
                self.assertIsNone(plan.sql_sub_intent)

    def test_planner_boolean_takes_precedence_over_legacy_intent(self) -> None:
        normalized = planner.normalize_planner_output({
            "route": "search", "analytics": False,
            "sql_sub_intent": "analytics", "analytics_scope": "global",
        })
        self.assertIsNone(normalized["sql_sub_intent"])
        self.assertIsNone(normalized["analytics_scope"])

    def test_planner_boolean_requires_analytics_scope(self) -> None:
        client = _Client({"route": "search", "analytics": True, "query_text": "Vues"})
        plan, _, _, verified = planner.run_planner("Vues", client)
        self.assertFalse(verified)
        self.assertEqual(plan.output_rejection_reason, "analytics_scope_missing")

    def test_entity_filters_force_specific_analytics_scope(self) -> None:
        normalized = planner.normalize_planner_output({
            "route": "search",
            "analytics": True,
            "analytics_scope": "global",
            "analytics_metric": "views",
            "analytics_order": "desc",
            "analytics_rank_start": 1,
            "analytics_rank_end": 2,
            "query_text": "Les deux vidéos Novares les plus vues",
            "title_hints": ["Témoignage Tuteur : Loïc Maréchal, Directeur HSE, Novares"],
            "persons": ["Loïc Maréchal"],
            "companies": ["Novares"],
        })

        self.assertEqual(normalized["analytics_scope"], "specific")
        self.assertIsNone(normalized["analytics_metric"])
        self.assertIsNone(normalized["analytics_order"])
        self.assertIsNone(normalized["analytics_rank_start"])
        self.assertIsNone(normalized["analytics_rank_end"])

        plan = PlannerPlan(
            query_text="Les deux vidéos Novares les plus vues",
            sql_sub_intent="analytics",
            analytics_scope="global",
            title_hints=["Témoignage Tuteur : Loïc Maréchal, Directeur HSE, Novares"],
        )
        self.assertEqual(plan.analytics_scope, "specific")

    def test_orchestration_graph_exposes_existing_planning_and_retrieval_routes(self) -> None:
        graph = orchestration_graph.RAG_ORCHESTRATION_GRAPH.get_graph()

        self.assertTrue(
            {
                "initialize",
                "reformulate",
                "plan",
                "resolve_entities",
                "build_execution_plan",
                "person_clarification",
                "direct",
                "sql_search",
                "vector_search",
            }.issubset(graph.nodes)
        )

    def test_orchestration_graph_uses_the_final_execution_route(self) -> None:
        def execution_plan(route: str) -> ExecutionPlan:
            return ExecutionPlan(
                route=route,
                raw_question="Question",
                query_text="Question",
                query_text_bm25="Question",
            )

        self.assertEqual(
            orchestration_graph.select_route(
                {
                    "person_resolution": {"ambiguous": True},
                    "execution_plan": execution_plan("person_clarification").model_dump(),
                }
            ),
            "person_clarification",
        )
        self.assertEqual(
            orchestration_graph.select_route(
                {
                    "person_resolution": {"ambiguous": False},
                    "execution_plan": execution_plan("direct").model_dump(),
                }
            ),
            "direct",
        )
        self.assertEqual(
            orchestration_graph.select_route(
                {
                    "person_resolution": {"ambiguous": False},
                    "execution_plan": execution_plan("sql_search").model_dump(),
                }
            ),
            "sql_search",
        )
        self.assertEqual(
            orchestration_graph.select_route(
                {
                    "person_resolution": {"ambiguous": False},
                    "execution_plan": execution_plan("vector_search").model_dump(),
                }
            ),
            "vector_search",
        )

    def test_orchestration_state_is_checkpoint_serializable(self) -> None:
        self.assertNotIn("client", orchestration_graph.RagOrchestrationState.__annotations__)
        serializer = JsonPlusSerializer()
        state = {
            "payload": RagRequest(question="Question").model_dump(),
            "planner_plan": PlannerPlan(query_text="Question").model_dump(),
            "execution_plan": ExecutionPlan(
                raw_question="Question",
                query_text="Question",
                query_text_bm25="Question",
            ).model_dump(),
            "sources": [],
            "retrieval": {},
        }

        restored = serializer.loads_typed(serializer.dumps_typed(state))

        self.assertEqual(restored["payload"]["question"], "Question")
        self.assertEqual(restored["planner_plan"]["query_text"], "Question")

    def test_sol_terra_luna_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_model_name("1 sol", "fallback"), "gpt-5.6-sol")
        self.assertEqual(normalize_model_name("2 terra", "fallback"), "gpt-5.6-terra")
        self.assertEqual(normalize_model_name("3 luna", "fallback"), "gpt-5.6-luna")

    def test_planner_uses_requested_model(self) -> None:
        client = _Client(
            {
                "route": "rag",
                "query_text": "Question",
                "persons": [],
                "companies": [],
            }
        )

        planner.run_planner("Question", client, "gpt-5.6-terra")

        self.assertEqual(
            client.responses.calls[0]["model"],
            "gpt-5.6-terra",
        )
        self.assertEqual(
            client.responses.calls[0]["response_schema"],
            planner.PLANNER_RESPONSE_SCHEMA,
        )

    def test_reformulation_uses_requested_model(self) -> None:
        client = _Client(
            {
                "follow_up": False,
                "reformulated_question": "Question reformulee",
            }
        )
        with patch.object(
            planner,
            "fetch_conversation_history",
            return_value=([], {"conversation_id": None}),
        ):
            reformulated, trace = planner.reformulate_question(
                "Question",
                None,
                client,
                "gpt-5.6-luna",
            )

        self.assertEqual(reformulated, "Question reformulee")
        self.assertEqual(client.responses.calls[0]["model"], "gpt-5.6-luna")
        self.assertEqual(
            client.responses.calls[0]["response_schema"],
            planner.REFORMULATION_RESPONSE_SCHEMA,
        )
        self.assertEqual(len(client.responses.calls[0]["input"]), 2)
        self.assertEqual(client.responses.calls[0]["input"][0]["role"], "system")
        self.assertIn(
            "Reformule le dernier message utilisateur",
            client.responses.calls[0]["input"][0]["content"],
        )
        self.assertEqual(client.responses.calls[0]["input"][1]["role"], "user")
        self.assertIn(
            "Message actuel : Question",
            client.responses.calls[0]["input"][1]["content"],
        )
        self.assertIn('"model": "gpt-5.6-luna"', trace["prompt"])

    def test_final_reformulation_only_requests_the_rewritten_question(self) -> None:
        client = _Client({"reformulated_question": "Question contextualisee"})
        with patch.object(
            planner,
            "fetch_conversation_history",
            return_value=([], {"conversation_id": None}),
        ):
            reformulated, trace = planner.reformulate_question(
                "Question", None, client, phase="final"
            )

        self.assertEqual(reformulated, "Question contextualisee")
        self.assertNotIn("follow_up", trace)
        self.assertEqual(
            client.responses.calls[0]["response_schema"],
            planner.FINAL_REFORMULATION_RESPONSE_SCHEMA,
        )
        self.assertNotIn(
            "follow_up", client.responses.calls[0]["input"][0]["content"]
        )

    def test_orchestration_uses_mistral_medium_for_every_step(self) -> None:
        client = object()
        payload = RagRequest(
            question="Question",
            reformulationModel="2 terra",
            plannerModel="3 luna",
            answerModel="2 terra",
            reformulationPrompt="Prompt reformulation personnalise",
            plannerPrompt="Prompt planner personnalise",
        )
        rag_plan = PlannerPlan(
            route="search",
            query_text="Question reformulee",
        )

        with (
            patch.object(orchestration, "get_llm_client", return_value=client),
            patch.object(
                orchestration,
                "load_conversation_memory",
                return_value={
                    "available": True,
                    "memory": {"current_topic": {"topic": "", "messages": []}, "previous_topics": []},
                },
            ),
            patch.object(
                orchestration,
                "reformulate_question",
                return_value=("Question reformulee", {"applied": True}),
            ) as reformulate,
            patch.object(
                orchestration,
                "run_planner",
                return_value=(rag_plan, "prompt", "raw", True),
            ) as run_planner,
            patch.object(
                orchestration,
                "retrieve_chunks",
                return_value=([], {"answer_model": "mistral-medium-latest"}),
            ),
        ):
            _answer, _sources, retrieval = orchestration.orchestrate_request(payload)

        reformulate.assert_called_once()
        self.assertEqual(reformulate.call_args.args[:5], (
            "Question", None, client, "mistral-medium-latest", "Prompt reformulation personnalise"
        ))
        self.assertEqual(
            reformulate.call_args.kwargs["memory_context"]["conversation_memory"]["previous_topics"],
            [],
        )
        run_planner.assert_called_once_with(
            "Question reformulee",
            client,
            "mistral-medium-latest",
            "Prompt planner personnalise",
        )
        self.assertEqual(retrieval["reformulation_model"], "mistral-medium-latest")
        self.assertEqual(retrieval["planner_model"], "mistral-medium-latest")
        self.assertEqual(retrieval["answer_model"], "mistral-medium-latest")

    def test_orchestration_uses_one_reformulation_with_conversation_json(self) -> None:
        client = object()
        payload = RagRequest(question="Elle a plus de vues qu'eux ?", conversationId=46)
        plan = PlannerPlan(route="search", query_text="Question autonome")
        memory = {
            "available": True,
            "memory": {
                "current_topic": {"topic": "Job de Lou-Ann", "messages": [{"role": "user", "content": "Je cherche le job de Lou-Ann."}]},
                "previous_topics": [{"topic": "Vues", "summary": "Déborah contre Simon."}],
            },
        }
        with (
            patch.object(orchestration, "get_llm_client", return_value=client),
            patch.object(orchestration, "load_conversation_memory", return_value=memory) as load_memory,
            patch.object(
                orchestration,
                "reformulate_question",
                return_value=("Lou-Ann a-t-elle plus de vues que Déborah et Simon ?", {"follow_up": False, "topic": "Vues de Lou-Ann"}),
            ) as reformulate,
            patch.object(orchestration, "run_planner", return_value=(plan, "prompt", "raw", True)),
            patch.object(orchestration, "retrieve_chunks", return_value=([], {})),
        ):
            _answer, _sources, retrieval = orchestration.orchestrate_request(payload)

        reformulate.assert_called_once()
        load_memory.assert_called_once_with(46)
        self.assertEqual(
            reformulate.call_args.kwargs["memory_context"]["conversation_memory"], memory["memory"]
        )
        self.assertEqual(retrieval["question_reformulation"]["strategy"], "conversation_json_single_rewrite")

    def test_empty_sql_does_not_fallback_to_rag(self) -> None:
        client = object()
        payload = RagRequest(question="Y a-t-il des commentaires ?")
        plan = PlannerPlan(
            route="search",
            sql_sub_intent="analytics",
            query_text="Y a-t-il des commentaires ?",
        )
        empty_sql_trace = {
            "mode": "analytics",
            "strategy": "deterministic_entity_stats",
            "result_count": 0,
        }
        with (
            patch.object(orchestration, "get_llm_client", return_value=client),
            patch.object(orchestration, "reformulate_question", return_value=(payload.question, {})),
            patch.object(orchestration, "run_planner", return_value=(plan, "prompt", "raw", True)),
            patch.object(
                orchestration,
                "resolve_person_filters",
                return_value=(
                    [],
                    {"ambiguous": False, "suggestion_transcripts": []},
                ),
            ),
            patch.object(orchestration, "run_deterministic_analytics", return_value=([], empty_sql_trace)),
            patch.object(orchestration, "retrieve_chunks") as retrieve_chunks,
        ):
            _answer, sources, retrieval = orchestration.orchestrate_request(payload)

        self.assertEqual(sources, [])
        self.assertEqual(retrieval["retrieval_mode"], "search+sql")
        self.assertEqual(retrieval["direct_lookup"], empty_sql_trace)
        retrieve_chunks.assert_not_called()

    def test_follow_up_uses_the_same_single_rewrite(self) -> None:
        client = object()
        payload = RagRequest(question="Et elle ?", conversationId=46)
        plan = PlannerPlan(route="direct", query_text="Et elle ?")
        memory = {"available": True, "memory": {"current_topic": {"topic": "Déborah", "messages": []}, "previous_topics": []}}
        with (
            patch.object(orchestration, "get_llm_client", return_value=client),
            patch.object(orchestration, "load_conversation_memory", return_value=memory) as load_memory,
            patch.object(
                orchestration,
                "reformulate_question",
                return_value=("Et Déborah ?", {"follow_up": True}),
            ) as reformulate,
            patch.object(orchestration, "run_planner", return_value=(plan, "prompt", "raw", True)),
        ):
            orchestration.orchestrate_request(payload)

        load_memory.assert_called_once_with(46)
        reformulate.assert_called_once()

    def test_prompt_builders_accept_custom_system_prompts(self) -> None:
        planner_system, _ = planner.build_planner_prompt(
            "Question",
            "Planner personnalise",
        )
        reformulation_system, _ = planner.build_question_reformulation_prompt(
            "Question",
            [],
            "Reformulation personnalisee",
        )

        self.assertEqual(planner_system, "Planner personnalise")
        self.assertEqual(reformulation_system, "Reformulation personnalisee")

    def test_answer_prompt_template_replaces_dynamic_placeholders(self) -> None:
        rendered = render_answer_system_prompt(
            "Debut\n{route_instructions}\n{answer_action_instruction}\nFin",
            route_instructions="Instructions RAG",
        )

        self.assertIn("Instructions RAG", rendered)
        self.assertIn("Choisis l'action answer ou abstain", rendered)
        self.assertNotIn("{route_instructions}", rendered)

    def test_default_answer_prompt_displays_real_instructions(self) -> None:
        self.assertIn(
            "Choisis l'action answer ou abstain",
            DEFAULT_ANSWER_PROMPT_TEMPLATE,
        )
        self.assertIn("Markdown", DEFAULT_ANSWER_PROMPT_TEMPLATE)
        self.assertIn("courte formule de politesse", DEFAULT_ANSWER_PROMPT_TEMPLATE)
        self.assertIn("{route_instructions}", DEFAULT_ANSWER_PROMPT_TEMPLATE)

    def test_custom_answer_prompt_cannot_drop_the_action_contract(self) -> None:
        rendered = render_answer_system_prompt(
            "Réponds très brièvement.",
            route_instructions="Instructions RAG",
        )

        self.assertIn("Réponds très brièvement.", rendered)
        self.assertIn("Choisis l'action answer ou abstain", rendered)


if __name__ == "__main__":
    unittest.main()
