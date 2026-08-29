from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from interface.backend import orchestration, planner
from interface.backend.generation import (
    DEFAULT_ANSWER_PROMPT_TEMPLATE,
    render_answer_system_prompt,
)
from interface.backend.schemas import PlannerPlan, RagRequest
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
        self.assertEqual(len(client.responses.calls[0]["input"]), 1)
        self.assertEqual(client.responses.calls[0]["input"][0]["role"], "user")
        self.assertIn(
            "Reformule le dernier message utilisateur",
            client.responses.calls[0]["input"][0]["content"],
        )
        self.assertIn(
            "Message actuel : Question",
            client.responses.calls[0]["input"][0]["content"],
        )
        self.assertIn('"model": "gpt-5.6-luna"', trace["prompt"])

    def test_orchestration_passes_independent_models_to_each_step(self) -> None:
        client = object()
        payload = RagRequest(
            question="Question",
            reformulationModel="2 terra",
            plannerModel="3 luna",
            answerModel="2 terra",
            reformulationPrompt="Prompt reformulation personnalise",
            plannerPrompt="Prompt planner personnalise",
        )
        multi_source_plan = PlannerPlan(
            route="multi_source",
            query_text="Question reformulee",
        )

        with (
            patch.object(orchestration, "get_llm_client", return_value=client),
            patch.object(
                orchestration,
                "reformulate_question",
                return_value=("Question reformulee", {"applied": True}),
            ) as reformulate,
            patch.object(
                orchestration,
                "run_planner",
                return_value=(multi_source_plan, "prompt", "raw", True),
            ) as run_planner,
            patch.object(
                orchestration,
                "retrieve_chunks",
                return_value=([], {}),
            ),
        ):
            _answer, _sources, retrieval = orchestration.orchestrate_request(payload)

        reformulate.assert_called_once_with(
            "Question",
            None,
            client,
            "gpt-5.6-terra",
            "Prompt reformulation personnalise",
        )
        run_planner.assert_called_once_with(
            "Question reformulee",
            client,
            "gpt-5.6-luna",
            "Prompt planner personnalise",
        )
        self.assertEqual(retrieval["reformulation_model"], "gpt-5.6-terra")
        self.assertEqual(retrieval["planner_model"], "gpt-5.6-luna")
        self.assertEqual(retrieval["answer_model"], "gpt-5.6-terra")

    def test_orchestration_uses_light_then_final_reformulation_with_memory(self) -> None:
        client = object()
        payload = RagRequest(question="Elle a plus de vues qu'eux ?", conversationId=46)
        plan = PlannerPlan(route="rag", query_text="Question autonome")
        memory = {
            "available": True,
            "active_topic": {"objective": "Job de Lou-Ann", "entities": ["Lou-Ann"]},
            "immediate_history": [{"role": "user", "text": "Je cherche le job de Lou-Ann."}],
            "episodes": [{"content": "Question : Qui a le plus de vues ?\nRéponse : Déborah contre Simon."}],
            "retrieval": {"selected_count": 1},
        }
        with (
            patch.object(orchestration, "get_llm_client", return_value=client),
            patch.object(orchestration, "load_reformulation_memory", side_effect=[memory, memory]) as load_memory,
            patch.object(orchestration, "assign_topic_id", return_value={"topic_id": 47, "decision": "new_topic"}) as assign_topic,
            patch.object(
                orchestration,
                "reformulate_question",
                side_effect=[("Elle a plus de vues qu'eux ?", {"phase": "light"}), ("Lou-Ann a-t-elle plus de vues que Déborah et Simon ?", {"phase": "final"})],
            ) as reformulate,
            patch.object(orchestration, "run_planner", return_value=(plan, "prompt", "raw", True)),
            patch.object(orchestration, "retrieve_chunks", return_value=([], {})),
        ):
            _answer, _sources, retrieval = orchestration.orchestrate_request(payload)

        self.assertEqual(reformulate.call_count, 2)
        assign_topic.assert_called_once_with(46, False)
        self.assertEqual(load_memory.call_args_list[0].kwargs["include_episodes"], False)
        self.assertEqual(load_memory.call_args_list[1].args[1], "Elle a plus de vues qu'eux ?")
        self.assertNotIn("memory_context", reformulate.call_args_list[0].kwargs)
        self.assertEqual(reformulate.call_args_list[0].kwargs["phase"], "light")
        self.assertEqual(reformulate.call_args_list[1].kwargs["phase"], "final")
        self.assertEqual(retrieval["question_reformulation"]["strategy"], "light_rewrite+topic_match+final_rewrite")

    def test_empty_structured_sql_does_not_fallback_to_rag(self) -> None:
        client = object()
        payload = RagRequest(question="Y a-t-il des commentaires ?")
        plan = PlannerPlan(
            route="rag",
            sql_sub_intent="analytics",
            query_text="Y a-t-il des commentaires ?",
            sql_main_source=True,
        )
        empty_sql_trace = {"sql": "SELECT ...", "params": [], "result_count": 0}
        with (
            patch.object(orchestration, "get_llm_client", return_value=client),
            patch.object(orchestration, "reformulate_question", return_value=(payload.question, {})),
            patch.object(orchestration, "run_planner", return_value=(plan, "prompt", "raw", True)),
            patch.object(orchestration, "resolve_person_filters", return_value=([], {"ambiguous": False, "matched_in_transcripts": []})),
            patch.object(orchestration, "run_analytics_text_to_sql", return_value=([], empty_sql_trace)),
            patch.object(orchestration, "retrieve_chunks") as retrieve_chunks,
        ):
            _answer, sources, retrieval = orchestration.orchestrate_request(payload)

        self.assertEqual(sources, [])
        self.assertEqual(retrieval["retrieval_mode"], "rag+structured_sql")
        self.assertEqual(retrieval["direct_lookup"], empty_sql_trace)
        retrieve_chunks.assert_not_called()

    def test_follow_up_skips_the_final_rewrite(self) -> None:
        client = object()
        payload = RagRequest(question="Et elle ?", conversationId=46)
        plan = PlannerPlan(route="direct", query_text="Et elle ?")
        memory = {
            "available": True,
            "active_topic": "Déborah Rolland : interview et questions posées.",
            "immediate_history": [{"role": "user", "text": "Quelles questions à Déborah ?"}],
            "episodes": [],
        }
        with (
            patch.object(orchestration, "get_llm_client", return_value=client),
            patch.object(orchestration, "load_reformulation_memory", return_value=memory) as load_memory,
            patch.object(orchestration, "assign_topic_id", return_value={"topic_id": 48, "decision": "current_topic"}) as assign_topic,
            patch.object(
                orchestration,
                "reformulate_question",
                return_value=("Et Déborah ?", {"follow_up": True}),
            ) as reformulate,
            patch.object(orchestration, "run_planner", return_value=(plan, "prompt", "raw", True)),
        ):
            orchestration.orchestrate_request(payload)

        load_memory.assert_called_once()
        assign_topic.assert_called_once_with(46, True)
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
        self.assertIn("Choisis l'action answer, clarify ou abstain", rendered)
        self.assertNotIn("{route_instructions}", rendered)

    def test_default_answer_prompt_displays_real_instructions(self) -> None:
        self.assertIn(
            "Choisis l'action answer, clarify ou abstain",
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
        self.assertIn("Choisis l'action answer, clarify ou abstain", rendered)


if __name__ == "__main__":
    unittest.main()
