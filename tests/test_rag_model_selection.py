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
            "fetch_conversation_memory",
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
        memory_plan = PlannerPlan(
            route="memory",
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
                return_value=(memory_plan, "prompt", "raw", True),
            ) as run_planner,
            patch.object(
                orchestration,
                "fetch_conversation_memory",
                return_value=([], {"conversation_id": None}),
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
        self.assertIn("exactement deux clés : answer et action", rendered)
        self.assertNotIn("{route_instructions}", rendered)

    def test_default_answer_prompt_displays_real_instructions(self) -> None:
        self.assertIn("exactement deux clés : answer et action", DEFAULT_ANSWER_PROMPT_TEMPLATE)
        self.assertIn("Markdown", DEFAULT_ANSWER_PROMPT_TEMPLATE)
        self.assertIn("courte formule de politesse", DEFAULT_ANSWER_PROMPT_TEMPLATE)
        self.assertIn("{route_instructions}", DEFAULT_ANSWER_PROMPT_TEMPLATE)

    def test_custom_answer_prompt_cannot_drop_the_action_contract(self) -> None:
        rendered = render_answer_system_prompt(
            "Réponds très brièvement.",
            route_instructions="Instructions RAG",
        )

        self.assertIn("Réponds très brièvement.", rendered)
        self.assertIn("exactement deux clés : answer et action", rendered)


if __name__ == "__main__":
    unittest.main()
