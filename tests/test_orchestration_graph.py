from __future__ import annotations

import unittest
from unittest.mock import patch

from interface.backend.orchestration_graph import (
    _execution_plan_trace,
    _exclude_ionis_stm_companies,
    build_execution_plan,
    _resolved_plan_companies,
    _resolved_plan_persons,
    _resolved_plan_title_hint,
    _resolved_plan_title_hints,
    _resolved_transcript_persons,
    select_after_plan,
    _sql_parent_output,
)
from interface.backend.schemas import ExecutionPlan, PlannerPlan


class ExecutionPlanTraceTests(unittest.TestCase):

    def test_allow_list_person_from_planner_short_circuits_the_pipeline(self) -> None:
        state = {
            "payload": {"question": "Qui est Laura Tyan ?"},
            "planner_plan": PlannerPlan(
                route="search", query_text="Qui est Laura Tyan ?", persons=["Laura Tyan"]
            ).model_dump()
        }
        with patch(
            "interface.backend.orchestration_graph.matching_person_scoped_annex_persons",
            return_value=["Laura Tyan"],
        ):
            route = select_after_plan(state)

        self.assertEqual(route, "annex_direct")

    def test_non_allow_list_person_keeps_entity_resolution_pipeline(self) -> None:
        state = {
            "payload": {"question": "Qui est quelqu'un ?"},
            "planner_plan": PlannerPlan(
                route="search", query_text="Qui est quelqu'un ?", persons=["Quelqu'un"]
            ).model_dump()
        }
        with patch(
            "interface.backend.orchestration_graph.matching_person_scoped_annex_persons",
            return_value=[],
        ), patch(
            "interface.backend.orchestration_graph.load_question_scoped_knowledge",
            return_value=(None, {"enabled": False}),
        ):
            route = select_after_plan(state)

        self.assertEqual(route, "resolve_entities")

    def test_exact_question_trigger_short_circuits_the_pipeline(self) -> None:
        state = {
            "payload": {"question": "Question annexe précise"},
            "planner_plan": PlannerPlan(
                route="search", query_text="Question annexe précise"
            ).model_dump(),
        }
        with (
            patch(
                "interface.backend.orchestration_graph.matching_person_scoped_annex_persons",
                return_value=[],
            ),
            patch(
                "interface.backend.orchestration_graph.load_question_scoped_knowledge",
                return_value=("Connaissance annexe", {"enabled": True}),
            ),
        ):
            route = select_after_plan(state)

        self.assertEqual(route, "annex_direct")

    def test_plan_companies_uses_only_confident_suggestions(self) -> None:
        companies = _resolved_plan_companies(
            {
                "company_resolution": {
                    "suggestion_companies": [
                        {"company": "alk", "score": 1.0},
                        {"company": "low-confidence", "score": 0.84},
                    ]
                }
            }
        )

        self.assertEqual(companies, ["alk"])

    def test_execution_companies_exclude_ionis_stm_variants(self) -> None:
        self.assertEqual(
            _exclude_ionis_stm_companies(
                ["Ionis-STM", "IONIS STM", "Groupe Ionis-STM", "Novares"]
            ),
            ["Groupe Ionis-STM", "Novares"],
        )

    def test_execution_plan_drops_fuzzy_ionis_stm_company_suggestions(self) -> None:
        question = "Pourquoi faire Ionis-STM ?"
        result = build_execution_plan(
            {
                "payload": {"question": question},
                "planner_plan": PlannerPlan(
                    route="search", query_text=question, companies=["Ionis-STM"]
                ).model_dump(),
                "database_persons": [],
                "person_resolution": {"ambiguous": False, "suggestion_transcripts": []},
                "company_resolution": {
                    "suggestion_companies": [
                        {"company": "ionis stm", "score": 1.0},
                        {"company": "lonis stm", "score": 0.889},
                    ]
                },
                "resolved_title_hints": [],
                "planner_prompt": None,
                "planner_raw": None,
                "pydantic_verification": True,
                "reformulation_trace": {},
                "contextual_question": question,
                "reformulation_model": "test",
                "planner_model": "test",
                "analytics_sql_model": "test",
                "title_resolution": {},
                "database_company": [],
                "topic_assignment": {},
            }
        )

        self.assertEqual(result["execution_plan"]["companies"], [])

    def test_plan_title_hint_uses_the_resolved_title(self) -> None:
        self.assertEqual(
            _resolved_plan_title_hint(
                {"resolved_title_hint": "Titre canonique"}
            ),
            "Titre canonique",
        )

    def test_plan_title_hints_keep_an_explicit_title_when_unresolved(self) -> None:
        self.assertEqual(
            _resolved_plan_title_hints(
                {
                    "resolved_title_hints": [],
                    "planner_plan": PlannerPlan(
                        query_text="Question",
                        title_hints=["Titre fourni"],
                    ).model_dump(),
                }
            ),
            ["Titre fourni"],
        )

    def test_transcript_matches_remain_separate_from_speaker_matches(self) -> None:
        self.assertEqual(
            _resolved_transcript_persons(
                {
                    "database_persons": ["Marie Martin"],
                    "person_resolution": {
                        "suggestion_transcripts": [
                            {"person": "Marie Martin", "score": 1.0},
                        ]
                    },
                }
            ),
            ["Marie Martin"],
        )

    def test_sql_parent_excludes_child_query_results(self) -> None:
        output = _sql_parent_output(
            {
                "sql": "SELECT ...",
                "result_count": 1,
                "query_results": [{"text": "Transcript only on the child span."}],
                "persons_table": {
                    "sql": "SELECT ...",
                    "query_results": [{"text": "Speaker transcript."}],
                },
            }
        )

        self.assertNotIn("query_results", output)
        self.assertNotIn("query_results", output["persons_table"])
        self.assertEqual(output["result_count"], 1)

    def test_plan_persons_merge_speaker_and_transcript_matches(self) -> None:
        plan = ExecutionPlan(
            raw_question="Combien de vues pour Fadila ?",
            query_text="Combien de vues pour Fadila ?",
            query_text_bm25="vues Fadila",
            persons=["Fadila"],
        )

        plan.persons = _resolved_plan_persons(
            {
                "database_persons": ["Fadila Ouro Sama"],
                "person_resolution": {
                    "suggestion_transcripts": [
                        {"person": "Fadila", "score": 1.0},
                    ]
                },
            },
        )
        trace = _execution_plan_trace(plan)

        self.assertEqual(trace["persons"], ["Fadila Ouro Sama", "Fadila"])
        self.assertNotIn("resolved_persons", trace)


if __name__ == "__main__":
    unittest.main()
