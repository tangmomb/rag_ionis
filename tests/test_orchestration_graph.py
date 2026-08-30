from __future__ import annotations

import unittest

from interface.backend.orchestration_graph import (
    _execution_plan_trace,
    _resolved_plan_companies,
    _resolved_plan_persons,
    _resolved_plan_title_hint,
    _structured_sql_parent_output,
)
from interface.backend.schemas import ExecutionPlan


class ExecutionPlanTraceTests(unittest.TestCase):

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

    def test_plan_title_hint_uses_the_resolved_title(self) -> None:
        self.assertEqual(
            _resolved_plan_title_hint(
                {"resolved_title_hint": "Titre canonique"}
            ),
            "Titre canonique",
        )

    def test_structured_sql_parent_excludes_child_query_results(self) -> None:
        output = _structured_sql_parent_output(
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
