from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from interface.backend import planner, retrieval
from interface.backend.schemas import ExecutionPlan, PlannerPlan


class _Cursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, sql):
        self.sql = sql

    def fetchall(self):
        return [
            ("Loucif Ouyahia",),
            ("Yannick Montesi",),
            ("Lou-Ann Corveddu",),
        ]


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self):
        return _Cursor()


class _Responses:
    def __init__(self, payload: dict):
        self.payload = payload

    def create(self, **kwargs):
        return SimpleNamespace(output_text=json.dumps(self.payload))


class _Client:
    def __init__(self, payload: dict):
        self.responses = _Responses(payload)


class PersonResolutionTests(unittest.TestCase):
    def test_planner_persons_are_preserved(self) -> None:
        client = _Client(
            {
                "route": "rag",
                "sql_sub_intent": None,
                "query_text": "stages",
                "query_text_bm25": "stages",
                "title_hint": None,
                "published_after": None,
                "published_before": None,
                "persons": ["Personne identifiée par le planner"],
            }
        )

        plan, _, _, validated = planner.run_planner("Que dit la vidéo ?", client)

        self.assertTrue(validated)
        self.assertEqual(plan.persons, ["Personne identifiée par le planner"])

    def test_empty_person_list_does_not_query_database(self) -> None:
        with patch.object(planner, "connect_database") as connect_database:
            resolved, resolution = planner.resolve_person_filters([])

        connect_database.assert_not_called()
        self.assertEqual(resolved, [])
        self.assertFalse(resolution["applied"])

    def test_unknown_person_falls_back_to_transcripts_without_clarification(self) -> None:
        with patch.object(planner, "connect_database", return_value=_Connection()):
            resolved, resolution = planner.resolve_person_filters(["Zoé Inconnue"])

        self.assertEqual(resolved, [])
        self.assertFalse(resolution["ambiguous"])
        self.assertTrue(resolution["fallback_to_transcripts"])
        self.assertEqual(resolution["unresolved_requests"], ["Zoé Inconnue"])

    def test_company_typo_is_resolved_to_close_term_contained_in_title(self) -> None:
        class TitleCursor(_Cursor):
            def fetchall(self):
                return [
                    ("Responsable affaires Bouygues",),
                    ("Directrice des partenariats EDF",),
                ]

        class TitleConnection(_Connection):
            def cursor(self):
                return TitleCursor()

        with patch.object(
            planner,
            "connect_database",
            return_value=TitleConnection(),
        ):
            resolved, resolution = planner.resolve_company_filters(["Bouygue"])

        self.assertEqual(resolved, ["bouygues"])
        self.assertEqual(resolution["matches"][0]["title"], "Responsable affaires Bouygues")
        self.assertGreaterEqual(resolution["matches"][0]["score"], 0.9)

    def test_company_lookup_filters_person_title_with_resolved_term(self) -> None:
        query = ExecutionPlan(
            raw_question="Trouve les vidéos de Bouygue",
            query_text="Bouygue",
            query_text_bm25="Bouygue",
            company=["Bouygue"],
            sql_sub_intent="lookup",
            sql_main_source=True,
        )

        clauses, params = retrieval.build_video_lookup_conditions(
            query,
            database_company=["bouygues"],
        )
        sql = " ".join(clauses)

        self.assertIn("person_row.title", sql)
        self.assertIn("LIKE", sql)
        self.assertEqual(params, ["%bouygues%"])

    def test_lookup_falls_back_from_persons_table_to_transcripts(self) -> None:
        executed: list[tuple[str, list]] = []

        class Cursor:
            def __init__(self, rows):
                self.rows = rows

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params):
                executed.append((sql, params))

            def fetchall(self):
                return self.rows

        class Connection:
            def __init__(self, rows):
                self.rows = rows

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def cursor(self):
                return Cursor(self.rows)

        transcript_row = (
            42,
            "Vidéo où Alice est citée",
            "https://example.test/video",
            None,
            [],
            "Alice Martin est mentionnée dans ce transcript.",
            "interview",
        )
        query = ExecutionPlan(
            raw_question="Trouve une vidéo avec Alice Martin ou Bob Durand",
            query_text="Alice Martin Bob Durand",
            query_text_bm25="Alice Martin Bob Durand",
            persons=["Alice Martin", "Bob Durand"],
            sql_sub_intent="lookup",
            sql_main_source=True,
        )
        with patch.object(
            retrieval,
            "connect_database",
            side_effect=[Connection([]), Connection([transcript_row])],
        ):
            sources, trace = retrieval.lookup_video_document(
                query,
                "lookup",
                database_persons=["Alice Martin", "Bob Durand"],
            )

        self.assertEqual(trace["lookup_strategy"], "transcript_fallback")
        self.assertEqual(sources[0]["video_title"], "Vidéo où Alice est citée")
        self.assertIn("FROM speakers", executed[0][0])
        self.assertIn(" OR ", executed[0][0])
        self.assertIn("JOIN transcripts", executed[1][0])
        self.assertIn(" OR ", executed[1][0])
        self.assertIn("%Alice Martin%", executed[1][1])
        self.assertIn("%Bob Durand%", executed[1][1])

    def test_compound_first_name_is_auto_resolved_when_it_is_the_only_match(self) -> None:
        plan = PlannerPlan(route="rag", query_text="Lou Ann", persons=["Lou Ann"])
        with patch.object(planner, "connect_database", return_value=_Connection()):
            resolved, resolution = planner.resolve_person_filters(plan.persons)

        self.assertEqual(resolved, ["Lou-Ann Corveddu"])
        self.assertFalse(resolution["ambiguous"])
        self.assertTrue(resolution["auto_resolved"])
        self.assertEqual(resolution["suggestions"], ["Lou-Ann Corveddu"])
        self.assertEqual(
            resolution["suggestion_scores"],
            [{"person": "Lou-Ann Corveddu", "score": 1.0}],
        )

    def test_single_name_can_match_a_surname(self) -> None:
        plan = PlannerPlan(route="rag", query_text="Ouyaiha", persons=["Ouyaiha"])
        with patch.object(planner, "connect_database", return_value=_Connection()):
            _, resolution = planner.resolve_person_filters(plan.persons)

        self.assertEqual(resolution["suggestions"], ["Loucif Ouyahia"])
        self.assertGreaterEqual(resolution["suggestion_scores"][0]["score"], 0.85)

    def test_multiple_matches_still_require_clarification(self) -> None:
        class TwoLouAnnCursor(_Cursor):
            def fetchall(self):
                return [("Lou-Ann Corveddu",), ("Lou-Ann Martin",)]

        class TwoLouAnnConnection(_Connection):
            def cursor(self):
                return TwoLouAnnCursor()

        plan = PlannerPlan(route="rag", query_text="Lou Ann", persons=["Lou Ann"])
        with patch.object(planner, "connect_database", return_value=TwoLouAnnConnection()):
            resolved, resolution = planner.resolve_person_filters(plan.persons)

        self.assertEqual(resolved, [])
        self.assertTrue(resolution["ambiguous"])
        self.assertEqual(resolution["suggestions"], ["Lou-Ann Martin", "Lou-Ann Corveddu"])
