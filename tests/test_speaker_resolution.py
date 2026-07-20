from __future__ import annotations

import unittest
from unittest.mock import patch

from interface.backend import planner
from interface.backend.schemas import PlannerPlan


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


class SpeakerResolutionTests(unittest.TestCase):
    def test_compound_first_name_is_auto_resolved_when_it_is_the_only_match(self) -> None:
        plan = PlannerPlan(route="rag", query_text="Lou Ann", speakers=["Lou Ann"])
        with patch.object(planner, "connect_database", return_value=_Connection()):
            resolved, resolution = planner.resolve_speaker_filters("video avec Lou Ann", plan)

        self.assertEqual(resolved, ["Lou-Ann Corveddu"])
        self.assertFalse(resolution["ambiguous"])
        self.assertTrue(resolution["auto_resolved"])
        self.assertEqual(resolution["suggestions"], ["Lou-Ann Corveddu"])
        self.assertEqual(
            resolution["suggestion_scores"],
            [{"speaker": "Lou-Ann Corveddu", "score": 1.0}],
        )

    def test_single_name_can_match_a_surname(self) -> None:
        plan = PlannerPlan(route="rag", query_text="Ouyaiha", speakers=["Ouyaiha"])
        with patch.object(planner, "connect_database", return_value=_Connection()):
            _, resolution = planner.resolve_speaker_filters("video avec Ouyaiha", plan)

        self.assertEqual(resolution["suggestions"], ["Loucif Ouyahia"])
        self.assertGreaterEqual(resolution["suggestion_scores"][0]["score"], 0.85)

    def test_multiple_matches_still_require_clarification(self) -> None:
        class TwoLouAnnCursor(_Cursor):
            def fetchall(self):
                return [("Lou-Ann Corveddu",), ("Lou-Ann Martin",)]

        class TwoLouAnnConnection(_Connection):
            def cursor(self):
                return TwoLouAnnCursor()

        plan = PlannerPlan(route="rag", query_text="Lou Ann", speakers=["Lou Ann"])
        with patch.object(planner, "connect_database", return_value=TwoLouAnnConnection()):
            resolved, resolution = planner.resolve_speaker_filters("video avec Lou Ann", plan)

        self.assertEqual(resolved, [])
        self.assertTrue(resolution["ambiguous"])
        self.assertEqual(resolution["suggestions"], ["Lou-Ann Martin", "Lou-Ann Corveddu"])
