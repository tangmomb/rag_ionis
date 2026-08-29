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

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params

    def fetchall(self):
        if "FROM transcripts" in self.sql:
            return []
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

    def test_new_planner_keys_trigger_person_and_company_lookup(self) -> None:
        client = _Client(
            {
                "route": "multi_source",
                "sql_sub_intent": "specific_persons",
                "query_text": "Gabriel Dumy Bouygues",
                "query_text_bm25": "Gabriel Dumy Bouygues",
                "title_hint": None,
                "persons": ["Gabriel Dumy"],
                "companies": ["Bouygues"],
                "published_after": None,
                "published_before": None,
            }
        )

        plan, _, _, validated = planner.run_planner(
            "Trouve les vidéos de Gabriel Dumy chez Bouygues.",
            client,
        )

        self.assertTrue(validated)
        self.assertEqual(plan.persons, ["Gabriel Dumy"])
        self.assertEqual(plan.companies, ["Bouygues"])
        self.assertEqual(plan.sql_sub_intent, "specific_persons")
        self.assertTrue(plan.sql_main_source)

    def test_empty_person_list_does_not_query_database(self) -> None:
        with patch.object(planner, "connect_database") as connect_database:
            resolved, resolution = planner.resolve_person_filters([])

        connect_database.assert_not_called()
        self.assertEqual(resolved, [])
        self.assertFalse(resolution["applied"])

    def test_unknown_person_requires_clarification_when_enriched_transcripts_do_not_match(self) -> None:
        with patch.object(planner, "connect_database", return_value=_Connection()):
            resolved, resolution = planner.resolve_person_filters(["Zoé Inconnue"])

        self.assertEqual(resolved, [])
        self.assertTrue(resolution["ambiguous"])
        self.assertEqual(resolution["matched_in_speakers"], [])
        self.assertEqual(resolution["matched_in_transcripts"], [])
        self.assertNotIn("fallback_to_transcripts", resolution)
        self.assertNotIn("transcript_matches", resolution)
        self.assertEqual(resolution["unresolved_requests"], ["Zoé Inconnue"])
        self.assertEqual(
            resolution["message"],
            "Peux-tu préciser le nom de l'intervenant ?",
        )

    def test_unknown_speaker_uses_exact_enriched_transcript_match(self) -> None:
        class TranscriptCursor(_Cursor):
            def fetchall(self):
                if "FROM transcripts" in self.sql:
                    return [("Zoé Inconnue",)]
                return super().fetchall()

        class TranscriptConnection(_Connection):
            def cursor(self):
                return TranscriptCursor()

        with patch.object(
            planner,
            "connect_database",
            return_value=TranscriptConnection(),
        ):
            resolved, resolution = planner.resolve_person_filters(["Zoé Inconnue"])

        self.assertEqual(resolved, [])
        self.assertFalse(resolution["ambiguous"])
        self.assertEqual(resolution["matched_in_speakers"], [])
        self.assertEqual(resolution["matched_in_transcripts"], ["Zoé Inconnue"])

    def test_exact_speaker_also_keeps_exact_enriched_transcript_match(self) -> None:
        class SpeakerAndTranscriptCursor(_Cursor):
            def fetchall(self):
                if "FROM transcripts" in self.sql:
                    return [("deborah martin",)]
                return [("Déborah Martin",)]

        class SpeakerAndTranscriptConnection(_Connection):
            def cursor(self):
                return SpeakerAndTranscriptCursor()

        with patch.object(
            planner,
            "connect_database",
            return_value=SpeakerAndTranscriptConnection(),
        ):
            resolved, resolution = planner.resolve_person_filters(["deborah martin"])

        self.assertEqual(resolved, ["Déborah Martin"])
        self.assertFalse(resolution["ambiguous"])
        self.assertEqual(resolution["matched_in_speakers"], ["Déborah Martin"])
        self.assertEqual(resolution["matched_in_transcripts"], ["deborah martin"])

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
            companies=["Bouygue"],
            sql_sub_intent="specific_persons",
            sql_main_source=True,
        )

        clauses, params = retrieval.build_video_lookup_conditions(
            query,
            database_company=["bouygues"],
        )
        sql = " ".join(clauses)

        self.assertIn("person_row.title", sql)
        self.assertIn("LIKE", sql)
        self.assertIn("regexp_replace", sql)
        self.assertEqual(params, ["bouygues"])

    def test_video_title_lookup_normalizes_punctuation(self) -> None:
        query = ExecutionPlan(
            raw_question="Que dit cette vidéo ?",
            query_text="manager parfait",
            query_text_bm25="manager parfait",
            title_hint=(
                "Rendez-vous de la double compétence : "
                "Le manager parfait existe-t-il ?"
            ),
        )

        clauses, params = retrieval.build_video_lookup_conditions(query)

        self.assertEqual(clauses, [retrieval.TITLE_CONTAINS_SQL])
        self.assertIn("[^[:alnum:]]+", clauses[0])
        self.assertEqual(
            params,
            [
                "Rendez-vous de la double compétence : "
                "Le manager parfait existe-t-il ?"
            ],
        )

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
            sql_sub_intent="specific_persons",
            sql_main_source=True,
        )
        with patch.object(
            retrieval,
            "connect_database",
            side_effect=[Connection([]), Connection([transcript_row])],
        ):
            sources, trace = retrieval.lookup_video_document(
                query,
                "specific_persons",
                database_persons=["Alice Martin", "Bob Durand"],
            )

        self.assertEqual(trace["lookup_strategy"], "transcript_fallback")
        self.assertEqual(sources[0]["video_title"], "Vidéo où Alice est citée")
        self.assertIn("JOIN speakers", executed[0][0])
        self.assertIn(" OR ", executed[0][0])
        self.assertIn("JOIN transcripts", executed[1][0])
        self.assertIn(" OR ", executed[1][0])
        self.assertIn("Alice Martin", executed[1][1])
        self.assertIn("Bob Durand", executed[1][1])

    def test_lookup_combines_speaker_rows_with_verified_transcript_matches(self) -> None:
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

        speaker_row = (
            41,
            "Vidéo d'Alice",
            "https://example.test/alice",
            None,
            [{"name": "Alice Martin", "title": "Directrice"}],
            "Alice Martin prend la parole.",
            "interview",
        )
        transcript_row = (
            42,
            "Vidéo citant Bob",
            "https://example.test/bob",
            None,
            [],
            "Bob Durand est mentionné dans ce transcript enrichi.",
            "interview",
        )
        query = ExecutionPlan(
            raw_question="Trouve Alice Martin et Bob Durand",
            query_text="Alice Martin Bob Durand",
            query_text_bm25="Alice Martin Bob Durand",
            persons=["Alice Martin", "Bob Durand"],
            sql_sub_intent="specific_persons",
            sql_main_source=True,
        )

        with patch.object(
            retrieval,
            "connect_database",
            side_effect=[
                Connection([speaker_row]),
                Connection([speaker_row, transcript_row]),
            ],
        ):
            sources, trace = retrieval.lookup_video_document(
                query,
                "specific_persons",
                database_persons=["Alice Martin"],
                transcript_persons=["Bob Durand"],
            )

        self.assertEqual(trace["lookup_strategy"], "persons_table+transcript_enriched")
        self.assertEqual(
            [source["video_title"] for source in sources],
            ["Vidéo d'Alice", "Vidéo citant Bob"],
        )
        self.assertEqual(executed[1][1], ["Alice Martin", "Bob Durand"])

    def test_single_high_confidence_compound_first_name_is_auto_resolved(self) -> None:
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

    def test_transcript_match_does_not_skip_canonical_speaker_resolution(self) -> None:
        class TranscriptCursor(_Cursor):
            def fetchall(self):
                if "FROM transcripts" in self.sql:
                    return [("Lou Ann",)]
                return super().fetchall()

        class TranscriptConnection(_Connection):
            def cursor(self):
                return TranscriptCursor()

        with patch.object(
            planner,
            "connect_database",
            return_value=TranscriptConnection(),
        ):
            resolved, resolution = planner.resolve_person_filters(["Lou Ann"])

        self.assertEqual(resolved, ["Lou-Ann Corveddu"])
        self.assertTrue(resolution["auto_resolved"])
        self.assertEqual(resolution["matched_in_transcripts"], ["Lou Ann"])

    def test_single_name_can_match_a_surname(self) -> None:
        plan = PlannerPlan(route="rag", query_text="Ouyaiha", persons=["Ouyaiha"])
        with patch.object(planner, "connect_database", return_value=_Connection()):
            _, resolution = planner.resolve_person_filters(plan.persons)

        self.assertTrue(resolution["ambiguous"])
        self.assertEqual(resolution["suggestions"], ["Loucif Ouyahia"])
        self.assertGreaterEqual(resolution["suggestion_scores"][0]["score"], 0.85)

    def test_exact_surname_does_not_hide_a_first_name_spelling_difference(self) -> None:
        class MatthieuCursor(_Cursor):
            def fetchall(self):
                if "FROM transcripts" in self.sql:
                    return []
                return [("Matthieu Dumontier",)]

        class MatthieuConnection(_Connection):
            def cursor(self):
                return MatthieuCursor()

        with patch.object(
            planner,
            "connect_database",
            return_value=MatthieuConnection(),
        ):
            resolved, resolution = planner.resolve_person_filters(
                ["Mathieu Dumontier"],
            )

        self.assertEqual(resolved, ["Matthieu Dumontier"])
        self.assertFalse(resolution["ambiguous"])
        self.assertTrue(resolution["auto_resolved"])
        self.assertEqual(resolution["matched_in_speakers"], [])
        self.assertEqual(resolution["suggestions"], ["Matthieu Dumontier"])
        self.assertGreater(resolution["suggestion_scores"][0]["score"], 0.9)
        self.assertLess(resolution["suggestion_scores"][0]["score"], 1.0)

    def test_one_fuzzy_person_is_auto_resolved_beside_an_exact_person(self) -> None:
        class TwoPersonsCursor(_Cursor):
            def fetchall(self):
                if "FROM transcripts" in self.sql:
                    return []
                return [
                    ("Matthieu Dumontier",),
                    ("Paula Ventura Pinkasz",),
                ]

        class TwoPersonsConnection(_Connection):
            def cursor(self):
                return TwoPersonsCursor()

        with patch.object(
            planner,
            "connect_database",
            return_value=TwoPersonsConnection(),
        ):
            resolved, resolution = planner.resolve_person_filters(
                ["Mathieu Dumontier", "Paula Ventura Pinkasz"],
            )

        self.assertEqual(
            resolved,
            ["Matthieu Dumontier", "Paula Ventura Pinkasz"],
        )
        self.assertFalse(resolution["ambiguous"])
        self.assertTrue(resolution["auto_resolved"])
        self.assertEqual(
            resolution["matched_in_speakers"],
            ["Paula Ventura Pinkasz"],
        )
        self.assertEqual(resolution["suggestions"], ["Matthieu Dumontier"])
        self.assertGreater(resolution["suggestion_scores"][0]["score"], 0.9)

    def test_multiple_fuzzy_persons_require_clarification(self) -> None:
        plan = PlannerPlan(
            route="multi_source",
            query_text="Ouyaiha Montessi",
            persons=["Ouyaiha", "Montessi"],
        )
        with patch.object(planner, "connect_database", return_value=_Connection()):
            resolved, resolution = planner.resolve_person_filters(plan.persons)

        self.assertEqual(resolved, [])
        self.assertTrue(resolution["ambiguous"])
        self.assertFalse(resolution["auto_resolved"])
        self.assertEqual(
            resolution["suggestions"],
            ["Loucif Ouyahia", "Yannick Montesi"],
        )

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
        self.assertEqual(
            resolution["message"],
            "Vous parlez de Lou-Ann Martin, Lou-Ann Corveddu ?",
        )

    def test_ambiguous_person_does_not_discard_exact_matches(self) -> None:
        class MixedCursor(_Cursor):
            def fetchall(self):
                return [
                    ("Alice Martin",),
                    ("Lou-Ann Corveddu",),
                    ("Lou-Ann Martin",),
                ]

        class MixedConnection(_Connection):
            def cursor(self):
                return MixedCursor()

        with patch.object(planner, "connect_database", return_value=MixedConnection()):
            resolved, resolution = planner.resolve_person_filters(
                ["Alice Martin", "Lou Ann"],
            )

        self.assertEqual(resolved, ["Alice Martin"])
        self.assertTrue(resolution["ambiguous"])
        self.assertEqual(resolution["ambiguous_requests"], ["Lou Ann"])
