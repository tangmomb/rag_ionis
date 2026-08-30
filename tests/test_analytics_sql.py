from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from interface.backend import analytics_sql, planner
from interface.backend.schemas import ExecutionPlan, PlannerPlan


class _Responses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class _Cursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object]] = []
        self.last_sql = ""
        self.description = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.executed.append((sql, params))
        if sql.startswith("SELECT * FROM"):
            self.description = [
                SimpleNamespace(name="video_id"),
                SimpleNamespace(name="video_title"),
                SimpleNamespace(name="video_url"),
                SimpleNamespace(name="view_count"),
            ]

    def fetchone(self):
        return ([{"Plan": {"Node Type": "Limit", "Total Cost": 12.34}}],)

    def fetchall(self):
        return [(12, "Vidéo populaire", "https://example.test/video", 21136)]


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self):
        return self._cursor


class AnalyticsSqlTests(unittest.TestCase):
    def test_prompt_exposes_only_analytics_schema_and_latest_snapshot_rule(self) -> None:
        query = ExecutionPlan(
            raw_question="Quelle vidéo a le plus de vues ?",
            query_text="Quelle vidéo a le plus de vues ?",
            query_text_bm25="plus de vues",
        )

        system_prompt, user_prompt = analytics_sql.build_analytics_sql_prompt(
            query.raw_question,
            query,
        )

        self.assertIn("videos(", system_prompt)
        self.assertIn("stats(", system_prompt)
        self.assertIn("dernier snapshot", system_prompt)
        self.assertIn("jamais un LIMIT 1 global", system_prompt)
        self.assertIn("comparaison retourne les stats", system_prompt)
        self.assertIn("Toute limite finale doit être au moins 6", system_prompt)
        self.assertIn("Le seul LIMIT 1 autorisé", system_prompt)
        self.assertIn("jamais AND", system_prompt)
        self.assertIn("WHERE sp.name ILIKE %s OR sp.name ILIKE %s", system_prompt)
        self.assertIn("thumbnail_medium_url AS thumbnail_medium_url", system_prompt)
        self.assertIn("v.title ILIKE %s", system_prompt)
        self.assertIn("title_hint` nul signifie aucun filtre", system_prompt)
        self.assertIn("invente jamais un titre", system_prompt)
        self.assertIn("v.id AS video_id", system_prompt)
        self.assertIn("SELECT", system_prompt)
        self.assertEqual(system_prompt.count("Exemple"), 1)
        self.assertNotIn("transcripts(", system_prompt)
        self.assertIn("Quelle vidéo a le plus de vues ?", user_prompt)

    def test_analytics_source_keeps_returned_thumbnail(self) -> None:
        sources = analytics_sql.analytics_rows_to_sources(
            ["video_id", "video_title", "video_url", "thumbnail_medium_url"],
            [
                (
                    12,
                    "Vidéo test",
                    "https://example.test/video",
                    "https://example.test/thumb.jpg",
                )
            ],
        )

        self.assertEqual(
            sources[0]["thumbnail_medium_url"],
            "https://example.test/thumb.jpg",
        )

    def test_validator_accepts_safe_parameterized_ranking(self) -> None:
        sql = (
            "SELECT v.id AS video_id, v.title AS video_title, s.view_count "
            "FROM videos v JOIN LATERAL (SELECT view_count FROM stats "
            "WHERE video_id = v.id ORDER BY snapshot_date DESC LIMIT 1) s ON TRUE "
            "ORDER BY s.view_count DESC LIMIT %s"
        )

        validation = analytics_sql.validate_analytics_sql(sql, [1])

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["relations"], ["stats", "videos"])

    def test_validator_rejects_global_latest_stats_cte(self) -> None:
        sql = (
            "WITH latest_stats AS (SELECT video_id, view_count FROM stats "
            "ORDER BY snapshot_date DESC, data_collected_date DESC, id DESC LIMIT 1) "
            "SELECT v.id AS video_id, ls.view_count FROM videos v "
            "JOIN latest_stats ls ON ls.video_id = v.id ORDER BY ls.view_count DESC LIMIT %s"
        )

        validation = analytics_sql.validate_analytics_sql(sql, [1])

        self.assertFalse(validation["valid"])
        self.assertIn("stats_latest_snapshot_not_per_video", validation["errors"])

    def test_validator_rejects_mutation_unknown_table_and_bad_params(self) -> None:
        mutation = analytics_sql.validate_analytics_sql(
            "WITH removed AS (DELETE FROM videos RETURNING id) SELECT id FROM removed",
            [],
        )
        unknown_table = analytics_sql.validate_analytics_sql(
            "SELECT secret FROM private_table",
            [],
        )
        bad_params = analytics_sql.validate_analytics_sql(
            "SELECT COUNT(*) FROM videos WHERE video_type = %s",
            [],
        )
        literal_value = analytics_sql.validate_analytics_sql(
            "SELECT COUNT(*) FROM videos WHERE video_type = 'interview'",
            [],
        )

        self.assertFalse(mutation["valid"])
        self.assertIn("disallowed_keyword", mutation["errors"])
        self.assertFalse(unknown_table["valid"])
        self.assertIn("unknown_relations:private_table", unknown_table["errors"])
        self.assertFalse(bad_params["valid"])
        self.assertIn("placeholder_count_mismatch", bad_params["errors"])
        self.assertFalse(literal_value["valid"])
        self.assertIn("literal_values_forbidden", literal_value["errors"])

    def test_analytics_policy_is_preserved_when_company_is_also_detected(self) -> None:
        question = "Quelle vidéo de Microsoft a le plus de vues ?"
        plan = PlannerPlan(
            route="rag",
            query_text=question,
            sql_sub_intent="analytics",
            companies=["Microsoft"],
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertEqual(plan.sql_sub_intent, "analytics")
        self.assertTrue(plan.sql_main_source)

    def test_publication_date_is_routed_to_analytics(self) -> None:
        question = "Quelle est la date de publication de la vidéo Parcoursup ?"
        plan = PlannerPlan(
            route="rag",
            query_text=question,
            title_hint="Parcoursup",
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertEqual(plan.sql_sub_intent, "analytics")
        self.assertTrue(plan.sql_main_source)

    def test_legacy_stats_intent_is_normalized_to_analytics(self) -> None:
        normalized = planner.normalize_planner_output(
            {
                "route": "rag",
                "sql_sub_intent": "stats",
                "query_text": "Quelle vidéo a le plus de vues ?",
                "persons": [],
                "companies": [],
            }
        )

        self.assertEqual(normalized["sql_sub_intent"], "analytics")

    def test_deterministic_analytics_deduplicates_entity_videos_before_loading_stats(self) -> None:
        query = ExecutionPlan(
            raw_question="Combien de vues pour Alice chez Acme ?",
            query_text="Combien de vues pour Alice chez Acme ?",
            query_text_bm25="vues Alice Acme",
            title_hint="Vidéo Alice",
            persons=["Alice Martin"],
            companies=["acme"],
            route="rag",
            sql_sub_intent="analytics",
            sql_main_source=True,
        )
        lookup_calls: list[dict] = []
        recorded_spans: list[str] = []

        @contextmanager
        def record_trace(name: str, **_kwargs):
            recorded_spans.append(name)
            yield SimpleNamespace(set_output=lambda _value: None)

        def lookup(entity, _query):
            lookup_calls.append(entity)
            return (
                [
                    {
                        "video_id": 42,
                        "video_title": "Vidéo Alice",
                        "video_url": "https://example.test/alice",
                        "thumbnail_medium_url": None,
                    }
                ],
                {"sql": "SELECT ...", "params": [entity["value"]]},
            )

        source = {
            "chunk_id": 42,
            "video_title": "Vidéo Alice",
            "video_url": "https://example.test/alice",
            "text": "Historique complet des statistiques",
            "stats": [{"snapshot_date": "2026-08-01", "view_count": 100}],
        }
        with (
            patch.object(analytics_sql, "_lookup_analytics_entity_videos", side_effect=lookup),
            patch.object(
                analytics_sql,
                "_analytics_stats_sources",
                return_value=([source], {"video_count": 1, "snapshot_count": 1}),
            ) as stats,
            patch.object(analytics_sql, "trace_operation", side_effect=record_trace),
        ):
            sources, trace = analytics_sql.run_deterministic_analytics(query)

        self.assertEqual([item["kind"] for item in lookup_calls], ["person", "company", "title"])
        self.assertEqual(trace["candidate_video_count"], 1)
        self.assertEqual(trace["result_count"], 1)
        self.assertEqual(sources, [source])
        self.assertEqual(stats.call_args.args[0], [42])
        self.assertEqual(
            recorded_spans,
            ["analytics_entity_lookup", "analytics_entity_lookup", "analytics_entity_lookup", "analytics_all_video_stats"],
        )

    def test_entity_lookup_without_date_filter_builds_a_valid_where_clause(self) -> None:
        class Cursor:
            def __init__(self) -> None:
                self.calls: list[tuple[str, object]] = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def execute(self, sql, params=None) -> None:
                self.calls.append((sql, params))

            def fetchall(self):
                return []

        cursor = Cursor()
        query = ExecutionPlan(
            raw_question="Question",
            query_text="Question",
            query_text_bm25="Question",
        )
        with patch.object(
            analytics_sql,
            "connect_analytics_database",
            return_value=_Connection(cursor),
        ):
            videos, _trace = analytics_sql._lookup_analytics_entity_videos(
                {"kind": "person", "value": "Lou-Ann Corveddu"}, query
            )

        sql = cursor.calls[-1][0]
        self.assertEqual(videos, [])
        self.assertIn("WHERE (EXISTS", sql)
        self.assertIn("FROM transcripts transcript_row", sql)
        self.assertIn("transcript_row.transcript_enriched IS NOT NULL", sql)
        self.assertEqual(cursor.calls[-1][1], ["Lou-Ann Corveddu", "Lou-Ann Corveddu"])
        self.assertNotIn("\n          TRUE", sql)

    def test_text_to_sql_is_validated_explained_executed_and_traced(self) -> None:
        sql = (
            "SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, "
            "s.view_count FROM videos v JOIN LATERAL (SELECT view_count FROM stats "
            "WHERE video_id = v.id ORDER BY snapshot_date DESC, "
            "data_collected_date DESC, id DESC LIMIT 1) s ON TRUE "
            "ORDER BY s.view_count DESC NULLS LAST LIMIT %s"
        )
        responses = _Responses(
            '{"sql":' + repr(sql).replace("'", '"') + ',"params":[1]}'
        )
        client = SimpleNamespace(responses=responses)
        cursor = _Cursor()
        recorded_spans: list[str] = []
        recorded_outputs: dict[str, object] = {}

        @contextmanager
        def record_trace(name: str, **kwargs):
            del kwargs
            recorded_spans.append(name)
            yield SimpleNamespace(
                set_output=lambda value: recorded_outputs.__setitem__(name, value)
            )

        query = ExecutionPlan(
            raw_question="Quelle vidéo a le plus de vues ?",
            query_text="Quelle vidéo a le plus de vues ?",
            query_text_bm25="plus de vues",
            route="rag",
            sql_sub_intent="analytics",
            sql_main_source=True,
        )
        with (
            patch.object(
                analytics_sql,
                "connect_analytics_database",
                return_value=_Connection(cursor),
            ),
            patch.object(analytics_sql, "trace_operation", side_effect=record_trace),
        ):
            sources, trace = analytics_sql.run_analytics_text_to_sql(
                query,
                client,
                "mistral-medium-latest",
            )

        self.assertEqual(
            recorded_spans,
            [
                "rag.analytics.sql_generation",
                "rag.analytics.sql_validation",
                "rag.analytics.sql_cost_validation",
                "rag.analytics.sql_execution",
            ],
        )
        self.assertEqual(trace["status"], "executed")
        self.assertEqual(
            trace["cost_validation"],
            {
                "valid": True,
                "total_cost": 12.34,
                "max_total_cost": analytics_sql.DEFAULT_MAX_ANALYTICS_TOTAL_COST,
            },
        )
        self.assertNotIn(
            "explain",
            recorded_outputs["rag.analytics.sql_cost_validation"],
        )
        self.assertEqual(
            responses.calls[0]["response_schema"],
            analytics_sql.ANALYTICS_SQL_RESPONSE_SCHEMA,
        )
        self.assertEqual(sources[0]["chunk_id"], 12)
        self.assertIn("21136", sources[0]["text"])
        self.assertEqual(cursor.executed[0][0], "SET TRANSACTION READ ONLY")
        self.assertTrue(cursor.executed[2][0].startswith("EXPLAIN (FORMAT JSON)"))
        self.assertEqual(cursor.executed[3][0], "SET TRANSACTION READ ONLY")
        self.assertTrue(cursor.executed[5][0].startswith("SELECT * FROM"))

    def test_explain_returns_only_cost_summary(self) -> None:
        cursor = _Cursor()

        with patch.object(
            analytics_sql,
            "connect_analytics_database",
            return_value=_Connection(cursor),
        ):
            validation = analytics_sql.explain_analytics_sql(
                "SELECT COUNT(*) AS video_count FROM videos",
                [],
            )

        self.assertEqual(
            validation,
            {
                "valid": True,
                "total_cost": 12.34,
                "max_total_cost": analytics_sql.DEFAULT_MAX_ANALYTICS_TOTAL_COST,
            },
        )

    def test_explain_cost_above_limit_is_rejected_before_execution(self) -> None:
        sql = "SELECT COUNT(*) AS video_count FROM videos"
        responses = _Responses('{"sql":"' + sql + '","params":[]}')
        client = SimpleNamespace(responses=responses)
        query = ExecutionPlan(
            raw_question="Combien de vidéos ?",
            query_text="Combien de vidéos ?",
            query_text_bm25="nombre vidéos",
            route="rag",
            sql_sub_intent="analytics",
            sql_main_source=True,
        )
        with (
            patch.object(
                analytics_sql,
                "explain_analytics_sql",
                return_value={
                    "valid": False,
                    "total_cost": 200_000.0,
                    "max_total_cost": 100_000.0,
                    "explain": [],
                },
            ),
            patch.object(analytics_sql, "execute_analytics_sql") as execute,
        ):
            sources, trace = analytics_sql.run_analytics_text_to_sql(
                query,
                client,
                "mistral-medium-latest",
            )

        self.assertEqual(sources, [])
        self.assertEqual(trace["status"], "cost_rejected")
        self.assertNotIn("explain", trace["cost_validation"])
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
