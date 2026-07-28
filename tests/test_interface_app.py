from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from interface.app import RagRequest, app
from interface.backend import generation, orchestration, planner, retrieval
from interface.backend.config import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
)
from interface.backend.schemas import ExecutionPlan, PlannerPlan


class InterfaceAppTests(unittest.TestCase):
    def test_planner_uses_sol_by_default(self) -> None:
        self.assertEqual(DEFAULT_PLANNER_MODEL, "gpt-5.6-sol")
        self.assertEqual(DEFAULT_REFORMULATION_MODEL, "gpt-5.6-sol")
        self.assertEqual(DEFAULT_GENERATION_MODEL, "gpt-5.6-sol")

    def test_reformulation_prompt_has_one_narrow_responsibility(self) -> None:
        system_prompt, user_prompt = planner.build_question_reformulation_prompt(
            "Et pour elle ?",
            [{"role": "user", "text": "Que dit Alice Martin ?"}],
        )

        self.assertLess(len(system_prompt), 700)
        self.assertIn("besoin de l'historique", system_prompt)
        self.assertIn("follow_up", system_prompt)
        self.assertIn("reformulated_question", system_prompt)
        self.assertNotIn("point d'interrogation final", system_prompt)
        self.assertIn("Message actuel : Et pour elle ?", user_prompt)

    def test_temporal_video_question_forces_transcript_qa(self) -> None:
        question = (
            "À quel moment de la vidéo Matthieu répond-il à la question sur "
            "le domaine dans lequel il se projette professionnellement ?"
        )
        plan = PlannerPlan(
            route="rag",
            query_text=question,
            use_rag=True,
            sql_main_source=False,
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertTrue(plan.sql_main_source)
        self.assertEqual(plan.sql_sub_intent, "transcript_qa")

    def test_planner_prompt_describes_transcript_intent(self) -> None:
        system_prompt, _ = planner.build_planner_prompt("Question")

        self.assertIn("transcript_verbatim", system_prompt)
        self.assertIn("transcript complet", system_prompt)

    def test_content_question_with_explicit_title_uses_full_transcript(self) -> None:
        question = (
            "Que dit Matthieu dans la vidéo « Apporter ma pierre à l’édifice "
            "Énergies Renouvelables – Matthieu, Responsable Affaires, Bouygues » ?"
        )
        plan = PlannerPlan(
            route="rag",
            query_text=question,
            title_hint=planner.extract_video_title_hint(question),
            use_rag=True,
            sql_main_source=True,
            sql_sub_intent="specific_persons",
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertTrue(plan.sql_main_source)
        self.assertEqual(plan.sql_sub_intent, "transcript_qa")
        self.assertTrue(plan.use_rag)
        self.assertEqual(
            plan.title_hint,
            "Apporter ma pierre à l’édifice Énergies Renouvelables – "
            "Matthieu, Responsable Affaires, Bouygues",
        )

    def test_explicit_transcript_request_keeps_sql_main_source(self) -> None:
        question = "Donne le transcript complet de la vidéo « Parcoursup »."
        plan = PlannerPlan(
            route="rag",
            query_text=question,
            use_rag=True,
            sql_main_source=False,
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertTrue(plan.sql_main_source)
        self.assertEqual(plan.sql_sub_intent, "transcript_verbatim")

    def test_sql_execution_plan_has_no_rag_limits(self) -> None:
        payload = RagRequest(question="Donne le transcript de la vidéo.")
        plan = PlannerPlan(
            route="rag",
            query_text=payload.question,
            use_rag=True,
            sql_main_source=True,
            sql_sub_intent="transcript_verbatim",
        )

        execution_plan = planner.build_execution_plan(payload, plan)

        self.assertIsNone(execution_plan.top_k)
        self.assertIsNone(execution_plan.final_k)

    def test_document_rag_execution_plan_keeps_rag_limits(self) -> None:
        payload = RagRequest(question="Que dit-on sur Parcoursup ?")
        plan = PlannerPlan(
            route="rag",
            query_text=payload.question,
            use_rag=True,
            sql_main_source=False,
        )

        execution_plan = planner.build_execution_plan(payload, plan)

        self.assertEqual(execution_plan.top_k, 40)
        self.assertEqual(execution_plan.final_k, 5)

    def test_execution_plan_only_exposes_requested_persons_and_companies(self) -> None:
        payload = RagRequest(question="Trouve les vidéos de Gabriel chez Bouygues.")
        plan = PlannerPlan(
            route="rag",
            query_text=payload.question,
            persons=["Gabriel Dumy"],
            companies=["Bouygues"],
            sql_sub_intent="specific_persons",
        )

        serialized_plan = planner.build_execution_plan(payload, plan).model_dump()

        self.assertEqual(serialized_plan["persons"], ["Gabriel Dumy"])
        self.assertEqual(serialized_plan["companies"], ["Bouygues"])
        self.assertNotIn("company", serialized_plan)
        self.assertNotIn("database_persons", serialized_plan)
        self.assertNotIn("database_company", serialized_plan)

    def test_transcript_prompt_answers_content_question_without_verbatim(self) -> None:
        prompt = generation.build_sql_sub_intent_prompt("transcript_qa")

        self.assertIn("synthetise", prompt)
        self.assertIn("ne restitue pas le transcript en entier", prompt)

    def test_transcript_prompt_preserves_explicit_verbatim_request(self) -> None:
        prompt = generation.build_sql_sub_intent_prompt("transcript_verbatim")

        self.assertIn("Restitue le transcript fidelement", prompt)

    def test_summary_with_explicit_title_uses_transcript_qa(self) -> None:
        question = "Résume la vidéo « Titre exact »."
        plan = PlannerPlan(
            route="rag",
            query_text=question,
            title_hint=planner.extract_video_title_hint(question),
            use_rag=True,
            sql_main_source=False,
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertTrue(plan.sql_main_source)
        self.assertEqual(plan.sql_sub_intent, "transcript_qa")

    def test_transcript_sub_intents_select_distinct_database_columns(self) -> None:
        executed_sql: list[str] = []

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params):
                executed_sql.append(sql)

            def fetchone(self):
                return None

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def cursor(self):
                return Cursor()

        query = ExecutionPlan(
            raw_question="Question",
            query_text="Question",
            query_text_bm25="Question",
            title_hint="Titre exact",
        )
        with patch.object(retrieval, "connect_database", return_value=Connection()):
            retrieval.lookup_video_document(query, "transcript_verbatim")
            retrieval.lookup_video_document(query, "transcript_qa")

        self.assertIn("t.transcript IS NOT NULL", executed_sql[0])
        self.assertNotIn("transcript_enriched", executed_sql[0])
        self.assertIn("t.transcript_enriched IS NOT NULL", executed_sql[1])

    def test_specific_persons_without_named_person_uses_rag_as_main_source(self) -> None:
        question = "Qui intervient dans la vidéo « Titre exact » ?"
        plan = PlannerPlan(
            route="rag",
            query_text=question,
            title_hint=planner.extract_video_title_hint(question),
            use_rag=True,
            sql_main_source=False,
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertFalse(plan.sql_main_source)
        self.assertEqual(plan.sql_sub_intent, "specific_persons")

    def test_person_job_question_forces_specific_persons_sql(self) -> None:
        question = "Quel est le métier de Gabriel Dumy ?"
        plan = PlannerPlan(
            route="rag",
            query_text="métier profession fonction Gabriel Dumy",
            query_text_bm25="Gabriel Dumy métier profession",
            persons=["Gabriel Dumy"],
            use_rag=True,
            sql_main_source=False,
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertTrue(plan.sql_main_source)
        self.assertEqual(plan.sql_sub_intent, "specific_persons")

    def test_identified_person_always_forces_specific_persons(self) -> None:
        question = "Que dit Gabriel Dumy dans cette vidéo ?"
        plan = PlannerPlan(
            route="rag",
            query_text=question,
            persons=["Gabriel Dumy"],
            sql_sub_intent="transcript_qa",
        )

        planner.apply_deterministic_sql_policy(question, plan)

        self.assertEqual(plan.sql_sub_intent, "specific_persons")
        self.assertTrue(plan.sql_main_source)

    def test_specific_persons_without_identified_person_is_not_sql_main_source(self) -> None:
        plan = PlannerPlan(
            route="rag",
            query_text="vidéos sur les stages",
            sql_sub_intent="specific_persons",
            persons=[],
        )

        planner.derive_plan_sources(plan)

        self.assertFalse(plan.sql_main_source)
        self.assertTrue(plan.use_rag)

    def test_specific_persons_with_identified_company_is_sql_main_source(self) -> None:
        plan = PlannerPlan(
            route="rag",
            query_text="Bouygues",
            sql_sub_intent="specific_persons",
            companies=["Bouygues"],
        )

        planner.derive_plan_sources(plan)

        self.assertTrue(plan.sql_main_source)

    def test_planner_prompt_maps_person_to_specific_persons(self) -> None:
        system_prompt, _ = planner.build_planner_prompt("Quel est le métier de Gabriel Dumy ?")

        self.assertIn("specific_persons", system_prompt)
        self.assertIn("personnes ou entreprises", system_prompt)

    def test_planner_prompt_routes_multiple_persons_to_multi_source(self) -> None:
        system_prompt, _ = planner.build_planner_prompt(
            "Compare les interventions de Gabriel Dumy et Alice Martin."
        )

        self.assertIn("plus d'une personne ou entreprise", system_prompt)
        self.assertIn("multi_source", system_prompt)

    def test_planner_identifies_companies_in_dedicated_key(self) -> None:
        system_prompt, _ = planner.build_planner_prompt(
            "Trouve les vidéos qui parlent de Bouygues et EDF."
        )

        self.assertIn("companies", system_prompt)
        self.assertIn("entreprises mentionnées", system_prompt)

    def test_planner_prompt_omits_derived_source_flags(self) -> None:
        system_prompt, _ = planner.build_planner_prompt("Question")

        self.assertNotIn("use_memory", system_prompt)
        self.assertNotIn("use_rag", system_prompt)
        self.assertNotIn("sql_main_source", system_prompt)

    def test_source_flags_are_derived_from_sql_sub_intent(self) -> None:
        normalized = planner.normalize_planner_output(
            {
                "route": "rag",
                "sql_sub_intent": "specific_persons",
                "query_text": "Gabriel Dumy",
                "query_text_bm25": "Gabriel Dumy",
                "title_hint": None,
                "persons": ["Gabriel Dumy"],
                "published_after": None,
                "published_before": None,
                "use_memory": True,
                "use_rag": False,
                "sql_main_source": False,
            }
        )
        plan = PlannerPlan.model_validate(normalized)

        planner.derive_plan_sources(plan)

        self.assertFalse(plan.use_memory)
        self.assertTrue(plan.use_rag)
        self.assertTrue(plan.sql_main_source)

    def test_lookup_returns_person_name_and_title(self) -> None:
        executed_sql: list[str] = []

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params):
                executed_sql.append(sql)

            def fetchall(self):
                return [
                    (
                        42,
                        "Portrait de Gabriel Dumy",
                        "https://example.test/video",
                        None,
                        [{"name": "Gabriel Dumy", "title": "Responsable affaires"}],
                        "[00:01] Gabriel Dumy: Bonjour.",
                        "interview",
                    )
                ]

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def cursor(self):
                return Cursor()

        query = ExecutionPlan(
            raw_question="Quel est le métier de Gabriel Dumy ?",
            query_text="métier Gabriel Dumy",
            query_text_bm25="Gabriel Dumy métier",
            persons=["Gabriel Dumy"],
            sql_main_source=True,
            sql_sub_intent="specific_persons",
        )
        with patch.object(retrieval, "connect_database", return_value=Connection()):
            sources, trace = retrieval.lookup_video_document(
                query,
                "specific_persons",
                database_persons=["Gabriel Dumy"],
            )

        self.assertIn("person_row.title", executed_sql[0])
        self.assertIn("transcript_enriched", executed_sql[0])
        self.assertIn("Gabriel Dumy: Responsable affaires", sources[0]["text"])
        self.assertEqual(
            sources[0]["transcript"],
            "[00:01] Gabriel Dumy: Bonjour.",
        )
        self.assertEqual(
            sources[0]["person_details"],
            [{"name": "Gabriel Dumy", "title": "Responsable affaires"}],
        )
        self.assertEqual(trace["mode"], "specific_persons")

    def test_every_sql_sub_intent_has_a_dedicated_prompt(self) -> None:
        prompts = {
            intent: generation.build_sql_sub_intent_prompt(intent)
            for intent in (
                "specific_persons",
                "stats",
                "description",
                "transcript_verbatim",
                "transcript_qa",
            )
        }

        self.assertEqual(len(set(prompts.values())), len(prompts))

    def test_answer_prompts_do_not_require_question_reformulation(self) -> None:
        prompts = [
            generation.FINAL_ANSWER_STYLE,
            *[
                generation.build_sql_sub_intent_prompt(intent)
                for intent in (
                    "specific_persons",
                    "stats",
                    "description",
                    "transcript_verbatim",
                    "transcript_qa",
                )
            ],
        ]

        for prompt in prompts:
            normalized = prompt.lower()
            self.assertNotIn("reformul", normalized)
            self.assertNotIn("commence par", normalized)

    def test_legacy_video_prefixed_intent_is_normalized(self) -> None:
        normalized = planner.normalize_planner_output(
            {
                "route": "rag",
                "sql_sub_intent": "video_transcript",
                "query_text": "Transcript",
                "query_text_bm25": "Transcript",
                "title_hint": None,
                "persons": [],
                "published_after": None,
                "published_before": None,
                "use_memory": False,
                "use_rag": True,
                "sql_main_source": True,
            }
        )

        self.assertEqual(normalized["sql_sub_intent"], "transcript_verbatim")

    def test_removed_agent_route_is_normalized_to_rag(self) -> None:
        normalized = planner.normalize_planner_output(
            {
                "route": "agent",
                "sql_sub_intent": None,
                "query_text": "Question complexe",
                "query_text_bm25": "Question complexe",
                "title_hint": None,
                "persons": [],
                "published_after": None,
                "published_before": None,
            }
        )

        self.assertEqual(normalized["route"], "rag")

    def test_explicit_title_is_used_directly_for_rag_prefilter(self) -> None:
        query = ExecutionPlan(
            raw_question="Que dit la vidéo ?",
            query_text="contenu",
            query_text_bm25="contenu",
            title_hint="Titre exact",
        )
        clauses, params = retrieval.build_prefilter_conditions(query)
        self.assertEqual(clauses, [retrieval.TITLE_CONTAINS_SQL])
        self.assertIn("regexp_replace", clauses[0])
        self.assertIn("concat(chr(37)", clauses[0])
        self.assertEqual(params, ["Titre exact"])

    def test_bm25_search_is_limited_to_detail_chunks(self) -> None:
        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params):
                if "c.chunk_level = 'detail'" not in sql:
                    raise AssertionError("Le filtre detail est absent.")
                self.sql = sql
                self.params = params

            def fetchall(self):
                return [
                    (
                        10,
                        "Video",
                        "https://example.test/video",
                        None,
                        2,
                        "detail",
                        1,
                        "Contenu detail",
                        ["Alice"],
                        0.75,
                    )
                ]

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def cursor(self):
                return Cursor()

        query = ExecutionPlan(
            raw_question="Question",
            query_text="Question",
            query_text_bm25="Question",
        )

        with patch.object(retrieval, "connect_database", return_value=Connection()):
            chunks, _ = retrieval.fetch_bm25_chunks(query, candidate_chunk_ids=None)

        self.assertEqual(chunks[0]["chunk_level"], "detail")
        self.assertEqual(chunks[0]["chunk_parent_id"], 1)

    def test_detail_results_are_expanded_with_section_and_global_context(self) -> None:
        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params):
                if "section.id = detail.chunk_parent_id" not in sql:
                    raise AssertionError("La jointure vers la section est absente.")
                if "global_chunk.id = section.chunk_parent_id" not in sql:
                    raise AssertionError("La jointure vers le global est absente.")
                self.params = params

            def fetchall(self):
                return [
                    (
                        10,
                        20,
                        2,
                        "Resume de section",
                        30,
                        1,
                        "Resume global",
                    )
                ]

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def cursor(self):
                return Cursor()

        chunks = [
            {
                "chunk_id": 10,
                "chunk_level": "detail",
                "chunk_index": 7,
                "text": "Contenu detail",
            }
        ]
        with patch.object(retrieval, "connect_database", return_value=Connection()):
            expanded, trace = retrieval.expand_detail_context(chunks)

        self.assertEqual(
            expanded[0]["section_context"],
            {
                "chunk_id": 20,
                "chunk_index": 2,
                "text": "Resume de section",
            },
        )
        self.assertEqual(
            expanded[0]["global_context"],
            {
                "chunk_id": 30,
                "chunk_index": 1,
                "text": "Resume global",
            },
        )
        self.assertEqual(trace["expanded_count"], 1)

    def test_generation_context_includes_hierarchical_parents(self) -> None:
        text = generation.source_context_text(
            {
                "text": "Contenu detail",
                "section_context": {"text": "Resume de section"},
                "global_context": {"text": "Resume global"},
            }
        )

        self.assertIn("Contenu detail", text)
        self.assertIn("Resume de section", text)
        self.assertIn("Resume global", text)

    def test_public_routes_are_preserved(self) -> None:
        client = TestClient(app)

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()["paths"]),
            {
                "/",
                "/api/rag",
                "/api/video-thumbnails",
                "/health",
                "/styles.css",
                "/version",
            },
        )

    def test_request_schema_remains_available_from_app(self) -> None:
        payload = RagRequest(question="Bonjour")

        self.assertTrue(payload.useSql)
        self.assertTrue(payload.useRerank)
        self.assertEqual(payload.topK, 40)
        self.assertEqual(payload.finalK, 5)


if __name__ == "__main__":
    unittest.main()
