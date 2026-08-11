from __future__ import annotations

import unittest
from types import SimpleNamespace
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
    def test_answer_video_url_selects_source_without_marker(self) -> None:
        sources = [
            {"video_url": "https://www.youtube.com/watch?v=video-1"},
            {"video_url": "https://www.youtube.com/watch?v=video-2"},
        ]

        answer, selected_sources = generation.select_answer_sources(
            "Voir https://www.youtube.com/watch?v=video-2 pour plus de détails.",
            sources,
        )

        self.assertEqual(
            answer,
            "Voir https://www.youtube.com/watch?v=video-2 pour plus de détails.",
        )
        self.assertEqual(selected_sources, [sources[1]])

    def test_answer_marker_and_video_url_select_source_once(self) -> None:
        source = {"video_url": "https://www.youtube.com/watch?v=video-1"}

        answer, selected_sources = generation.select_answer_sources(
            "La réponse est ici [S1] : https://www.youtube.com/watch?v=video-1",
            [source],
        )

        self.assertEqual(
            answer,
            "La réponse est ici : https://www.youtube.com/watch?v=video-1",
        )
        self.assertEqual(selected_sources, [source])

    def test_all_llm_steps_use_mistral_medium_by_default(self) -> None:
        self.assertEqual(DEFAULT_PLANNER_MODEL, "mistral-medium-latest")
        self.assertEqual(DEFAULT_REFORMULATION_MODEL, "mistral-medium-latest")
        self.assertEqual(DEFAULT_GENERATION_MODEL, "mistral-medium-latest")

    def test_all_structured_llm_steps_define_strict_schemas(self) -> None:
        from interface.backend.analytics_sql import ANALYTICS_SQL_RESPONSE_SCHEMA
        from interface.backend.answer_judge import ANSWER_JUDGE_RESPONSE_SCHEMA

        schemas = (
            planner.REFORMULATION_RESPONSE_SCHEMA,
            planner.PLANNER_RESPONSE_SCHEMA,
            ANALYTICS_SQL_RESPONSE_SCHEMA,
            generation.ANSWER_RESPONSE_SCHEMA,
            ANSWER_JUDGE_RESPONSE_SCHEMA,
        )
        for schema in schemas:
            self.assertEqual(schema["type"], "object")
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(
                set(schema["required"]),
                set(schema["properties"]),
            )

    def test_interface_fixes_all_llm_steps_to_mistral_medium(self) -> None:
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertNotIn('id="reformulationModel"', html)
        self.assertNotIn('id="plannerModel"', html)
        self.assertNotIn('id="answerModel"', html)
        self.assertNotIn('fetch("/api/llm-models")', html)
        self.assertEqual(html.count('"mistral-medium-latest"'), 3)

    def test_reformulation_prompt_has_one_narrow_responsibility(self) -> None:
        system_prompt, user_prompt = planner.build_question_reformulation_prompt(
            "Et pour elle ?",
            [
                {"role": "user", "text": "Que dit Alice Martin ?"},
                {"role": "assistant", "text": "Alice présente son parcours."},
            ],
        )

        self.assertLess(len(system_prompt), 1800)
        self.assertIn("besoin de l'historique", system_prompt)
        self.assertIn("peut n'avoir aucun rapport avec l'historique précédent", system_prompt)
        self.assertIn("changer de sujet", system_prompt)
        self.assertIn("Sois le plus simple et concis possible", system_prompt)
        self.assertIn("follow_up", system_prompt)
        self.assertIn("reformulated_question", system_prompt)
        self.assertIn("échange le plus récent", system_prompt)
        self.assertIn("conserve-les tous", system_prompt)
        self.assertIn("texte normal, sans Markdown", system_prompt)
        self.assertIn("Règle absolue d'autonomie", system_prompt)
        self.assertIn("entièrement compréhensible", system_prompt)
        self.assertIn("la première vidéo mentionnée", system_prompt)
        self.assertIn("copie le titre exact", system_prompt)
        self.assertIn("Test obligatoire avant de répondre", system_prompt)
        self.assertIn("Que dit Alice dans la vidéo « Vidéo A » ?", system_prompt)
        self.assertIn("du plus vieux au plus récent", system_prompt)
        self.assertIn("dernier bloc user/assistant", system_prompt)
        self.assertIn(
            "Historique récent (du plus vieux au plus récent ; le dernier bloc est "
            "prioritaire) :\n\nuser: Que dit Alice Martin ?",
            user_prompt,
        )
        self.assertIn(
            "Que dit Alice Martin ?\n\nassistant: Alice présente son parcours.",
            user_prompt,
        )
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
        self.assertIn("uniquement si le mot exact 'description'", system_prompt)
        self.assertIn("texte normal, sans Markdown", system_prompt)

    def test_planner_prompt_limits_direct_route_to_greetings_and_politeness(self) -> None:
        system_prompt, _ = planner.build_planner_prompt("Bonjour")

        self.assertIn(
            "route='direct' si la question ou le message est une salutation ou une formule de politesse",
            system_prompt,
        )
        self.assertNotIn(
            "ne demande rien à propos de la base de données",
            system_prompt,
        )

    def test_planner_prompt_preserves_utf8_french_text(self) -> None:
        system_prompt, _ = planner.build_planner_prompt("Question")

        self.assertIn("Première étape", system_prompt)
        self.assertIn("clés", system_prompt)
        self.assertNotIn("Ã", system_prompt)

    def test_true_social_message_keeps_direct_route(self) -> None:
        plan = PlannerPlan(route="direct", query_text="Bonjour")

        correction = planner.apply_deterministic_sql_policy("Bonjour", plan)

        self.assertIsNone(correction)
        self.assertEqual(plan.route, "direct")
        self.assertFalse(plan.use_rag)

    def test_non_social_direct_route_is_repaired_to_rag(self) -> None:
        question = "Explique-moi les RAG."
        plan = PlannerPlan(route="direct", query_text=question)

        correction = planner.apply_deterministic_sql_policy(question, plan)

        self.assertEqual(correction, "direct_non_social_to_rag")
        self.assertEqual(plan.route, "rag")
        self.assertTrue(plan.use_rag)

    def test_direct_route_with_company_forces_specific_persons_sql(self) -> None:
        question = "Trouve une personne qui travaille chez Microsoft."
        plan = PlannerPlan(
            route="direct",
            query_text=question,
            companies=["Microsoft"],
        )

        correction = planner.apply_deterministic_sql_policy(question, plan)

        self.assertEqual(correction, "direct_with_entities_to_rag")
        self.assertEqual(plan.route, "rag")
        self.assertEqual(plan.sql_sub_intent, "specific_persons")
        self.assertTrue(plan.sql_main_source)

    def test_greeting_with_information_request_is_not_social(self) -> None:
        self.assertTrue(planner.is_social_message("Bonjour !"))
        self.assertFalse(
            planner.is_social_message("Bonjour, trouve une personne chez Microsoft")
        )

    def test_obvious_elliptical_message_repairs_follow_up_flag(self) -> None:
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    output_text=(
                        '{"follow_up": false, "reformulated_question": '
                        '"et une personne qui travaille chez Microsoft"}'
                    )
                )
            )
        )
        memory = [
            {"role": "user", "text": "Je cherche un étudiant qui fait des RAG"},
            {"role": "assistant", "text": "Tom Baucher fait des RAG."},
        ]
        with patch.object(
            planner,
            "fetch_conversation_memory",
            return_value=(memory, {"applied": True, "message_count": 2}),
        ):
            reformulated, trace = planner.reformulate_question(
                "et une qui bosse chez Microsoft",
                184,
                client,
            )

        self.assertEqual(
            reformulated,
            "et une personne qui travaille chez Microsoft",
        )
        self.assertTrue(trace["follow_up"])
        self.assertEqual(trace["reason"], "deterministic_follow_up_detected")

    def test_reformulation_excludes_old_emric_context_from_yassin_comparison(self) -> None:
        calls: list[dict] = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=(
                    '{"follow_up": true, "reformulated_question": '
                    '"Quels sont les points communs entre Yassin, Fadila Ouro Sama et Hugo Gérardin ?"}'
                )
            )

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        memory = [
            {"role": "user", "text": "Qui est Emric ?"},
            {"role": "assistant", "text": "Emric est alternant chez Novares."},
            {"role": "user", "text": "Elle a fait combien de vues ?"},
            {"role": "assistant", "text": "La vidéo de Fadila a 552 vues."},
            {"role": "user", "text": "Quel entretien a le plus de vues ?"},
            {"role": "assistant", "text": "La vidéo de Hugo Gérardin a 21 136 vues."},
            {"role": "user", "text": "Compare leurs deux vidéos."},
            {
                "role": "assistant",
                "text": "Comparaison des vidéos de Fadila Ouro Sama et Hugo Gérardin.",
            },
        ]
        with patch.object(
            planner,
            "fetch_conversation_memory",
            return_value=(memory, {"applied": True, "message_count": len(memory)}),
        ) as fetch_memory:
            reformulated, trace = planner.reformulate_question(
                "Des points communs avec Yassin ?",
                211,
                client,
            )

        fetch_memory.assert_called_once_with(
            211,
            limit=planner.REFORMULATION_MEMORY_EXCHANGES,
        )
        reformulation_prompt = calls[0]["input"][1]["content"]
        self.assertNotIn("Emric", reformulation_prompt)
        self.assertIn("Fadila Ouro Sama", reformulation_prompt)
        self.assertIn("Hugo Gérardin", reformulation_prompt)
        self.assertEqual(trace["memory_message_count"], 6)
        self.assertEqual(
            reformulated,
            "Quels sont les points communs entre Yassin, Fadila Ouro Sama et Hugo Gérardin ?",
        )

    def test_video_ranking_follow_up_prioritizes_latest_comparison_exchange(self) -> None:
        calls: list[dict] = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=(
                    '{"follow_up": true, "reformulated_question": '
                    '"Laquelle des vidéos de Loucif Ouyahia et Sophie Ollivier a le plus de vues ?"}'
                )
            )

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        memory = [
            {"role": "user", "text": "Je cherche la vidéo de Sophie"},
            {
                "role": "assistant",
                "text": "Deux vidéos présentent Sophie Vanderpol.",
            },
            {
                "role": "user",
                "text": "Des points communs entre Loucif et Sophie Ollivier ?",
            },
            {
                "role": "assistant",
                "text": "Loucif Ouyahia et Sophie Ollivier travaillent dans la pharmacie.",
            },
        ]
        with patch.object(
            planner,
            "fetch_conversation_memory",
            return_value=(memory, {"applied": True, "message_count": len(memory)}),
        ):
            reformulated, trace = planner.reformulate_question(
                "Laquelle des 2 a le plus de vues ?",
                212,
                client,
            )

        reformulation_prompt = calls[0]["input"][1]["content"]
        self.assertNotIn("Sophie Vanderpol", reformulation_prompt)
        self.assertIn("Loucif", reformulation_prompt)
        self.assertIn("Sophie Ollivier", reformulation_prompt)
        self.assertEqual(trace["memory"]["prompt_message_count"], 2)
        self.assertEqual(
            reformulated,
            "Laquelle des vidéos de Loucif Ouyahia et Sophie Ollivier a le plus de vues ?",
        )

    def test_video_comparison_keeps_two_exchanges_when_each_introduces_one_person(self) -> None:
        memory = [
            {"role": "user", "text": "Qui est Emric ?"},
            {"role": "assistant", "text": "Emric est alternant."},
            {"role": "user", "text": "Elle a combien de vues ?"},
            {"role": "assistant", "text": "La vidéo de Fadila a 552 vues."},
            {"role": "user", "text": "Quel entretien a le plus de vues ?"},
            {"role": "assistant", "text": "La vidéo de Hugo a 21 136 vues."},
        ]

        selected = planner.select_reformulation_memory(
            "Compare leurs deux vidéos",
            memory,
        )

        selected_text = "\n".join(item["text"] for item in selected)
        self.assertNotIn("Emric", selected_text)
        self.assertIn("Fadila", selected_text)
        self.assertIn("Hugo", selected_text)
        self.assertEqual(len(selected), 4)

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

    def test_specific_persons_prompt_answers_from_video_sources(self) -> None:
        prompt = generation.build_sql_sub_intent_prompt("specific_persons")

        self.assertIn("réponds directement à la question", prompt)
        self.assertIn("sources vidéo structurées", prompt)
        self.assertIn("Ne réduis pas la réponse à une liste de vidéos", prompt)
        self.assertIn("titre ou le lien", prompt)
        self.assertNotIn("Présente chaque vidéo trouvée", prompt)

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
                "analytics",
                "description",
                "transcript_verbatim",
                "transcript_qa",
            )
        }

        self.assertEqual(len(set(prompts.values())), len(prompts))

    def test_sql_sub_intent_name_is_not_exposed_in_final_user_prompt(self) -> None:
        calls: list[dict] = []

        class Responses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    output_text='{"answer":"Réponse","action":"answer"}',
                )

        generation.generate_sql_answer(
            SimpleNamespace(responses=Responses()),
            "Donne-moi les chiffres de cette vidéo.",
            "mistral-medium-latest",
            "analytics",
            [
                {
                    "video_title": "Vidéo test",
                    "video_url": "https://example.test/video",
                    "text": "Vues: 42",
                }
            ],
        )

        messages = calls[0]["input"]
        self.assertIn("demande analytique", messages[0]["content"])
        self.assertNotIn("Sous-route SQL", messages[1]["content"])
        self.assertNotIn("sql_sub_intent", messages[1]["content"])
        self.assertIn("Sources pour répondre :", messages[1]["content"])
        self.assertIn("Source 1 :", messages[1]["content"])
        self.assertNotIn("Resultat 1", messages[1]["content"])

        generation.generate_multi_source_answer(
            SimpleNamespace(responses=Responses()),
            "Compare ces personnes.",
            "mistral-medium-latest",
            "multi_source",
            [],
            [
                {
                    "video_title": "Vidéo test",
                    "video_url": "https://example.test/video",
                    "chunk_index": 1,
                    "text": "Information comparative",
                }
            ],
        )

        multi_source_user_prompt = calls[1]["input"][1]["content"]
        self.assertNotIn("Route planifiee", multi_source_user_prompt)
        self.assertNotIn("multi_source", multi_source_user_prompt)
        self.assertIn("Sources pour répondre :", multi_source_user_prompt)

    def test_answer_prompts_do_not_require_question_reformulation(self) -> None:
        prompts = [
            generation.FINAL_ANSWER_STYLE,
            *[
                generation.build_sql_sub_intent_prompt(intent)
                for intent in (
                    "specific_persons",
                    "analytics",
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

    def test_interface_uses_sanitized_markdown_renderer(self) -> None:
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn("markdown-it@15.0.0", response.text)
        self.assertIn("dompurify@3.4.13", response.text)
        self.assertIn("html: false", response.text)
        self.assertIn("window.DOMPurify.sanitize", response.text)
        self.assertNotIn("function renderInlineMarkdown", response.text)

    def test_public_routes_are_preserved(self) -> None:
        client = TestClient(app)

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()["paths"]),
            {
                "/",
                "/api/llm-models",
                "/api/rag",
                "/api/video-thumbnails",
                "/health",
                "/styles.css",
                "/version",
            },
        )

    def test_llm_model_catalog_exposes_only_mistral(self) -> None:
        response = TestClient(app).get("/api/llm-models")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(
            data["defaults"],
            {
                "reformulationModel": "mistral-medium-latest",
                "plannerModel": "mistral-medium-latest",
                "answerModel": "mistral-medium-latest",
            },
        )
        self.assertIn(
            "mistral-medium-latest",
            {model["id"] for model in data["models"]},
        )
        self.assertEqual(
            {model["provider"] for model in data["models"]},
            {"mistral"},
        )

    def test_public_rag_endpoint_rejects_non_mistral_models(self) -> None:
        response = TestClient(app).post(
            "/api/rag",
            json={
                "question": "Bonjour",
                "reformulationModel": "gpt-5.6-sol",
                "plannerModel": "mistral-medium-latest",
                "answerModel": "mistral-medium-latest",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("uniquement Mistral", response.json()["detail"])
        self.assertIn("run_phoenix_experiment.py", response.json()["detail"])

    def test_request_schema_remains_available_from_app(self) -> None:
        payload = RagRequest(question="Bonjour")

        self.assertTrue(payload.useSql)
        self.assertTrue(payload.useRerank)
        self.assertEqual(payload.topK, 40)
        self.assertEqual(payload.finalK, 5)


if __name__ == "__main__":
    unittest.main()
