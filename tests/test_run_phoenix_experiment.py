from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from interface.backend.schemas import ChunkSource, RagResponse
from utils import run_phoenix_experiment


class RunPhoenixExperimentTests(unittest.TestCase):
    def test_default_experiment_name_has_no_timestamp(self) -> None:
        self.assertEqual(
            run_phoenix_experiment.default_experiment_name(),
            "rag-ionis",
        )

    def test_build_rag_request_reads_nested_question(self) -> None:
        settings = run_phoenix_experiment.RagExperimentSettings(
            question_key="payload.question",
            reformulation_model="gpt-5.6-terra",
            planner_model="gpt-5.6-luna",
            answer_model="answer-model",
            reformulation_prompt="Prompt reformulation",
            planner_prompt="Prompt planner",
            answer_prompt="Prompt reponse",
            embedding_model="embedding-model",
            rerank_model="rerank-model",
            use_rerank=False,
            top_k=12,
            final_k=4,
        )

        request = run_phoenix_experiment.build_rag_request(
            {"payload": {"question": "  Ma question ?  "}},
            settings,
        )

        self.assertEqual(request.question, "Ma question ?")
        self.assertEqual(request.reformulationModel, "gpt-5.6-terra")
        self.assertEqual(request.plannerModel, "gpt-5.6-luna")
        self.assertEqual(request.answerModel, "answer-model")
        self.assertEqual(request.reformulationPrompt, "Prompt reformulation")
        self.assertEqual(request.plannerPrompt, "Prompt planner")
        self.assertEqual(request.answerPrompt, "Prompt reponse")
        self.assertEqual(request.embeddingModel, "embedding-model")
        self.assertEqual(request.rerankModel, "rerank-model")
        self.assertFalse(request.useRerank)
        self.assertEqual(request.topK, 12)
        self.assertEqual(request.finalK, 4)

    def test_build_rag_request_rejects_missing_question(self) -> None:
        settings = run_phoenix_experiment.RagExperimentSettings()

        with self.assertRaisesRegex(ValueError, "Champs input disponibles: prompt"):
            run_phoenix_experiment.build_rag_request(
                {"prompt": "Question"},
                settings,
            )

    def test_task_returns_compact_serializable_output(self) -> None:
        response = RagResponse(
            conversation_id=8,
            message_id=13,
            answer="Une reponse.",
            action="answer",
            sources=[
                ChunkSource(
                    chunk_id=21,
                    video_title="Titre",
                    video_url="https://example.com/video",
                    chunk_index=2,
                    text="Contenu volontairement absent de la sortie compacte.",
                    cohere_relevance_score=0.91,
                )
            ],
            retrieval={
                "execution_plan": {
                    "route": "rag",
                    "sql_sub_intent": None,
                },
                "retrieval_mode": "prefilter+bm25+vector+rrf",
                "used_rerank": True,
                "answer_action": "answer",
                "telemetry": {"trace_id": "abc123"},
            },
        )
        task = run_phoenix_experiment.build_rag_task(
            run_phoenix_experiment.RagExperimentSettings(shadow_evaluation=True)
        )

        def run_with_shadow(_request, **kwargs):
            self.assertTrue(kwargs["shadow_evaluation_enabled_override"])
            self.assertEqual(
                kwargs["shadow_evaluation_model_override"],
                run_phoenix_experiment.DEFAULT_REFORMULATION_MODEL,
            )
            self.assertFalse(kwargs["correction_loop_enabled_override"])
            kwargs["shadow_evaluation_sink"].update(
                {
                    "status": "acceptable",
                    "reason": "Réponse étayée",
                }
            )
            return response

        with patch.object(run_phoenix_experiment, "rag", side_effect=run_with_shadow):
            output = task({"question": "Question"})

        self.assertEqual(output["answer"], "Une reponse.")
        self.assertEqual(output["trace_id"], "abc123")
        self.assertEqual(output["sources"][0]["chunk_id"], 21)
        self.assertNotIn("text", output["sources"][0])
        self.assertEqual(
            output["diagnostics"]["reformulation_provider"],
            None,
        )
        self.assertEqual(
            output["diagnostics"]["shadow_evaluation"]["status"],
            "acceptable",
        )

    def test_builtin_evaluators_capture_transport_quality_and_action(self) -> None:
        output = {
            "answer": "Reponse",
            "action": "clarify",
            "diagnostics": {
                "shadow_evaluation": {
                    "verdict": "acceptable",
                    "issue": "none",
                    "status": "acceptable",
                    "reason": "Réponse étayée",
                    "retrieval_quality": 0.85,
                    "answer_grounded": True,
                }
            },
        }

        self.assertTrue(run_phoenix_experiment.response_nonempty(output))
        self.assertEqual(
            run_phoenix_experiment.answer_action(output),
            {"label": "clarify"},
        )
        self.assertEqual(
            run_phoenix_experiment.answer_action_match(
                output,
                {"action": "clarify"},
            )[1],
            "match",
        )
        self.assertEqual(
            run_phoenix_experiment.shadow_status(output)["label"],
            "acceptable",
        )
        self.assertEqual(
            run_phoenix_experiment.shadow_verdict(output)["label"],
            "acceptable",
        )
        self.assertEqual(
            run_phoenix_experiment.shadow_issue(output)["label"],
            "none",
        )
        self.assertEqual(
            run_phoenix_experiment.shadow_grounded(output)[0],
            1.0,
        )
        self.assertEqual(
            run_phoenix_experiment.shadow_retrieval_quality(output)[0],
            0.85,
        )
        self.assertEqual(
            run_phoenix_experiment.shadow_verdict_match(
                output,
                {"shadow_verdict": "acceptable"},
            )[1],
            "match",
        )
        self.assertEqual(
            run_phoenix_experiment.shadow_issue_match(
                output,
                {"shadow_issue": "none"},
            )[1],
            "match",
        )

    def test_parser_enables_shadow_calibration_explicitly(self) -> None:
        args = run_phoenix_experiment.build_parser().parse_args(
            ["--dataset", "questions-rag", "--shadow-evaluation"]
        )

        self.assertTrue(args.shadow_evaluation)
        self.assertEqual(
            args.shadow_evaluation_model,
            run_phoenix_experiment.DEFAULT_REFORMULATION_MODEL,
        )
        self.assertEqual(args.llm_timeout, 60)
        self.assertEqual(args.llm_max_retries, 0)

    def test_parser_enables_bounded_correction_explicitly(self) -> None:
        args = run_phoenix_experiment.build_parser().parse_args(
            [
                "--dataset",
                "questions-rag",
                "--shadow-evaluation",
                "--correction-loop",
            ]
        )

        self.assertTrue(args.shadow_evaluation)
        self.assertTrue(args.correction_loop)

    def test_builtin_evaluators_handle_failed_task_output(self) -> None:
        self.assertFalse(run_phoenix_experiment.response_nonempty(None))
        self.assertEqual(
            run_phoenix_experiment.answer_action(None),
            {"label": "unknown"},
        )
        self.assertEqual(
            run_phoenix_experiment.answer_action_match(
                None,
                {"action": "answer"},
            )[1],
            "mismatch",
        )
        self.assertEqual(
            run_phoenix_experiment.shadow_status(None)["label"],
            "not_run",
        )

    def test_correction_evaluators_report_attempt_and_strategy(self) -> None:
        output = {
            "diagnostics": {
                "correction": {
                    "attempted": True,
                    "count": 1,
                    "succeeded": True,
                    "strategy": "regenerate_answer",
                    "issue": "unsupported_answer",
                }
            }
        }

        self.assertEqual(
            run_phoenix_experiment.correction_outcome(output)["label"],
            "succeeded",
        )
        self.assertEqual(
            run_phoenix_experiment.correction_count(output),
            (1.0, "attempted"),
        )
        output["diagnostics"]["shadow_evaluation"] = {
            "verdict": "acceptable"
        }
        self.assertEqual(
            run_phoenix_experiment.correction_effectiveness(output)["label"],
            "effective",
        )
        self.assertEqual(
            run_phoenix_experiment.correction_outcome(None)["label"],
            "not_attempted",
        )

    def test_parser_supports_single_example_dry_run(self) -> None:
        args = run_phoenix_experiment.build_parser().parse_args(
            ["--dataset", "questions-rag", "--dry-run"]
        )

        self.assertEqual(args.dataset, "questions-rag")
        self.assertEqual(args.dry_run, 1)

    def test_parser_accepts_fast_openai_service_tier(self) -> None:
        with patch.dict(os.environ, {"OPENAI_SERVICE_TIER": ""}, clear=False):
            args = run_phoenix_experiment.build_parser().parse_args(
                [
                    "--dataset",
                    "questions-rag",
                    "--openai-service-tier",
                    "fast",
                ]
            )

        self.assertEqual(args.openai_service_tier, "fast")

    def test_parser_opens_gui_when_dataset_is_omitted(self) -> None:
        args = run_phoenix_experiment.build_parser().parse_args([])

        self.assertIsNone(args.dataset)

    def test_main_uses_gui_selection_when_dataset_is_omitted(self) -> None:
        selection = run_phoenix_experiment.ExperimentSelection(
            dataset="cases_phoenix",
            experiment_name="rag-gui",
            openai_service_tier="fast",
        )
        with (
            patch.object(
                run_phoenix_experiment,
                "prompt_experiment_selection",
                return_value=selection,
            ) as prompt,
            patch.object(
                run_phoenix_experiment,
                "run",
                return_value={"url": "http://phoenix/experiment"},
            ) as run,
        ):
            exit_code = run_phoenix_experiment.main([])

        self.assertEqual(exit_code, 0)
        prompt.assert_called_once()
        args = run.call_args.args[0]
        self.assertEqual(args.dataset, "cases_phoenix")
        self.assertEqual(args.experiment_name, "rag-gui")
        self.assertEqual(args.reformulation_model, "mistral-medium-latest")
        self.assertEqual(args.planner_model, "mistral-medium-latest")
        self.assertEqual(args.answer_model, "mistral-medium-latest")
        self.assertEqual(args.openai_service_tier, "fast")

    def test_openai_service_tier_labels_cover_gui_choices(self) -> None:
        self.assertEqual(
            run_phoenix_experiment.openai_service_tier_option_label(None),
            "Configuration du projet",
        )
        self.assertEqual(
            run_phoenix_experiment.openai_service_tier_option_label("default"),
            "Standard",
        )
        self.assertEqual(
            run_phoenix_experiment.openai_service_tier_option_label("fast"),
            "Fast",
        )

    def test_model_choices_cover_all_supported_providers(self) -> None:
        self.assertEqual(
            run_phoenix_experiment.LLM_MODEL_OPTIONS,
            (
                ("OpenAI - Sol", "gpt-5.6-sol"),
                ("OpenAI - Terra", "gpt-5.6-terra"),
                ("OpenAI - Luna", "gpt-5.6-luna"),
                ("Mistral - Medium", "mistral-medium-latest"),
                ("Mistral - Small", "mistral-small-latest"),
                ("Mistral - Large", "mistral-large-latest"),
                (
                    "Google - Gemini 3.1 Flash-Lite",
                    "gemini-3.1-flash-lite",
                ),
                ("Google - Gemini 3.6 Flash", "gemini-3.6-flash"),
                (
                    "Google - Gemini 3.5 Flash-Lite",
                    "gemini-3.5-flash-lite",
                ),
            ),
        )

    def test_parser_accepts_models_from_other_providers(self) -> None:
        args = run_phoenix_experiment.build_parser().parse_args(
            [
                "--dataset",
                "questions-rag",
                "--reformulation-model",
                "mistral-small-latest",
                "--planner-model",
                "gemini-3.1-flash-lite",
                "--answer-model",
                "gemini-3.6-flash",
            ]
        )

        self.assertEqual(args.reformulation_model, "mistral-small-latest")
        self.assertEqual(args.planner_model, "gemini-3.1-flash-lite")
        self.assertEqual(args.answer_model, "gemini-3.6-flash")

    def test_experiment_name_contains_models_in_pipeline_order(self) -> None:
        settings = run_phoenix_experiment.RagExperimentSettings(
            reformulation_model="gpt-5.6-terra",
            planner_model="gpt-5.6-luna",
            answer_model="gpt-5.6-sol",
        )

        name = run_phoenix_experiment.experiment_name_with_models(
            "rag-v1",
            settings,
        )

        self.assertEqual(name, "rag-v1_terra_luna_sol")

    def test_model_suffix_is_not_added_twice(self) -> None:
        settings = run_phoenix_experiment.RagExperimentSettings()

        name = run_phoenix_experiment.experiment_name_with_models(
            "rag-v1_mistral-medium_mistral-medium_mistral-medium",
            settings,
        )

        self.assertEqual(
            name,
            "rag-v1_mistral-medium_mistral-medium_mistral-medium",
        )

    def test_experiment_name_distinguishes_external_providers(self) -> None:
        settings = run_phoenix_experiment.RagExperimentSettings(
            reformulation_model="mistral-medium-latest",
            planner_model="gemini-3.1-flash-lite",
            answer_model="gemini-3.6-flash",
        )

        name = run_phoenix_experiment.experiment_name_with_models(
            "rag-multi",
            settings,
        )

        self.assertEqual(
            name,
            "rag-multi_mistral-medium_gemini-3-1-flash-lite_gemini-3-6-flash",
        )


if __name__ == "__main__":
    unittest.main()
