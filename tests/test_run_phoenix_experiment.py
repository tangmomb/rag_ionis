from __future__ import annotations

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
                "source_evaluation": {"reason": "sources_available"},
                "telemetry": {"trace_id": "abc123"},
            },
        )
        task = run_phoenix_experiment.build_rag_task(
            run_phoenix_experiment.RagExperimentSettings()
        )

        with patch.object(run_phoenix_experiment, "rag", return_value=response):
            output = task({"question": "Question"})

        self.assertEqual(output["answer"], "Une reponse.")
        self.assertEqual(output["trace_id"], "abc123")
        self.assertEqual(output["sources"][0]["chunk_id"], 21)
        self.assertNotIn("text", output["sources"][0])

    def test_builtin_evaluators_capture_transport_quality_and_action(self) -> None:
        output = {"answer": "Reponse", "action": "clarify"}

        self.assertTrue(run_phoenix_experiment.response_nonempty(output))
        self.assertEqual(
            run_phoenix_experiment.answer_action(output),
            {"label": "clarify"},
        )

    def test_parser_supports_single_example_dry_run(self) -> None:
        args = run_phoenix_experiment.build_parser().parse_args(
            ["--dataset", "questions-rag", "--dry-run"]
        )

        self.assertEqual(args.dataset, "questions-rag")
        self.assertEqual(args.dry_run, 1)

    def test_parser_opens_gui_when_dataset_is_omitted(self) -> None:
        args = run_phoenix_experiment.build_parser().parse_args([])

        self.assertIsNone(args.dataset)

    def test_main_uses_gui_selection_when_dataset_is_omitted(self) -> None:
        selection = run_phoenix_experiment.ExperimentSelection(
            dataset="cases_phoenix",
            experiment_name="rag-gui",
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
        self.assertEqual(args.reformulation_model, "gpt-5.6-sol")
        self.assertEqual(args.planner_model, "gpt-5.6-sol")
        self.assertEqual(args.answer_model, "gpt-5.6-sol")

    def test_model_choices_match_requested_sol_terra_luna_order(self) -> None:
        self.assertEqual(
            run_phoenix_experiment.LLM_MODEL_OPTIONS,
            (
                ("1 - sol", "gpt-5.6-sol"),
                ("2 - terra", "gpt-5.6-terra"),
                ("3 - luna", "gpt-5.6-luna"),
            ),
        )

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
            "rag-v1_sol_sol_sol",
            settings,
        )

        self.assertEqual(name, "rag-v1_sol_sol_sol")


if __name__ == "__main__":
    unittest.main()
