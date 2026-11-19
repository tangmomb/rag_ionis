from __future__ import annotations

import json
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest import mock

from interface.monitoring import quality


def span(name, span_id, *, parent_id=None, kind="CHAIN", trace_id="t1", **attributes):
    return {
        "name": name,
        "context": {"trace_id": trace_id, "span_id": span_id},
        "parent_id": parent_id,
        "span_kind": kind,
        "attributes": attributes,
    }


def rag_trace(trace_id="t1", *, with_generation=True):
    spans = [
        span(
            "request",
            f"{trace_id}-root",
            trace_id=trace_id,
            **{
                "input.value": json.dumps({"question": "Qui a parlé de Rust ?"}),
                "output.value": json.dumps({"answer": "Alice, dans la vidéo 3."}),
                "rag.answer_ready_ms": 3200,
            },
        ),
        span("generation", f"{trace_id}-gen", parent_id=f"{trace_id}-root", trace_id=trace_id),
        span("planner", f"{trace_id}-plan", parent_id=f"{trace_id}-root", trace_id=trace_id),
        span(
            "mistral-medium-latest",
            f"{trace_id}-plan-llm",
            parent_id=f"{trace_id}-plan",
            kind="LLM",
            trace_id=trace_id,
            **{
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.1.message.role": "user",
                "llm.input_messages.1.message.content": "planifie",
            },
        ),
    ]
    if with_generation:
        spans.append(
            span(
                "mistral-medium-latest",
                f"{trace_id}-gen-llm",
                parent_id=f"{trace_id}-gen",
                kind="LLM",
                trace_id=trace_id,
                **{
                    "llm.input_messages.0.message.role": "system",
                    "llm.input_messages.0.message.content": "consignes",
                    "llm.input_messages.1.message.role": "user",
                    "llm.input_messages.1.message.content": "[S1] Alice parle de Rust.",
                },
            )
        )
    return spans


class JudgeInputTest(unittest.TestCase):
    def test_context_is_the_user_message_sent_to_the_answer_model(self) -> None:
        item = quality.judge_input_from_trace(rag_trace())

        self.assertEqual(item.span_id, "t1-root")
        self.assertEqual(item.question, "Qui a parlé de Rust ?")
        self.assertEqual(item.answer, "Alice, dans la vidéo 3.")
        # Not the planner prompt, nor the system instructions.
        self.assertEqual(item.context, "[S1] Alice parle de Rust.")

    def test_answer_without_generation_or_sources_has_no_context(self) -> None:
        item = quality.judge_input_from_trace(rag_trace(with_generation=False))

        self.assertIsNone(item.context)
        self.assertIn("(aucune source)", quality.judge_user_prompt(item))

    def test_database_answer_is_judged_against_its_returned_sources(self) -> None:
        trace = rag_trace(with_generation=False)
        trace[0]["attributes"]["output.value"] = json.dumps(
            {
                "answer": "Voici la description : Interview de Nella.",
                "sources": [
                    {
                        "video_title": "Chef de projet en énergie",
                        "video_url": "https://youtu.be/x",
                        "video_description": "Interview de Nella.",
                        "transcript": None,
                    }
                ],
            }
        )

        item = quality.judge_input_from_trace(trace)

        self.assertIn("[S1]", item.context)
        self.assertIn("Description : Interview de Nella.", item.context)
        self.assertNotIn("Transcription", item.context)

    def test_trace_without_root_or_answer_is_skipped(self) -> None:
        self.assertIsNone(quality.judge_input_from_trace(rag_trace()[1:]))
        trace = rag_trace()
        trace[0]["attributes"]["output.value"] = json.dumps({"answer": ""})
        self.assertIsNone(quality.judge_input_from_trace(trace))


class JudgePayloadTest(unittest.TestCase):
    def test_labels_map_to_scores(self) -> None:
        result = quality.parse_judge_payload(
            '{"fidelite": "partiellement_fidele", "pertinence": "pertinente", "explication": " ok "}'
        )

        self.assertEqual(result.faithfulness_score, 0.5)
        self.assertEqual(result.relevance_score, 1.0)
        self.assertEqual(result.explanation, "ok")

    def test_label_outside_the_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quality.parse_judge_payload(
                '{"fidelite": "excellent", "pertinence": "pertinente", "explication": ""}'
            )


class JudgeTracesTest(unittest.TestCase):
    def test_traffic_and_canary_share_one_batch_and_keep_their_own_annotations(self) -> None:
        client = mock.Mock()
        client.spans.get_spans.side_effect = lambda **kwargs: rag_trace(kwargs["trace_ids"][0])
        results = {
            "2": quality.JudgeResult("non_fidele", "pertinente", "Vidéo 3 absente."),
            "0": quality.JudgeResult("fidele", "pertinente", "Réponse correcte."),
        }
        with mock.patch.object(quality, "judge_batch", return_value=results) as batch:
            summaries = quality.judge_traces(
                client, {"rag-ionis": ["t1", "t2", "t1"], "canary": ["t3"]},
                model="judge", deadline=123,
            )

        batch.assert_called_once()
        self.assertEqual([item.trace_id for item in batch.call_args.args[0].values()], ["t1", "t2", "t3"])
        self.assertEqual(batch.call_args.kwargs, {"model": "judge", "deadline": 123})
        self.assertEqual(summaries["rag-ionis"].faithfulness, [1.0])
        self.assertEqual(summaries["rag-ionis"].errors, 1)
        self.assertEqual(summaries["canary"].faithfulness, [0.0])
        self.assertEqual(len(summaries["canary"].flagged), 1)

        calls = client.spans.add_span_annotation.call_args_list
        self.assertEqual(
            [(call.kwargs["span_id"], call.kwargs["annotation_name"]) for call in calls],
            [("t1-root", "fidelite"), ("t1-root", "pertinence"),
             ("t3-root", "fidelite"), ("t3-root", "pertinence")],
        )
        for call in calls:
            self.assertEqual(call.kwargs["annotator_kind"], "LLM")
            self.assertEqual(call.kwargs["identifier"], quality.ANNOTATION_IDENTIFIER)

    def test_a_failing_trace_does_not_stop_the_others(self) -> None:
        client = mock.Mock()
        client.spans.get_spans.side_effect = [RuntimeError("quota"), rag_trace("t2")]
        with mock.patch.object(quality, "judge_batch", return_value={
            "0": quality.JudgeResult("fidele", "pertinente", "ok"),
        }):
            summary = quality.judge_traces(
                client, {"rag-ionis": ["t1", "t2"]}, model="j", deadline=123,
            )["rag-ionis"]

        self.assertEqual((summary.judged, summary.errors), (1, 1))


class JudgeBatchTest(unittest.TestCase):
    def test_submits_jsonl_and_reads_partial_results(self) -> None:
        items = {
            "0": quality.judge_input_from_trace(rag_trace("t1")),
            "1": quality.judge_input_from_trace(rag_trace("t2")),
        }
        content = json.dumps({"fidelite": "fidele", "pertinence": "pertinente", "explication": "OK"})
        rows = [
            {"custom_id": "1", "error": {"message": "failed"}},
            {"custom_id": "0", "response": {"status_code": 200, "body": {"choices": [
                {"finish_reason": "stop", "message": {"content": content}},
            ]}}},
        ]
        with (
            mock.patch("mistralai.client.Mistral") as factory,
            mock.patch.dict("os.environ", {"MISTRAL_API_KEY": "test"}),
            mock.patch.object(quality.time, "sleep"),
        ):
            mistral = factory.return_value.__enter__.return_value
            mistral.files.upload.return_value = SimpleNamespace(id="input")
            mistral.batch.jobs.create.return_value = SimpleNamespace(id="job", status="QUEUED")
            mistral.batch.jobs.get.return_value = SimpleNamespace(
                id="job", status="SUCCESS", output_file="output", error_file="errors"
            )
            mistral.files.download.return_value.read.return_value = "\n".join(
                json.dumps(row) for row in rows
            ).encode()
            results = quality.judge_batch(items, model="mistral-large-latest", deadline=float("inf"))

        upload = mistral.files.upload.call_args.kwargs
        self.assertEqual(upload["purpose"], "batch")
        self.assertEqual(upload["file"]["file_name"], "quality-judge.jsonl")
        requests = [json.loads(line) for line in upload["file"]["content"].splitlines()]
        self.assertEqual([row["custom_id"] for row in requests], ["0", "1"])
        for row, item in zip(requests, items.values()):
            self.assertEqual(row["body"]["messages"][1]["content"], quality.judge_user_prompt(item))
        mistral.batch.jobs.create.assert_called_once_with(
            input_files=["input"], endpoint="/v1/chat/completions", model="mistral-large-latest",
            timeout_hours=2, metadata={"purpose": "rag-ionis-quality"},
        )
        mistral.batch.jobs.get.assert_called_once_with(job_id="job")
        self.assertEqual(results, {"0": quality.JudgeResult("fidele", "pertinente", "OK")})
        self.assertEqual(
            mistral.files.delete.call_args_list,
            [mock.call(file_id=file_id) for file_id in ("input", "output", "errors")],
        )


class HelpersTest(unittest.TestCase):
    def test_paris_day_window_follows_daylight_saving(self) -> None:
        start, end = quality.paris_day_window(date(2026, 11, 17))

        self.assertEqual(start, datetime(2026, 11, 16, 23, 0, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 11, 17, 23, 0, tzinfo=timezone.utc))
        summer_start, _ = quality.paris_day_window(date(2026, 7, 1))
        self.assertEqual(summer_start.hour, 22)

    def test_percentile_uses_nearest_rank(self) -> None:
        values = [float(value) for value in range(1, 21)]

        self.assertEqual(quality.percentile(values, 50), 10.0)
        self.assertEqual(quality.percentile(values, 95), 19.0)
        self.assertIsNone(quality.percentile([], 50))

    def test_canary_rotates_through_all_questions(self) -> None:
        questions = [f"q{index}" for index in range(25)]
        seen = set()
        for offset in range(3):
            selection = quality.select_canary_questions(
                questions, 10, date.fromordinal(date(2026, 11, 18).toordinal() + offset)
            )
            self.assertEqual(len(set(selection)), 10)
            seen.update(selection)

        self.assertEqual(len(seen), 25)
        self.assertEqual(quality.select_canary_questions(questions[:3], 10, date.today()), questions[:3])

    def test_llm_cost_skips_unpriced_models(self) -> None:
        spans = [
            {"attributes": {"llm.model_name": "mistral-medium-latest", "llm.token_count.prompt": 1_000_000, "llm.token_count.completion": 0}},
            {"attributes": {"llm.model_name": "inconnu", "llm.token_count.prompt": 10, "llm.token_count.completion": 10}},
        ]

        total, unpriced = quality.estimated_cost_of_llm_spans(spans)

        self.assertAlmostEqual(total, 1.5)
        self.assertEqual(unpriced, 1)


class ReportMetricsTest(unittest.TestCase):
    def test_metrics_are_json_serializable_numbers(self) -> None:
        metrics = quality.report_metrics(
            day=date(2026, 11, 17),
            requests=2,
            errors=0,
            answer_ready_ms=[3000.0, 5000.0],
            traffic_cost_usd=0.0123456789,
            feedback=(1, 2),
            traffic_judge=quality.JudgeSummary(judged=2, faithfulness=[1.0, 0.5], relevance=[1.0, 1.0]),
            canary=quality.CanarySummary(asked=10, failures=["x"], cost_usd=0.05, answer_ready_ms=[3400.0]),
            canary_judge=quality.JudgeSummary(judged=9, faithfulness=[1.0] * 9, relevance=[1.0] * 9),
            judge_model="mistral-large-latest",
        )

        self.assertEqual(json.loads(json.dumps(metrics)), metrics)
        self.assertEqual(metrics["day"], "2026-11-17")
        self.assertEqual(metrics["traffic_faithfulness"], 0.75)
        self.assertEqual(metrics["traffic_cost_usd"], 0.012346)
        self.assertEqual((metrics["canary_asked"], metrics["canary_failures"]), (10, 1))
        self.assertEqual(metrics["feedback_negative"], 2)

    def test_metrics_without_canary_omit_canary_fields(self) -> None:
        metrics = quality.report_metrics(
            day=date(2026, 11, 17),
            requests=0,
            errors=0,
            answer_ready_ms=[],
            traffic_cost_usd=0.0,
            feedback=(0, 0),
            traffic_judge=quality.JudgeSummary(),
            canary=None,
            canary_judge=None,
            judge_model="j",
        )

        self.assertIsNone(metrics["traffic_faithfulness"])
        self.assertNotIn("canary_asked", metrics)


class ReportTest(unittest.TestCase):
    def report(self, **overrides):
        values = dict(
            day=date(2026, 11, 17),
            requests=4,
            errors=0,
            answer_ready_ms=[3000.0, 4000.0],
            traffic_cost_usd=0.02,
            feedback=(1, 0),
            traffic_judge=quality.JudgeSummary(judged=2, faithfulness=[1.0, 1.0], relevance=[1.0, 0.5]),
            canary=quality.CanarySummary(asked=10, cost_usd=0.07, answer_ready_ms=[3500.0]),
            canary_judge=quality.JudgeSummary(judged=10, faithfulness=[1.0] * 10, relevance=[1.0] * 10),
            budget_usd=1.0,
            min_faithfulness=0.8,
            max_canary_failures=2,
        )
        values.update(overrides)
        return quality.build_report(**values)

    def test_healthy_day_is_green(self) -> None:
        message, anomalies = self.report()

        self.assertEqual(anomalies, [])
        self.assertTrue(message.startswith("🟢"))
        self.assertIn("Canari : 10/10 réponses, fidélité 100%", message)

    def test_budget_errors_and_low_canary_quality_are_anomalies(self) -> None:
        message, anomalies = self.report(
            errors=1,
            traffic_cost_usd=1.5,
            canary=quality.CanarySummary(asked=10, failures=["a", "b", "c"]),
            canary_judge=quality.JudgeSummary(judged=7, faithfulness=[0.5] * 7, relevance=[1.0] * 7),
        )

        self.assertTrue(message.startswith("🔴"))
        self.assertEqual(len(anomalies), 4)

    def test_judge_failing_on_every_answer_is_an_anomaly(self) -> None:
        message, anomalies = self.report(
            traffic_judge=quality.JudgeSummary(errors=3),
            canary_judge=quality.JudgeSummary(errors=10),
        )

        self.assertIn("juge LLM indisponible", anomalies)
        self.assertIn("13 évaluation(s) du juge en échec", message)

    def test_report_without_canary_still_works(self) -> None:
        message, anomalies = self.report(canary=None, canary_judge=None)

        self.assertNotIn("Canari", message)
        self.assertEqual(anomalies, [])

    def test_full_canary_reports_how_many_answers_were_judged(self) -> None:
        message, anomalies = self.report(
            canary=quality.CanarySummary(asked=84, full_dataset=True),
            canary_judge=quality.JudgeSummary(judged=83, faithfulness=[1.0] * 83),
        )

        self.assertIn("83/84 notées", message)
        self.assertIn("canari incomplet (83/84 réponses notées)", anomalies)


class DailyScheduleTest(unittest.TestCase):
    def test_saturday_runs_the_entire_dataset_and_friday_only_judges_traffic(self) -> None:
        questions = [f"Question {index} ?" for index in range(84)]
        args = quality.build_parser().parse_args(["daily"])
        for day, expected in ((20, False), (21, True)):
            with (
                self.subTest(day=day),
                mock.patch.object(quality, "datetime") as clock,
                mock.patch.object(quality, "phoenix_client") as client,
                mock.patch.object(quality, "feedback_counts", return_value=(0, 0)),
                mock.patch.object(quality, "run_canary", return_value=quality.CanarySummary(asked=84)) as run_canary,
                mock.patch.object(quality, "judge_traces", return_value={
                    quality.TRAFFIC_PROJECT: quality.JudgeSummary(judged=12),
                    "rag-ionis-canary": quality.JudgeSummary(judged=84),
                }) as judge,
                mock.patch.object(quality.time, "sleep"),
                mock.patch("interface.backend.telemetry.configure_telemetry"),
                mock.patch("interface.backend.telemetry.shutdown_telemetry"),
                mock.patch("interface.backend.golden_recall.load_golden_cases", return_value=[
                    SimpleNamespace(question=question) for question in questions
                ]) as load,
                mock.patch.dict("os.environ", {
                    "GOLDEN_DATASET_URL": "https://example.test/dataset",
                    "PHOENIX_PROJECT_NAME": "rag-ionis-canary",
                }),
            ):
                clock.now.return_value = datetime(2026, 11, day, 5, tzinfo=quality.PARIS)
                clock.combine.side_effect = datetime.combine
                client.return_value.spans.get_spans.return_value = []
                result = quality.run_daily(args)

            judge.assert_called_once()
            self.assertEqual(result["metrics"]["traffic_judged"], 12)
            self.assertEqual("rag-ionis-canary" in judge.call_args.args[1], expected)
            if expected:
                run_canary.assert_called_once_with(questions, delay_seconds=quality.DEFAULT_CANARY_DELAY_SECONDS)
                self.assertTrue(result["metrics"]["canary_full_dataset"])
                self.assertEqual(result["metrics"]["canary_judged"], 84)
            else:
                load.assert_not_called()
                run_canary.assert_not_called()
                self.assertNotIn("canary_asked", result["metrics"])


if __name__ == "__main__":
    unittest.main()
