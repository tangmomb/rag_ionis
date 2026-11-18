from __future__ import annotations

import json
import unittest
from datetime import date, datetime, timezone
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
    def test_scores_are_published_as_idempotent_llm_annotations(self) -> None:
        client = mock.Mock()
        client.spans.get_spans.side_effect = lambda **kwargs: rag_trace(kwargs["trace_ids"][0])
        judge = mock.Mock(
            return_value=quality.JudgeResult("non_fidele", "pertinente", "Vidéo 3 absente.")
        )

        summary = quality.judge_traces(client, "rag-ionis", ["t1"], model="judge", judge=judge)

        self.assertEqual(summary.judged, 1)
        self.assertEqual(summary.faithfulness, [0.0])
        self.assertEqual(len(summary.flagged), 1)
        calls = client.spans.add_span_annotation.call_args_list
        self.assertEqual(
            [call.kwargs["annotation_name"] for call in calls], ["fidelite", "pertinence"]
        )
        for call in calls:
            self.assertEqual(call.kwargs["span_id"], "t1-root")
            self.assertEqual(call.kwargs["annotator_kind"], "LLM")
            self.assertEqual(call.kwargs["identifier"], quality.ANNOTATION_IDENTIFIER)

    def test_a_failing_trace_does_not_stop_the_others(self) -> None:
        client = mock.Mock()
        client.spans.get_spans.side_effect = lambda **kwargs: rag_trace(kwargs["trace_ids"][0])
        judge = mock.Mock(
            side_effect=[
                RuntimeError("quota"),
                quality.JudgeResult("fidele", "pertinente", "ok"),
            ]
        )

        summary = quality.judge_traces(client, "rag-ionis", ["t1", "t2"], model="j", judge=judge)

        self.assertEqual((summary.judged, summary.errors), (1, 1))


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


if __name__ == "__main__":
    unittest.main()
