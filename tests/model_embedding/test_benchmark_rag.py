import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.model_embedding.benchmark import EvalCase
from tests.model_embedding.benchmark_rag import (
    RagSettings,
    reciprocal_rank_fusion,
    rerank_results,
    stage_record,
    summarize_records,
    write_rag_report,
)


def result(chunk_id, key, text="texte"):
    return {"chunk_id": chunk_id, "key": key, "text": text}


class FakeCohereClient:
    def __init__(self):
        self.calls = 0

    def rerank(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=1, relevance_score=0.9),
                SimpleNamespace(index=0, relevance_score=0.5),
            ]
        )


class RagBenchmarkTests(unittest.TestCase):
    def test_rrf_rewards_chunk_present_in_both_rankings(self):
        lexical = [result(1, "v:1"), result(2, "v:2")]
        vector = [result(3, "v:3"), result(1, "v:1")]
        fused, _ = reciprocal_rank_fusion(lexical, vector, limit=3, rank_constant=60)
        self.assertEqual(fused[0]["key"], "v:1")
        self.assertEqual(fused[0]["rank_sources"], {"bm25": 1, "vector": 2})

    def test_rerank_cache_avoids_second_api_call(self):
        client = FakeCohereClient()
        candidates = [result(1, "v:1", "un"), result(2, "v:2", "deux")]
        cache = {}
        first, _, cached_first = rerank_results(
            client, "question", candidates, "model", 2, cache, "q1"
        )
        second, _, cached_second = rerank_results(
            client, "question", candidates, "model", 2, cache, "q1"
        )
        self.assertFalse(cached_first)
        self.assertTrue(cached_second)
        self.assertEqual(client.calls, 1)
        self.assertEqual([item["key"] for item in first], ["v:2", "v:1"])
        self.assertEqual([item["key"] for item in second], ["v:2", "v:1"])

    def test_summary_marks_recall_beyond_stage_limit_unavailable(self):
        case = EvalCase("q1", "question", ("v:1",), "contenu_courant")
        record = stage_record(
            "large-2000",
            "rerank",
            case,
            [result(1, "v:1")],
            result_limit=5,
            elapsed_ms=10,
        )
        summary = summarize_records([record], (1, 5, 10))[0]
        self.assertEqual(summary["recall_at_1"], 1.0)
        self.assertEqual(summary["recall_at_5"], 1.0)
        self.assertIsNone(summary["recall_at_10"])

    def test_report_contains_all_hybrid_stages(self):
        settings = RagSettings(
            configuration_names=("large-2000", "large-3072"),
            lexical_limit=40,
            vector_limit=40,
            rrf_limit=30,
            rerank_limit=5,
            rrf_rank_constant=60,
            rerank_model="rerank-v4.0-fast",
            rerank_min_interval_seconds=6.2,
            top_k=(1, 5, 10),
        )
        case = EvalCase("q1", "question", ("v:1",), "contenu_courant")
        records = []
        for configuration, stage, limit in (
            ("shared", "lexical", 40),
            ("large-2000", "vector", 40),
            ("large-2000", "rrf", 30),
            ("large-2000", "rerank", 5),
            ("large-3072", "vector", 40),
            ("large-3072", "rrf", 30),
            ("large-3072", "rerank", 5),
        ):
            records.append(
                stage_record(
                    configuration,
                    stage,
                    case,
                    [result(1, "v:1")],
                    limit,
                    1,
                )
            )
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "rag_report.md"
            write_rag_report(path, settings, [case], records, 0, 2)
            report = path.read_text(encoding="utf-8")
        self.assertIn("Recherche lexicale PostgreSQL", report)
        self.assertIn("RRF", report)
        self.assertIn("rerank", report)
        self.assertIn("large-2000", report)
        self.assertIn("large-3072", report)


if __name__ == "__main__":
    unittest.main()
