from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from interface.backend import retrieval
from interface.backend.schemas import ExecutionPlan, RagRequest


def _chunk(chunk_id: int) -> dict:
    return {"chunk_id": chunk_id, "video_url": f"https://youtu.be/{chunk_id}", "text": f"Passage {chunk_id}"}


class RerankFallbackTests(unittest.TestCase):
    def _retrieve(self, rerank_side_effect) -> tuple[list, dict]:
        plan = ExecutionPlan(raw_question="Q ?", query_text="Q ?", query_text_bm25="Q", final_k=2)
        with ExitStack() as stack:
            stack.enter_context(patch.object(retrieval, "get_openai_client", return_value=None))
            stack.enter_context(patch.object(retrieval, "trace_formatted_sql"))
            stack.enter_context(
                patch.object(retrieval, "prefilter_candidate_chunk_ids", return_value=(None, {"applied": False, "general_question_only": True}))
            )
            stack.enter_context(
                patch.object(retrieval, "fetch_bm25_chunks", return_value=([_chunk(1), _chunk(2), _chunk(3)], {}))
            )
            stack.enter_context(patch.object(retrieval, "fetch_vector_chunks", return_value=([], {})))
            stack.enter_context(
                patch.object(retrieval, "expand_detail_context", side_effect=lambda chunks: (chunks, {}))
            )
            stack.enter_context(patch.object(retrieval, "rerank_chunks", side_effect=rerank_side_effect))
            return retrieval.retrieve_chunks(RagRequest(question="Q ?", useRerank=True), plan)

    def test_rerank_failure_keeps_rrf_order_instead_of_failing(self) -> None:
        chunks, trace = self._retrieve(RuntimeError("Echec du rerank Cohere (429)"))

        self.assertEqual([c["chunk_id"] for c in chunks], [1, 2])
        self.assertFalse(trace["used_rerank"])
        self.assertEqual(trace["rerank"]["reason"], "rerank_failed")
        self.assertIn("429", trace["rerank"]["error"])

    def test_successful_rerank_is_reported(self) -> None:
        chunks, trace = self._retrieve(lambda query, fused, limit, model: ([fused[2]], {"applied": True}))

        self.assertEqual([c["chunk_id"] for c in chunks], [3])
        self.assertTrue(trace["used_rerank"])


if __name__ == "__main__":
    unittest.main()
