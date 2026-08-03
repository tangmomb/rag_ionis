from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from interface.backend import retrieval
from interface.backend.telemetry import TraceOperation
from interface.backend.utilities import format_sql_pretty


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, name: str, value: Any) -> None:
        self.attributes[name] = value


class TraceOperationTests(unittest.TestCase):
    def test_set_output_text_uses_plain_text_mime_type(self) -> None:
        span = _RecordingSpan()

        TraceOperation(span).set_output_text("SELECT c.id\nFROM chunks c")

        self.assertEqual(span.attributes["output.value"], "SELECT c.id\nFROM chunks c")
        self.assertEqual(span.attributes["output.mime_type"], "text/plain")

    def test_set_documents_uses_openinference_retrieval_shape(self) -> None:
        span = _RecordingSpan()

        TraceOperation(span).set_documents(
            "retrieval.documents",
            [{"chunk_id": 3, "text": "SQL result", "bm25_score": 1.25}],
        )

        self.assertEqual(span.attributes["retrieval.documents.0.document.id"], "3")
        self.assertEqual(
            span.attributes["retrieval.documents.0.document.content"],
            "SQL result",
        )
        self.assertEqual(
            span.attributes["retrieval.documents.0.document.score"],
            1.25,
        )

    def test_format_sql_pretty_places_major_clauses_on_separate_lines(self) -> None:
        formatted = format_sql_pretty(
            "SELECT c.id, c.content FROM chunks c JOIN videos v ON v.id = c.video_id "
            "WHERE c.id = ANY(%s) ORDER BY c.id ASC LIMIT %s"
        )

        self.assertIsNotNone(formatted)
        assert formatted is not None
        self.assertIn("\nFROM chunks c", formatted)
        self.assertIn("\nJOIN videos v", formatted)
        self.assertIn("\nWHERE c.id = ANY(%s)", formatted)
        self.assertIn("\nORDER BY c.id ASC", formatted)
        self.assertIn("\nLIMIT %s", formatted)

    def test_structured_lookup_formats_both_sql_queries_in_execution_order(self) -> None:
        recorded: list[dict[str, Any]] = []

        class FormattedSpan:
            def __init__(self, record: dict[str, Any]) -> None:
                self.record = record

            def set_output_text(self, value: str) -> None:
                self.record["output"] = value

        @contextmanager
        def record_trace(name: str, **kwargs):
            record = {"name": name, **kwargs}
            recorded.append(record)
            yield FormattedSpan(record)

        trace = {
            "sql": "SELECT transcript_enriched FROM transcripts WHERE video_id = %s",
            "params": [7],
            "persons_table": {
                "sql": "SELECT name FROM speakers WHERE id = %s",
                "params": [3],
            },
        }

        with patch.object(retrieval, "trace_operation", side_effect=record_trace):
            retrieval.trace_formatted_sql("rag.structured_sql", trace)

        self.assertEqual(
            [item["name"] for item in recorded],
            [
                "rag.structured_sql.persons_table.sql_formatted",
                "rag.structured_sql.transcript_enriched.sql_formatted",
            ],
        )
        self.assertEqual(recorded[0]["input_value"]["params"], [3])
        self.assertEqual(recorded[1]["input_value"]["params"], [7])



if __name__ == "__main__":
    unittest.main()
