from __future__ import annotations

import argparse
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from interface.backend.schemas import RagResponse
from utils import replay_phoenix_conversation as replay


def _context() -> replay.ReplayContext:
    return replay.ReplayContext(
        original_trace_id="original-trace",
        original_conversation_id=218,
        target_message_id=289,
        target_question="On lui a posé quelles questions ?",
        history=(
            replay.ReplayMessage(
                message_id=287,
                user_message="Je cherche la vidéo de Fadila",
                answer_message="La vidéo de Fadila est disponible.",
                trace_id="previous-trace",
            ),
        ),
    )


class ReplayPhoenixConversationTests(unittest.TestCase):
    def test_fetch_trace_request_prefers_complete_nested_request(self) -> None:
        full_request = {
            "question": "Question",
            "conversationId": 218,
            "topK": 12,
        }
        client = SimpleNamespace(
            spans=SimpleNamespace(
                get_spans=lambda **_: [
                    {
                        "parent_id": None,
                        "attributes": {
                            "session.id": "218",
                            "input.value": json.dumps(
                                {
                                    "question": "Question",
                                    "request": full_request,
                                }
                            ),
                        },
                    },
                    {"parent_id": "root", "attributes": {}},
                ]
            )
        )

        request, session_id = replay.fetch_trace_request(
            client,
            project_identifier="rag-ionis",
            trace_id="trace-1",
            timeout=10,
        )

        self.assertEqual(request, full_request)
        self.assertEqual(session_id, "218")

    def test_build_replay_request_supports_legacy_trace_keys(self) -> None:
        request = replay.build_replay_request(
            {
                "reformulation_model": "mistral-medium-latest",
                "planner_model": "mistral-medium-latest",
                "answer_model": "mistral-medium-latest",
                "embedding_model": "text-embedding-3-large",
                "rerank_model": "cohere-rerank",
                "use_rerank": False,
            },
            _context(),
            444,
        )

        self.assertEqual(request.question, "On lui a posé quelles questions ?")
        self.assertEqual(request.conversationId, 444)
        self.assertFalse(request.useRerank)
        self.assertEqual(request.rerankModel, "cohere-rerank")

    def test_seed_replay_conversation_copies_answers_without_trace_ids(self) -> None:
        class Cursor:
            def __init__(self) -> None:
                self.executed: list[tuple[str, object]] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, sql, params=None):
                self.executed.append((sql, params))

            def fetchone(self):
                return (444,)

        class Connection:
            def __init__(self, cursor: Cursor) -> None:
                self._cursor = cursor
                self.committed = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def cursor(self):
                return self._cursor

            def commit(self):
                self.committed = True

        cursor = Cursor()
        connection = Connection(cursor)
        with (
            patch.object(replay, "ensure_chat_schema"),
            patch.object(replay, "connect_database", return_value=connection),
        ):
            conversation_id = replay.seed_replay_conversation(_context())

        self.assertEqual(conversation_id, 444)
        self.assertTrue(connection.committed)
        self.assertEqual(
            cursor.executed[1][1],
            (444, "Je cherche la vidéo de Fadila", "La vidéo de Fadila est disponible."),
        )
        self.assertIn("NULL", cursor.executed[1][0])

    def test_run_replay_creates_visible_parent_and_seed_spans(self) -> None:
        context = _context()
        traced_request = {
            "question": context.target_question,
            "conversationId": 218,
        }
        response = RagResponse(
            conversation_id=444,
            message_id=500,
            answer="Les questions posées à Fadila sont...",
            action="answer",
            sources=[],
            retrieval={},
        )
        recorded_names: list[str] = []
        recorded_sessions: list[int] = []
        recorded_outputs: dict[str, object] = {}

        @contextmanager
        def record_trace(name: str, **kwargs):
            del kwargs
            recorded_names.append(name)
            span = SimpleNamespace(
                set_session_id=lambda value: recorded_sessions.append(value),
                set_output=lambda value: recorded_outputs.__setitem__(name, value),
            )
            yield span

        args = argparse.Namespace(
            trace_id=context.original_trace_id,
            mode="exact-context",
            phoenix_base_url="http://localhost:6006",
            project="rag-ionis",
            timeout=10,
            dry_run=False,
        )
        with (
            patch("phoenix.client.Client", return_value=object()),
            patch.object(
                replay,
                "fetch_trace_request",
                return_value=(traced_request, "218"),
            ),
            patch.object(replay, "load_replay_context", return_value=context),
            patch.object(replay, "seed_replay_conversation", return_value=444),
            patch.object(replay, "configure_telemetry"),
            patch.object(replay, "shutdown_telemetry") as shutdown,
            patch.object(replay, "trace_operation", side_effect=record_trace),
            patch.object(replay, "current_trace_id", return_value="replay-trace"),
            patch.object(replay, "run_rag", return_value=response) as run_rag,
        ):
            result = replay.run_replay(args)

        self.assertEqual(
            recorded_names,
            ["replay", "replay.seed_history"],
        )
        self.assertIn(444, recorded_sessions)
        self.assertEqual(result["replay_trace_id"], "replay-trace")
        self.assertEqual(result["replay_conversation_id"], 444)
        self.assertEqual(
            recorded_outputs["replay.seed_history"]["seeded_message_count"],
            1,
        )
        self.assertEqual(run_rag.call_args.args[0].conversationId, 444)
        shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
