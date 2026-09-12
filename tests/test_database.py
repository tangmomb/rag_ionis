from __future__ import annotations

import unittest
from unittest.mock import patch

from interface.backend import database


class _Cursor:
    def __init__(self) -> None:
        self.parameters = None

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def execute(self, _sql, parameters) -> None:
        self.parameters = parameters

    def fetchone(self):
        return (9,)


class _Connection:
    def __init__(self) -> None:
        self.cursor_instance = _Cursor()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def cursor(self):
        return self.cursor_instance

    def commit(self) -> None:
        return None


class DatabaseTests(unittest.TestCase):

    def test_history_reads_messages_for_the_conversation(self) -> None:
        class Cursor:
            def __init__(self) -> None:
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def execute(self, sql, parameters) -> None:
                self.calls.append((sql, parameters))

            def fetchall(self):
                return [("Question du topic", "Réponse du topic")]

        class Connection:
            def __init__(self) -> None:
                self.cursor_instance = Cursor()

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def cursor(self):
                return self.cursor_instance

        connection = Connection()
        with (
            patch.object(database, "ensure_chat_schema"),
            patch.object(database, "connect_database", return_value=connection),
        ):
            history, trace = database.fetch_conversation_history(3, limit=3)

        self.assertEqual(connection.cursor_instance.calls[0][1], (3, 3))
        self.assertNotIn("topic_id", connection.cursor_instance.calls[0][0])
        self.assertEqual(history[0]["text"], "Question du topic")
        self.assertFalse(trace["latest_topic_only"])

    def test_store_chat_message_strips_postgresql_nul_characters(self) -> None:
        connection = _Connection()
        with (
            patch.object(database, "ensure_chat_schema"),
            patch.object(database, "connect_database", return_value=connection),
            patch.object(database, "ensure_conversation", return_value=3),
        ):
            result = database.store_chat_message(
                3,
                "Question\x00",
                "Réponse\x00 corrigée",
                "trace\x00",
            )

        self.assertEqual(result, (3, 9))
        self.assertEqual(
            connection.cursor_instance.parameters,
            (3, "Question", "Réponse corrigée", "trace"),
        )

    def test_store_message_feedback_updates_only_answer_messages(self) -> None:
        connection = _Connection()
        connection.cursor_instance.rowcount = 1
        with (
            patch.object(database, "ensure_chat_schema"),
            patch.object(database, "connect_database", return_value=connection),
        ):
            stored = database.store_message_feedback(9, True)

        self.assertTrue(stored)
        self.assertEqual(connection.cursor_instance.parameters, (True, 9))


if __name__ == "__main__":
    unittest.main()
