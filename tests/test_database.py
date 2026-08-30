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
            (3, None, "Question", "Réponse corrigée", "trace"),
        )


if __name__ == "__main__":
    unittest.main()
