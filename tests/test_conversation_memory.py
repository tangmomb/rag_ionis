from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from interface.backend import conversation_memory


class ConversationMemoryTests(unittest.TestCase):
    def test_memory_is_optional_without_a_conversation(self) -> None:
        memory = conversation_memory.load_reformulation_memory(None, "Et la deuxième ?")

        self.assertFalse(memory["available"])
        self.assertEqual(memory["reason"], "no_conversation_id")

    def test_closed_topic_is_summarized_and_saved_in_json(self) -> None:
        written: dict[str, object] = {}

        class Cursor:
            rowcount = 1

            def execute(self, _query, parameters) -> None:
                written["memory"] = parameters[0]

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

        class Connection:
            def cursor(self):
                return Cursor()

            def commit(self) -> None:
                return None

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    output_text='{"summary":"Informations utiles sur Lou-Ann."}'
                )
            )
        )
        previous_state = {
            "available": True,
            "memory": {
                "current_topic": {
                    "topic": "Lou-Ann",
                    "messages": [{"role": "user", "content": "Quel est son job ?"}],
                    "videos_discussed": ["Portrait de Lou-Ann"],
                },
                "previous_topics": [],
            },
        }
        with (
            patch.object(conversation_memory, "load_conversation_memory", return_value=previous_state),
            patch.object(conversation_memory, "connect_database", return_value=Connection()),
        ):
            result = conversation_memory.remember_conversation_json_turn(
                12,
                "Et les vues de Simon ?",
                "Simon a 10 vues.",
                {"follow_up": False, "topic": "Vues de Simon"},
                videos_discussed=["Portrait de Simon"],
                summary_client=client,
            )

        stored = conversation_memory.json.loads(written["memory"])
        self.assertTrue(result["available"])
        self.assertTrue(result["topic_changed"])
        self.assertEqual(stored["previous_topics"][0]["summary"], "Informations utiles sur Lou-Ann.")
        self.assertEqual(stored["current_topic"]["topic"], "Vues de Simon")

    def test_tenth_topic_compacts_closed_topics_in_json(self) -> None:
        cursor = MagicMock()
        cursor.rowcount = 1
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        connection.__enter__.return_value = connection
        memory = {
            "current_topic": {"id": 9, "topic": "Sujet 9", "messages": [{"role": "user", "content": "Question 9"}]},
            "previous_topics": [
                {"id": index, "topic": f"Sujet {index}", "summary": f"Résumé {index}"}
                for index in range(1, 9)
            ],
        }
        with (
            patch.object(conversation_memory, "load_conversation_memory", return_value={"available": True, "memory": memory}),
            patch.object(conversation_memory, "connect_database", return_value=connection),
            patch.object(conversation_memory, "summarize_topic_messages", return_value=("Résumé 9", {})),
            patch.object(conversation_memory, "summarize_previous_topics", return_value=("Résumé des sujets 1 à 9", {})),
        ):
            result = conversation_memory.remember_conversation_json_turn(
                12, "Question 10", "Réponse 10", {"follow_up": False, "topic": "Sujet 10"}
            )

        stored = conversation_memory.json.loads(cursor.execute.call_args.args[1][0])
        self.assertEqual(stored["current_topic"]["id"], 10)
        self.assertEqual(stored["previous_topics"], [])
        self.assertEqual(stored["old_topics_summary"], "Résumé des sujets 1 à 9")
        self.assertEqual(result["previous_topic_count"], 0)

    def test_legacy_reformulation_context_comes_from_json(self) -> None:
        memory = {
            "current_topic": {
                "id": 2,
                "topic": "Simon",
                "messages": [{"role": "user", "content": "Qui est Simon ?"}],
            },
            "previous_topics": [{"id": 1, "topic": "Lou-Ann", "summary": "Lou-Ann travaille chez ALK."}],
        }
        with patch.object(conversation_memory, "load_conversation_memory", return_value={"available": True, "memory": memory}):
            result = conversation_memory.load_reformulation_memory(12, "Et lui ?")

        self.assertEqual(result["active_topic"], "Simon")
        self.assertEqual(result["episodes"][0]["content"], "Lou-Ann travaille chez ALK.")
        self.assertEqual(result["retrieval"]["source"], "chat.conversations.memory_json")


if __name__ == "__main__":
    unittest.main()
