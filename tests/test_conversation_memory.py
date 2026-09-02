from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from interface.backend import conversation_memory


class ConversationMemoryTests(unittest.TestCase):
    def test_signals_keep_named_entities_and_drop_common_words(self) -> None:
        entities, keywords = conversation_memory.extract_memory_signals(
            "Quel est le job de Lou-Ann Corveddu chez ALK ?"
        )

        self.assertIn("Lou-Ann Corveddu", entities)
        self.assertIn("ALK", entities)
        self.assertIn("corveddu", keywords)
        self.assertNotIn("quel", keywords)

    def test_topic_score_prioritizes_entities_over_keyword_overlap(self) -> None:
        by_entity = conversation_memory._topic_score(
            {"Lou-Ann Corveddu"}, {"job"}, {"Lou-Ann Corveddu"}, set()
        )
        by_keywords = conversation_memory._topic_score(
            set(), {"job", "marketing"}, set(), {"job", "marketing"}
        )

        self.assertGreater(by_entity, by_keywords)

    def test_memory_is_optional_without_a_conversation(self) -> None:
        memory = conversation_memory.load_reformulation_memory(None, "Et la deuxième ?")

        self.assertFalse(memory["available"])
        self.assertEqual(memory["reason"], "no_conversation_id")

    def test_memory_failure_keeps_the_request_available(self) -> None:
        with patch.object(
            conversation_memory,
            "fetch_conversation_history",
            side_effect=RuntimeError("database unavailable"),
        ):
            memory = conversation_memory.load_reformulation_memory(12, "Et elle ?")

        self.assertFalse(memory["available"])
        self.assertEqual(memory["reason"], "database unavailable")

    def test_topic_summary_is_text_generated_by_the_llm(self) -> None:
        calls: list[dict] = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text='{"summary":"Lou-Ann Corveddu : vidéo et commentaires à vérifier."}'
            )

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        summary, trace = conversation_memory.summarize_topic_turn(
            client,
            "mistral-medium-latest",
            "Sujet précédent : Déborah.",
            "Et la vidéo de Lou Ann ?",
            "Je recherche la vidéo de Lou-Ann Corveddu.",
        )

        self.assertEqual(summary, "Lou-Ann Corveddu : vidéo et commentaires à vérifier.")
        self.assertEqual(trace["status"], "completed")
        self.assertEqual(calls[0]["response_schema"], conversation_memory.MEMORY_SUMMARY_RESPONSE_SCHEMA)

    def test_memory_turn_is_optional_when_database_unavailable(self) -> None:
        with patch.object(
            conversation_memory,
            "connect_database",
            side_effect=RuntimeError("database unavailable"),
        ):
            result = conversation_memory.remember_conversation_turn(
                12,
                34,
                "Et elle ?",
                "Réponse.",
            )
        self.assertFalse(result["available"])

    def test_closed_topic_is_summarized_and_moved_to_previous_topics(self) -> None:
        written: dict[str, object] = {}

        class Cursor:
            rowcount = 1

            def execute(self, query, parameters) -> None:
                written["query"] = query
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
                summary_client=client,
            )

        stored = conversation_memory.json.loads(written["memory"])
        self.assertTrue(result["available"])
        self.assertTrue(result["topic_changed"])
        self.assertEqual(stored["previous_topics"][0]["id"], 1)
        self.assertEqual(stored["previous_topics"][0]["topic"], "Lou-Ann")
        self.assertEqual(stored["previous_topics"][0]["summary"], "Informations utiles sur Lou-Ann.")
        self.assertEqual(stored["current_topic"]["topic"], "Vues de Simon")
        self.assertEqual(stored["current_topic"]["id"], 2)
        self.assertEqual(stored["current_topic"]["messages"][0]["content"], "Et les vues de Simon ?")

    def test_tenth_topic_compacts_closed_topics_into_one_summary(self) -> None:
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
            patch.object(conversation_memory, "summarize_previous_topics", return_value=("Résumé des sujets 1 à 9", {})) as compact,
        ):
            result = conversation_memory.remember_conversation_json_turn(
                12, "Question 10", "Réponse 10", {"follow_up": False, "topic": "Sujet 10"}
            )

        stored = conversation_memory.json.loads(cursor.execute.call_args.args[1][0])
        compact.assert_called_once()
        self.assertEqual(stored["current_topic"]["id"], 10)
        self.assertEqual(stored["previous_topics"], [])
        self.assertEqual(stored["old_topics_summary"], "Résumé des sujets 1 à 9")
        self.assertEqual(result["previous_topic_count"], 0)

    def test_memory_embedding_is_traced_without_the_vector(self) -> None:
        recorded: dict = {}

        class Span:
            def set_output(self, value: dict) -> None:
                recorded["output"] = value

        @contextmanager
        def trace(name: str, **kwargs):
            recorded["name"] = name
            recorded.update(kwargs)
            yield Span()

        client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1, 0.2])]
                )
            )
        )
        with (
            patch.object(conversation_memory, "get_openai_client", return_value=client),
            patch.object(conversation_memory, "trace_operation", side_effect=trace),
        ):
            result = conversation_memory._embedding("Résumé du sujet")

        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(recorded["name"], "rag.conversation_memory.embedding")
        self.assertEqual(recorded["kind"], "EMBEDDING")
        self.assertEqual(recorded["input_value"]["purpose"], "topic_summary")
        self.assertNotIn("text", recorded["input_value"])
        self.assertEqual(recorded["output"]["status"], "completed")
        self.assertEqual(recorded["output"]["dimensions"], 2)

    def test_topic_similarity_search_has_its_own_span(self) -> None:
        recorded: list[dict] = []

        class Span:
            def __init__(self, record: dict) -> None:
                self.record = record

            def set_output(self, value: dict) -> None:
                self.record["output"] = value

        @contextmanager
        def trace(name: str, **kwargs):
            record = {"name": name, **kwargs}
            recorded.append(record)
            yield Span(record)

        class Cursor:
            def __init__(self) -> None:
                self.calls = 0

            def execute(self, *_args) -> None:
                self.calls += 1

            def fetchone(self):
                return (1, "Topic actif")

            def fetchall(self):
                return [(2, "Topic proche", 0.2)]

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

        class Connection:
            def cursor(self):
                return Cursor()

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

        client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1, 0.2])]
                )
            )
        )
        with (
            patch.object(conversation_memory, "fetch_conversation_history", return_value=([], {})),
            patch.object(conversation_memory, "get_openai_client", return_value=client),
            patch.object(conversation_memory, "connect_database", return_value=Connection()),
            patch.object(conversation_memory, "trace_operation", side_effect=trace),
        ):
            memory = conversation_memory.load_reformulation_memory(
                12, "Question sur le topic", include_episodes=False, embed_question=True
            )

        span = next(item for item in recorded if item["name"] == "topic_similarity_search")
        self.assertEqual(span["kind"], "RETRIEVER")
        self.assertEqual(span["output"]["result_count"], 1)
        self.assertEqual(span["output"]["results"][0]["topic_id"], 2)
        self.assertEqual(memory["related_topics"][0]["topic_id"], 2)


if __name__ == "__main__":
    unittest.main()
