from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
