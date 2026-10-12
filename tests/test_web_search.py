from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from interface.backend import api, web_search


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _reference(title: str, url: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_reference", title=title, url=url)


class WebSearchParsingTests(unittest.TestCase):
    def test_outputs_join_text_and_deduplicate_web_sources(self) -> None:
        outputs = [
            SimpleNamespace(type="tool.execution", name="web_search"),
            SimpleNamespace(
                type="message.output",
                content=[
                    _text("Les frais sont de 8 000 €"),
                    _reference("Tarifs", "https://www.ionis-stm.com/tarifs"),
                    _text("."),
                    _reference("Tarifs (doublon)", "https://www.ionis-stm.com/tarifs"),
                    _reference("Lien non web", "javascript:alert(1)"),
                ],
            ),
        ]

        result = web_search.parse_web_search_outputs(outputs)

        self.assertEqual(result["answer"], "Les frais sont de 8 000 €.")
        self.assertEqual(
            result["sources"],
            [{"title": "Tarifs", "url": "https://www.ionis-stm.com/tarifs"}],
        )

    def test_search_web_only_exposes_the_web_search_tool(self) -> None:
        client = MagicMock()
        client.beta.conversations.start.return_value = SimpleNamespace(
            outputs=[SimpleNamespace(type="message.output", content="Réponse web.")]
        )

        result = web_search.search_web("Question ?", client=client)

        kwargs = client.beta.conversations.start.call_args.kwargs
        self.assertEqual(kwargs["tools"], [{"type": "web_search"}])
        self.assertFalse(kwargs["store"])
        self.assertEqual(result, {"answer": "Réponse web.", "sources": []})


class WebSearchEndpointTests(unittest.TestCase):
    def _message(self, **overrides):
        message = {
            "user_message": "Et lui ?",
            "contextual_question": "Que pense Stanislas Blanchard du télétravail ?",
            "answer_action": "abstain",
            "web_answer": None,
        }
        message.update(overrides)
        return message

    def test_unknown_message_is_404(self) -> None:
        with patch.object(api, "fetch_web_search_message", return_value=None):
            with self.assertRaises(HTTPException) as raised:
                api.web_search_message(1)
        self.assertEqual(raised.exception.status_code, 404)

    def test_answered_message_cannot_trigger_a_web_search(self) -> None:
        with (
            patch.object(
                api,
                "fetch_web_search_message",
                return_value=self._message(answer_action="answer"),
            ),
            patch.object(api, "search_web") as search,
        ):
            with self.assertRaises(HTTPException) as raised:
                api.web_search_message(1)
        self.assertEqual(raised.exception.status_code, 409)
        search.assert_not_called()

    def test_cached_web_answer_is_returned_without_new_search(self) -> None:
        cached = {"answer": "Déjà cherché.", "sources": []}
        with (
            patch.object(
                api,
                "fetch_web_search_message",
                return_value=self._message(web_answer=cached),
            ),
            patch.object(api, "search_web") as search,
        ):
            result = api.web_search_message(7)
        self.assertEqual(result, {"message_id": 7, **cached})
        search.assert_not_called()

    def test_abstained_message_searches_contextual_question_and_caches(self) -> None:
        web_answer = {"answer": "Réponse web.", "sources": []}
        with (
            patch.object(api, "fetch_web_search_message", return_value=self._message()),
            patch.object(api, "search_web", return_value=web_answer) as search,
            patch.object(api, "store_web_answer") as store,
        ):
            result = api.web_search_message(7)
        search.assert_called_once_with("Que pense Stanislas Blanchard du télétravail ?")
        store.assert_called_once_with(7, web_answer)
        self.assertEqual(result["answer"], "Réponse web.")


if __name__ == "__main__":
    unittest.main()
