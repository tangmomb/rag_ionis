from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from interface.backend import api, research_agent


def _call(name: str, arguments: dict, call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        type="function.call",
        name=name,
        arguments=json.dumps(arguments),
        tool_call_id=call_id,
    )


def _message(*chunks) -> SimpleNamespace:
    return SimpleNamespace(type="message.output", content=list(chunks))


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _reference(title: str, url: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_reference", title=title, url=url)


def _response(*outputs) -> SimpleNamespace:
    return SimpleNamespace(conversation_id="conv-1", outputs=list(outputs))


VIDEO = {
    "video_title": "Témoignage Novares",
    "video_url": "https://www.youtube.com/watch?v=abc",
    "thumbnail_medium_url": "https://i.ytimg.com/vi/abc/mqdefault.jpg",
    "text": "Le tuteur parle de l'autonomie de son alternant.",
}


class ResearchAgentLoopTests(unittest.TestCase):
    def _client(self, start, *appends) -> MagicMock:
        client = MagicMock()
        client.beta.conversations.start.return_value = start
        client.beta.conversations.append.side_effect = list(appends)
        return client

    def test_tool_results_are_sent_back_and_cited_videos_kept(self) -> None:
        client = self._client(
            _response(_call("rechercher_videos", {"requete": "autonomie Novares"})),
            _response(
                SimpleNamespace(type="tool.execution", name="web_search"),
                _message(
                    _text(f"Le tuteur valorise l'autonomie ({VIDEO['video_url']}). "),
                    _text("D'après le web, l'alternance dure 2 ans."),
                    _reference("Article", "https://example.test/article"),
                )
            ),
        )
        search = MagicMock(return_value=({"resultats": []}, [VIDEO]))
        with patch.dict(research_agent.TOOL_HANDLERS, {"rechercher_videos": search}):
            result = research_agent.run_research_agent(
                "Que disent les tuteurs sur l'autonomie ?", "autonomie des alternants", client=client
            )

        search.assert_called_once_with("autonomie Novares")
        sent = client.beta.conversations.append.call_args.kwargs["inputs"]
        self.assertEqual(sent[0]["type"], "function.result")
        self.assertEqual(sent[0]["tool_call_id"], "call-1")
        self.assertEqual([s["video_url"] for s in result["sources"]], [VIDEO["video_url"]])
        self.assertEqual(result["web_sources"], [{"title": "Article", "url": "https://example.test/article"}])
        start_kwargs = client.beta.conversations.start.call_args.kwargs
        self.assertIn({"type": "web_search"}, start_kwargs["tools"])
        self.assertIn("Piste proposée", start_kwargs["inputs"])
        client.beta.conversations.delete.assert_called_once_with(conversation_id="conv-1")

    def test_tool_rounds_are_capped(self) -> None:
        looping = _response(_call("rechercher_videos", {"requete": "encore"}))
        appends = [looping] * research_agent.MAX_TOOL_ROUNDS + [_response(SimpleNamespace(type="tool.execution", name="web_search"), _message(_text("Rien trouvé.")))]
        client = self._client(looping, *appends)
        handler = MagicMock(return_value=({"resultats": []}, []))
        with patch.dict(research_agent.TOOL_HANDLERS, {"rechercher_videos": handler}):
            result = research_agent.run_research_agent("Question ?", client=client)

        self.assertEqual(handler.call_count, research_agent.MAX_TOOL_ROUNDS)
        last_inputs = client.beta.conversations.append.call_args.kwargs["inputs"]
        self.assertIn("Limite de recherches atteinte", last_inputs[0]["result"])
        self.assertEqual(result["answer"], "Rien trouvé.")

    def test_total_tool_calls_are_capped_across_parallel_calls(self) -> None:
        def parallel(prefix: str) -> SimpleNamespace:
            return _response(
                *[_call("rechercher_videos", {"requete": f"{prefix}{i}"}, f"{prefix}{i}") for i in range(4)]
            )

        client = self._client(
            parallel("a"), parallel("b"), _response(SimpleNamespace(type="tool.execution", name="web_search"), _message(_text("Conclusion.")))
        )
        handler = MagicMock(return_value=({"resultats": []}, []))
        with patch.dict(research_agent.TOOL_HANDLERS, {"rechercher_videos": handler}):
            research_agent.run_research_agent("Question ?", client=client)

        self.assertEqual(handler.call_count, research_agent.MAX_TOOL_CALLS)
        second_round = client.beta.conversations.append.call_args_list[1].kwargs["inputs"]
        limited = [r for r in second_round if "Limite de recherches atteinte" in r["result"]]
        self.assertEqual(len(limited), 8 - research_agent.MAX_TOOL_CALLS)

    def test_conversation_is_deleted_even_when_a_call_fails(self) -> None:
        client = self._client(_response(_call("rechercher_videos", {"requete": "x"})))
        client.beta.conversations.append.side_effect = RuntimeError("réseau")
        with patch.dict(
            research_agent.TOOL_HANDLERS,
            {"rechercher_videos": MagicMock(return_value=({"resultats": []}, []))},
        ):
            with self.assertRaises(RuntimeError):
                research_agent.run_research_agent("Question ?", client=client)
        client.beta.conversations.delete.assert_called_once_with(conversation_id="conv-1")

    def test_failing_tool_returns_an_error_to_the_agent(self) -> None:
        with patch.dict(
            research_agent.TOOL_HANDLERS,
            {"interroger_statistiques": MagicMock(side_effect=ValueError("boom"))},
        ):
            result, sources = research_agent.run_tool_call(
                _call("interroger_statistiques", {"question": "Combien de vues ?"})
            )
        self.assertEqual(json.loads(result), {"erreur": "boom"})
        self.assertEqual(sources, [])

    def test_video_search_skips_the_shared_rerank_quota(self) -> None:
        with patch.object(research_agent, "retrieve_chunks", return_value=([VIDEO], {})) as retrieve:
            result, sources = research_agent.search_videos("autonomie")
        self.assertFalse(retrieve.call_args.args[0].useRerank)
        self.assertEqual(sources, [VIDEO])
        self.assertEqual(result["resultats"][0]["video_url"], VIDEO["video_url"])

    def test_statistics_tool_explains_rejected_sql(self) -> None:
        trace = {"status": "validation_rejected", "validation": {"errors": ["unavailable_metric"]}}
        with patch.object(research_agent, "run_analytics_text_to_sql", return_value=([], trace)), patch.object(
            research_agent, "get_llm_client", return_value=None
        ):
            result, sources = research_agent.query_statistics("Combien de partages ?")
        self.assertEqual(result["erreurs"], ["unavailable_metric"])
        self.assertEqual(sources, [])


class WebProvenanceTests(unittest.TestCase):
    def _run(self, *final_outputs) -> dict:
        client = MagicMock()
        client.beta.conversations.start.return_value = _response(*final_outputs)
        return research_agent.run_research_agent("Capitale de l'Australie ?", client=client)

    def test_web_claim_without_web_search_is_removed(self) -> None:
        final = _message(_text("Rien dans les vidéos.\n\nD'après le web, c'est **Canberra**."))
        client = MagicMock()
        client.beta.conversations.start.return_value = _response(final)
        client.beta.conversations.append.return_value = _response(final)
        result = research_agent.run_research_agent("Capitale de l'Australie ?", client=client)
        self.assertEqual(result["answer"], "Rien dans les vidéos.")
        self.assertTrue(result["web_claim_removed"])
        self.assertFalse(result["web_searched"])

    def test_web_claim_with_list_and_no_source_falls_back(self) -> None:
        result = self._run(
            SimpleNamespace(type="tool.execution", name="web_search"),
            _message(_text("D’après le web, les taux sont :\n\n- A : 98 %\n\n- B : 70 %")),
        )
        self.assertEqual(result["answer"], research_agent.NO_RELIABLE_ANSWER)
        self.assertEqual(result["web_sources"], [])

    def test_sourced_web_claim_is_kept(self) -> None:
        result = self._run(
            SimpleNamespace(type="tool.execution", name="web_search"),
            _message(
                _text("D'après le web, le taux est de 96 %."),
                _reference("IONIS-STM", "https://www.ionis-stm.com/ecole"),
            ),
        )
        self.assertEqual(result["answer"], "D'après le web, le taux est de 96 %.")
        self.assertFalse(result["web_claim_removed"])
        self.assertEqual(len(result["web_sources"]), 1)


class WebLastResortTests(unittest.TestCase):
    def test_agent_is_nudged_once_to_try_the_web_before_giving_up(self) -> None:
        client = MagicMock()
        client.beta.conversations.start.return_value = _response(_message(_text("Rien dans les vidéos.")))
        client.beta.conversations.append.return_value = _response(
            SimpleNamespace(type="tool.execution", name="web_search"),
            _message(_text("D'après le web, c'est Canberra."), _reference("Gov", "https://www.australia.gov.au")),
        )

        result = research_agent.run_research_agent("Capitale de l'Australie ?", client=client)

        nudge = client.beta.conversations.append.call_args_list[0].kwargs["inputs"][0]
        self.assertEqual(nudge["type"], "message.input")
        self.assertEqual(client.beta.conversations.append.call_count, 1)
        self.assertTrue(result["web_nudged"])
        self.assertTrue(result["web_searched"])
        self.assertEqual(result["answer"], "D'après le web, c'est Canberra.")

    def test_no_nudge_when_videos_answer_the_question(self) -> None:
        client = MagicMock()
        client.beta.conversations.start.return_value = _response(
            _call("rechercher_videos", {"requete": "autonomie"})
        )
        client.beta.conversations.append.return_value = _response(
            _message(_text(f"Le tuteur valorise l'autonomie ({VIDEO['video_url']})."))
        )
        with patch.dict(
            research_agent.TOOL_HANDLERS,
            {"rechercher_videos": MagicMock(return_value=({"resultats": []}, [VIDEO]))},
        ):
            result = research_agent.run_research_agent("Autonomie ?", client=client)

        self.assertFalse(result["web_nudged"])
        self.assertEqual(client.beta.conversations.append.call_count, 1)

    def test_agent_uses_the_research_model(self) -> None:
        client = MagicMock()
        client.beta.conversations.start.return_value = _response(SimpleNamespace(type="tool.execution", name="web_search"), _message(_text("Réponse.")))
        research_agent.run_research_agent("Question ?", client=client)
        self.assertEqual(
            client.beta.conversations.start.call_args.kwargs["model"],
            research_agent.DEFAULT_RESEARCH_AGENT_MODEL,
        )


class ExpandSearchEndpointTests(unittest.TestCase):
    def _message(self, **overrides):
        message = {
            "user_message": "Et lui ?",
            "contextual_question": "Que pense Stanislas Blanchard du télétravail ?",
            "search_hint": "Avis de Stanislas Blanchard sur le télétravail",
            "answer_action": "abstain",
            "expanded_answer": None,
            "conversation_id": 42,
        }
        message.update(overrides)
        return message

    def test_unknown_message_is_404(self) -> None:
        with patch.object(api, "fetch_expanded_search_message", return_value=None):
            with self.assertRaises(HTTPException) as raised:
                api.expand_search_message(1)
        self.assertEqual(raised.exception.status_code, 404)

    def test_answered_message_cannot_start_the_agent(self) -> None:
        with (
            patch.object(api, "fetch_expanded_search_message", return_value=self._message(answer_action="answer")),
            patch.object(api, "run_research_agent") as agent,
        ):
            with self.assertRaises(HTTPException) as raised:
                api.expand_search_message(1)
        self.assertEqual(raised.exception.status_code, 409)
        agent.assert_not_called()

    def test_cached_answer_is_returned_without_new_agent_run(self) -> None:
        cached = {"answer": "Déjà cherché.", "sources": [], "web_sources": [], "tool_calls": []}
        with (
            patch.object(api, "fetch_expanded_search_message", return_value=self._message(expanded_answer=cached)),
            patch.object(api, "run_research_agent") as agent,
        ):
            result = api.expand_search_message(7)
        self.assertEqual(result, {"message_id": 7, **cached})
        agent.assert_not_called()

    def test_abstained_message_runs_agent_with_hint_and_caches(self) -> None:
        expanded = {"answer": "Réponse.", "sources": [], "web_sources": [], "tool_calls": []}
        with (
            patch.object(api, "fetch_expanded_search_message", return_value=self._message()),
            patch.object(api, "run_research_agent", return_value=expanded) as agent,
            patch.object(api, "store_expanded_answer") as store,
        ):
            result = api.expand_search_message(7)
        agent.assert_called_once_with(
            "Que pense Stanislas Blanchard du télétravail ?",
            "Avis de Stanislas Blanchard sur le télétravail",
            conversation_id=42,
            message_id=7,
        )
        store.assert_called_once_with(7, expanded)
        self.assertEqual(result["answer"], "Réponse.")


class _RecordedSpan:
    def __init__(self, name: str, kind: str, attributes: dict | None, root: bool) -> None:
        self.name = name
        self.kind = kind
        self.root = root
        self.attributes = dict(attributes or {})
        self.session_id = None

    def set_attribute(self, name, value) -> None:
        self.attributes[name] = value

    def set_session_id(self, conversation_id) -> None:
        self.session_id = conversation_id

    def set_output(self, value) -> None:
        self.output = value


class ResearchAgentTracingTests(unittest.TestCase):
    def test_agent_trace_is_a_session_root_with_llm_spans(self) -> None:
        spans: list[_RecordedSpan] = []

        @contextmanager
        def fake_trace_operation(name, *, kind="CHAIN", input_value=None, attributes=None, root=False):
            span = _RecordedSpan(name, kind, attributes, root)
            spans.append(span)
            yield span

        client = MagicMock()
        client.beta.conversations.start.return_value = _response(
            _call("rechercher_videos", {"requete": "autonomie"})
        )
        client.beta.conversations.append.return_value = SimpleNamespace(
            conversation_id="conv-1",
            outputs=[
                SimpleNamespace(type="tool.execution", name="web_search"),
                _message(_text("Réponse.")),
            ],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
        )
        search = MagicMock(return_value=({"resultats": []}, []))
        with (
            patch.object(research_agent, "trace_operation", fake_trace_operation),
            patch.dict(research_agent.TOOL_HANDLERS, {"rechercher_videos": search}),
        ):
            research_agent.run_research_agent(
                "Question ?", client=client, conversation_id=42, message_id=7
            )

        agent = spans[0]
        self.assertEqual((agent.name, agent.kind, agent.root), ("research_agent", "AGENT", True))
        self.assertEqual(agent.session_id, 42)
        self.assertEqual(agent.attributes["rag.message_id"], 7)

        llm_spans = [span for span in spans if span.kind == "LLM"]
        self.assertEqual(
            [span.attributes["llm.invocation_parameters"]["operation"] for span in llm_spans],
            ["start", "append"],
        )
        start, append = llm_spans
        self.assertEqual(start.attributes["llm.input_messages.0.message.role"], "system")
        self.assertEqual(
            start.attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"],
            "rechercher_videos",
        )
        self.assertEqual(append.attributes["llm.input_messages.0.message.role"], "tool")
        self.assertEqual(append.attributes["llm.output_messages.0.message.content"], "Réponse.")
        self.assertTrue(append.attributes["research_agent.web_searched"])
        self.assertEqual(append.attributes["llm.token_count.total"], 1200)
        self.assertIn("llm.cost.total", append.attributes)
        self.assertIn("research_agent.rechercher_videos", [span.name for span in spans])


if __name__ == "__main__":
    unittest.main()
