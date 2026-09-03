"""Persistent, topic-aware memory for chat reformulation.

The memory is deliberately separate from the RAG corpus. It keeps raw exchanges
in ``chat.messages`` and one compact text summary per topic.
"""
from __future__ import annotations

import json
import re
from typing import Any

from interface.backend.config import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
)
from interface.backend.database import connect_database, ensure_chat_schema, fetch_conversation_history
from interface.backend.telemetry import trace_operation
from interface.backend.utilities import get_openai_client, normalize_text


IMMEDIATE_HISTORY_EXCHANGES = 3
MEMORY_MESSAGE_LIMIT = 3
MEMORY_TOKEN_BUDGET_CHARS = 3_600
MEMORY_SUMMARY_MAX_CHARS = 900
TOPIC_COMPACTION_THRESHOLD = 10
TOPIC_MATCH_MAX_COSINE_DISTANCE = 0.35
_STOPWORDS = {
    "avec", "dans", "pour", "plus", "moins", "quel", "quelle", "quels", "elles",
    "elle", "eux", "nous", "vous", "leur", "leurs", "deux", "video", "videos",
    "vues", "faire", "fait", "sont", "est", "une", "des", "les", "que", "qui",
}


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", normalize_text(value))
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _entities(value: str) -> set[str]:
    """Extract display entities conservatively; the answer supplies canonical names."""
    entities = set()
    for match in re.finditer(r"\b(?:[A-ZÉÈÀÂÎÔÛÇ][\w'’.-]+(?:\s+|$)){1,4}", value):
        candidate = " ".join(match.group(0).split()).strip(" .,:;!?()")
        if len(candidate) >= 3 and candidate.lower() not in {"la", "le", "les", "une"}:
            entities.add(candidate)
    return entities


def extract_memory_signals(*values: str) -> tuple[list[str], list[str]]:
    text = "\n".join(value or "" for value in values)
    return sorted(_entities(text)), sorted(_tokens(text))[:48]


def _embedding(
    text: str,
    *,
    purpose: str = "topic_summary",
    span_name: str = "rag.conversation_memory.embedding",
) -> list[float] | None:
    """Embed memory text and expose the outcome without recording its vector."""
    client = get_openai_client()
    with trace_operation(
        span_name,
        kind="EMBEDDING",
        input_value={
            "purpose": purpose,
            "model": DEFAULT_EMBEDDING_MODEL,
            "dimensions": DEFAULT_EMBEDDING_DIMENSIONS,
            "text_length": len(text.strip()),
        },
    ) as embedding_span:
        if client is None:
            embedding_span.set_output(
                {"status": "skipped", "reason": "openai_client_unavailable"}
            )
            return None
        if not text.strip():
            embedding_span.set_output({"status": "skipped", "reason": "empty_text"})
            return None
        try:
            response = client.embeddings.create(
                model=DEFAULT_EMBEDDING_MODEL,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
                input=text[:12_000],
            )
            embedding = list(response.data[0].embedding)
            embedding_span.set_output(
                {
                    "status": "completed",
                    "model": DEFAULT_EMBEDDING_MODEL,
                    "dimensions": len(embedding),
                }
            )
            return embedding
        except Exception as exc:
            embedding_span.set_output(
                {"status": "failed", "error_type": type(exc).__name__}
            )
            return None


def _vector_literal(values: list[float] | None) -> str | None:
    if not values:
        return None
    return "[" + ",".join(str(value) for value in values) + "]"


MEMORY_SUMMARY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

EMPTY_CONVERSATION_MEMORY: dict[str, Any] = {
    "current_topic": {"id": 1, "topic": "", "messages": []},
    "previous_topics": [],
}


def normalize_conversation_memory(value: Any) -> dict[str, Any]:
    """Return the stable JSON shape persisted for one conversation."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    value = value if isinstance(value, dict) else {}
    current = value.get("current_topic")
    current = current if isinstance(current, dict) else {}
    messages = current.get("messages")
    messages = messages if isinstance(messages, list) else []
    normalized_messages = [
        {
            "role": str(message.get("role") or "user"),
            "content": str(message.get("content") or "").strip(),
        }
        for message in messages
        if isinstance(message, dict) and str(message.get("content") or "").strip()
    ]
    previous = value.get("previous_topics")
    previous = previous if isinstance(previous, list) else []
    normalized_previous = []
    for index, topic in enumerate(previous, start=1):
        if not isinstance(topic, dict) or not str(topic.get("summary") or "").strip():
            continue
        raw_id = topic.get("id")
        topic_id = raw_id if isinstance(raw_id, int) and raw_id > 0 else index
        normalized_previous.append(
            {
                "id": topic_id,
                "topic": str(topic.get("topic") or "").strip(),
                "summary": str(topic.get("summary") or "").strip(),
            }
        )
    # Keep ordering deterministic even if a hand-edited JSON has duplicate ids.
    for index, topic in enumerate(normalized_previous, start=1):
        topic["id"] = index
    raw_current_id = current.get("id")
    current_id = (
        raw_current_id
        if isinstance(raw_current_id, int) and raw_current_id > len(normalized_previous)
        else len(normalized_previous) + 1
    )
    old_topics_summary = str(
        value.get("old_topics_summary") or ""
    ).strip()
    return {
        "current_topic": {
            "id": current_id,
            "topic": str(current.get("topic") or "").strip(),
            "messages": normalized_messages,
        },
        "previous_topics": normalized_previous,
        **(
            {"old_topics_summary": old_topics_summary}
            if old_topics_summary
            else {}
        ),
    }


def load_conversation_memory(conversation_id: int | None) -> dict[str, Any]:
    """Load the complete JSON memory passed to the reformulation model."""
    if conversation_id is None:
        return {"available": False, "reason": "no_conversation_id", "memory": normalize_conversation_memory({})}
    try:
        ensure_chat_schema()
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT memory_json FROM chat.conversations WHERE id = %s",
                    (conversation_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return {"available": False, "reason": "conversation_not_found", "memory": normalize_conversation_memory({})}
        return {"available": True, "memory": normalize_conversation_memory(row[0])}
    except Exception as exc:
        return {"available": False, "reason": str(exc), "memory": normalize_conversation_memory({})}


def summarize_topic_messages(
    client: Any,
    model: str,
    topic: str,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    """Compress a closed topic once, preserving the current JSON's useful facts."""
    rendered_messages = json.dumps(messages, ensure_ascii=False)
    fallback = (f"Sujet : {topic}\n" + "\n".join(
        f"{item.get('role', 'user')} : {item.get('content', '')}" for item in messages
    ))[:MEMORY_SUMMARY_MAX_CHARS]
    if client is None:
        return fallback, {"status": "fallback", "reason": "no_llm_client"}
    system_prompt = (
        "Résume un sujet de conversation clos. Conserve les entités, faits établis, "
        "décisions et éléments nécessaires pour y revenir ultérieurement. "
        f"Réponds avec un résumé concis de moins de {MEMORY_SUMMARY_MAX_CHARS} caractères."
    )
    user_prompt = (
        f"Sujet : {topic or '(non libellé)'}\n\n"
        f"Messages du sujet (JSON) :\n{rendered_messages}"
    )
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_schema=MEMORY_SUMMARY_RESPONSE_SCHEMA,
        )
        raw = str(getattr(response, "output_text", "") or "").strip()
        summary = str(json.loads(raw).get("summary") or "").strip()
        if not summary:
            raise ValueError("empty_summary")
        return summary[:MEMORY_SUMMARY_MAX_CHARS], {"status": "completed", "model": model, "response_raw": raw}
    except Exception as exc:
        return fallback, {"status": "fallback", "reason": str(exc)}


def summarize_previous_topics(
    client: Any,
    model: str,
    previous_topics: list[dict[str, Any]],
    previous_summary: str = "",
) -> tuple[str, dict[str, Any]]:
    """Compress ten closed topics into the long-term conversation summary."""
    payload = {
        "old_topics_summary": previous_summary,
        "previous_topics": previous_topics,
    }
    fallback = json.dumps(payload, ensure_ascii=False)[:MEMORY_SUMMARY_MAX_CHARS]
    if client is None:
        return fallback, {"status": "fallback", "reason": "no_llm_client"}
    system_prompt = (
        "Résume la mémoire longue d'une conversation. Fusionne le résumé existant "
        "et les sujets clos ; conserve les entités, faits et décisions utiles pour "
        "retrouver un ancien sujet. "
        f"Réponds avec un résumé concis de moins de {MEMORY_SUMMARY_MAX_CHARS} caractères."
    )
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_schema=MEMORY_SUMMARY_RESPONSE_SCHEMA,
        )
        raw = str(getattr(response, "output_text", "") or "").strip()
        summary = str(json.loads(raw).get("summary") or "").strip()
        if not summary:
            raise ValueError("empty_summary")
        return summary[:MEMORY_SUMMARY_MAX_CHARS], {"status": "completed", "model": model, "response_raw": raw}
    except Exception as exc:
        return fallback, {"status": "fallback", "reason": str(exc)}


def remember_conversation_json_turn(
    conversation_id: int,
    question: str,
    answer: str,
    reformulation: dict[str, Any],
    *,
    summary_client: Any = None,
    summary_model: str = DEFAULT_GENERATION_MODEL,
) -> dict[str, Any]:
    """Append a turn, closing and summarising the prior topic only on a topic change."""
    loaded = load_conversation_memory(conversation_id)
    if not loaded["available"]:
        return {"available": False, "reason": loaded.get("reason")}
    memory = normalize_conversation_memory(loaded["memory"])
    current = memory["current_topic"]
    follow_up = bool(reformulation.get("follow_up", False))
    requested_topic = str(reformulation.get("topic") or "").strip()
    turn = [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]
    summary_trace: dict[str, Any] | None = None
    old_topics_summary_trace: dict[str, Any] | None = None
    topic_changed = bool(current["messages"]) and not follow_up
    if topic_changed:
        summary, summary_trace = summarize_topic_messages(
            summary_client, summary_model, current["topic"], current["messages"]
        )
        memory["previous_topics"].append(
            {"id": current["id"], "topic": current["topic"], "summary": summary}
        )
        memory["current_topic"] = {
            "id": current["id"] + 1,
            "topic": requested_topic,
            "messages": turn,
        }
        if (
            memory["current_topic"]["id"] % TOPIC_COMPACTION_THRESHOLD == 0
            and memory["previous_topics"]
        ):
            long_summary, old_topics_summary_trace = summarize_previous_topics(
                summary_client,
                summary_model,
                memory["previous_topics"],
                str(memory.get("old_topics_summary") or ""),
            )
            memory["old_topics_summary"] = long_summary
            memory["previous_topics"] = []
    else:
        if requested_topic:
            current["topic"] = requested_topic
        current["messages"].extend(turn)
    memory = normalize_conversation_memory(memory)
    try:
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE chat.conversations SET memory_json = %s::jsonb WHERE id = %s",
                    (json.dumps(memory, ensure_ascii=False), conversation_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("conversation_not_found")
            connection.commit()
        return {
            "available": True,
            "follow_up": follow_up,
            "topic_changed": topic_changed,
            "current_topic": memory["current_topic"]["topic"],
            "previous_topic_count": len(memory["previous_topics"]),
            "summary_trace": summary_trace,
            "old_topics_summary_trace": old_topics_summary_trace,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "summary_trace": summary_trace,
            "old_topics_summary_trace": old_topics_summary_trace,
        }


def summarize_topic_turn(
    client: Any,
    model: str,
    previous_summary: str,
    question: str,
    answer: str,
) -> tuple[str, dict[str, Any]]:
    """Update one compact topic summary; never let this auxiliary call fail a turn."""
    fallback = (
        f"Dernière demande : {question.strip()}\n"
        f"Dernière réponse : {answer.strip()}"
    )[:MEMORY_SUMMARY_MAX_CHARS]
    if client is None:
        return fallback, {"status": "fallback", "reason": "no_llm_client"}
    system_prompt = (
        "Tu mets à jour le résumé texte d'un seul sujet de conversation. "
        "Conserve uniquement les entités, faits établis, décisions et question en cours "
        "utiles pour les prochaines relances. Oublie le détail inutile. "
        f"Réponds avec un résumé concis de moins de {MEMORY_SUMMARY_MAX_CHARS} caractères."
    )
    user_prompt = (
        f"Résumé précédent :\n{previous_summary or '(aucun)'}\n\n"
        f"Nouveau message utilisateur :\n{question}\n\n"
        f"Nouvelle réponse assistant :\n{answer}"
    )
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_schema=MEMORY_SUMMARY_RESPONSE_SCHEMA,
        )
        raw = str(getattr(response, "output_text", "") or "").strip()
        parsed = json.loads(raw)
        summary = str(parsed.get("summary") or "").strip()
        if not summary:
            raise ValueError("empty_summary")
        return summary[:MEMORY_SUMMARY_MAX_CHARS], {
            "status": "completed",
            "model": model,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_raw": raw,
        }
    except Exception as exc:
        return fallback, {"status": "fallback", "reason": str(exc)}


def _topic_score(entities: set[str], keywords: set[str], topic_entities: set[str], topic_keywords: set[str]) -> float:
    entity_overlap = len({normalize_text(item) for item in entities} & {normalize_text(item) for item in topic_entities})
    keyword_overlap = len(keywords & topic_keywords)
    return entity_overlap * 5 + keyword_overlap


def assign_topic_id(conversation_id: int | None, follow_up: bool) -> dict[str, Any]:
    """Reserve the topic immediately after light reformulation.

    A follow-up keeps the latest topic. Any other message inserts a fresh row and
    lets PostgreSQL's increasing BIGSERIAL id allocate the next topic id.
    """
    if conversation_id is None:
        return {"reason": "no_conversation_id"}
    try:
        ensure_chat_schema()
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT t.id, t.summary
                    FROM chat.conversation_topics AS ct
                    JOIN chat.topics AS t ON t.id = ct.topic_id
                    WHERE ct.conversation_id = %s
                    ORDER BY t.id DESC
                    LIMIT 1
                    """,
                    (conversation_id,),
                )
                active = cursor.fetchone()
                if follow_up and active is not None:
                    return {
                        "topic_id": int(active[0]),
                        "decision": "current_topic",
                        "summary": str(active[1] or ""),
                    }
                cursor.execute(
                    """
                    INSERT INTO chat.topics (summary)
                    VALUES ('')
                    RETURNING id
                    """,
                )
                topic_id = int(cursor.fetchone()[0])
                cursor.execute(
                    """
                    INSERT INTO chat.conversation_topics (conversation_id, topic_id)
                    VALUES (%s, %s)
                    """,
                    (conversation_id, topic_id),
                )
            connection.commit()
        return {
            "topic_id": topic_id,
            "decision": "new_topic",
            "summary": "",
        }
    except Exception as exc:
        return {"reason": str(exc)}


def remember_conversation_turn(
    conversation_id: int,
    message_id: int,
    question: str,
    answer: str,
    *,
    summary_client: Any = None,
    summary_model: str = DEFAULT_GENERATION_MODEL,
    topic_id: int | None = None,
) -> dict[str, Any]:
    """Update the compact summary of the topic assigned before the answer."""
    try:
        with connect_database() as connection:
            with connection.cursor() as cursor:
                if topic_id is None:
                    raise RuntimeError("Aucun topic_id attribué pour ce message")
                cursor.execute(
                    """
                    SELECT t.summary
                    FROM chat.topics AS t
                    JOIN chat.conversation_topics AS ct ON ct.topic_id = t.id
                    WHERE t.id = %s AND ct.conversation_id = %s
                    """,
                    (topic_id, conversation_id),
                )
                selected_topic = cursor.fetchone()
                if selected_topic is None:
                    raise RuntimeError(f"Topic introuvable: {topic_id}")
                previous_summary = str(selected_topic[0] or "")

                summary, summary_trace = summarize_topic_turn(
                    summary_client,
                    summary_model,
                    previous_summary,
                    question,
                    answer,
                )
                embedding = _embedding(summary)
                cursor.execute(
                    """
                    UPDATE chat.topics
                    SET summary = %s, embedding = %s::vector, updated_at = now()
                    WHERE id = %s
                    """,
                    (summary, _vector_literal(embedding), topic_id),
                )
            connection.commit()
        return {"available": True, "topic_id": topic_id, "decision": "assigned_topic", "summary": summary, "embedded": embedding is not None, "summary_trace": summary_trace}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def load_reformulation_memory(
    conversation_id: int | None,
    question: str,
    *,
    include_episodes: bool = True,
    embed_question: bool = False,
    exclude_topic_id: int | None = None,
) -> dict[str, Any]:
    """Return recent exchanges, the active topic, and bounded older episodes."""
    if conversation_id is None:
        return {"available": False, "reason": "no_conversation_id", "immediate_history": [], "episodes": []}
    try:
        immediate_history, history_trace = fetch_conversation_history(
            conversation_id,
            limit=IMMEDIATE_HISTORY_EXCHANGES,
            latest_topic_only=True,
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc), "immediate_history": [], "episodes": []}
    # This path is used only for a non-follow-up, between the light and final
    # reformulation calls. It makes the current question comparable to topic
    # summary embeddings before the second LLM call.
    question_embedding = (
        _embedding(
            question,
            purpose="topic_match_query",
            span_name="embedding",
        )
        if embed_question
        else None
    )
    try:
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT t.id, t.summary
                    FROM chat.conversation_topics AS ct
                    JOIN chat.topics AS t ON t.id = ct.topic_id
                    WHERE ct.conversation_id = %s
                    ORDER BY t.updated_at DESC
                    LIMIT 1
                    """,
                    (conversation_id,),
                )
                active = cursor.fetchone()
                if question_embedding:
                    with trace_operation(
                        "topic_similarity_search",
                        kind="RETRIEVER",
                        input_value={
                            "conversation_id": conversation_id,
                            "excluded_topic_id": exclude_topic_id,
                            "result_limit": 3,
                            "match_max_cosine_distance": TOPIC_MATCH_MAX_COSINE_DISTANCE,
                        },
                    ) as similarity_span:
                        cursor.execute(
                            """
                            SELECT t.id, t.summary, t.embedding <=> %s::vector AS distance
                            FROM chat.conversation_topics AS ct
                            JOIN chat.topics AS t ON t.id = ct.topic_id
                            WHERE ct.conversation_id = %s
                              AND t.embedding IS NOT NULL
                              AND (%s::bigint IS NULL OR t.id <> %s::bigint)
                            ORDER BY t.embedding <=> %s::vector
                            LIMIT 3
                            """,
                            (
                                _vector_literal(question_embedding),
                                conversation_id,
                                exclude_topic_id,
                                exclude_topic_id,
                                _vector_literal(question_embedding),
                            ),
                        )
                        related_topics = cursor.fetchall()
                        similarity_span.set_output(
                            {
                                "result_count": len(related_topics),
                                "matching_topic_count": sum(
                                    float(row[2]) <= TOPIC_MATCH_MAX_COSINE_DISTANCE
                                    for row in related_topics
                                ),
                                "results": [
                                    {
                                        "topic_id": int(row[0]),
                                        "summary": str(row[1] or ""),
                                        "distance": round(float(row[2]), 4),
                                    }
                                    for row in related_topics
                                ],
                            }
                        )
                else:
                    related_topics = []
                if include_episodes:
                    cursor.execute(
                        """
                        SELECT id, topic_id, user_message, answer_message
                        FROM chat.messages
                        WHERE conversation_id = %s
                        ORDER BY id DESC
                        LIMIT 40
                        """,
                        (conversation_id,),
                    )
                    rows = cursor.fetchall()
                else:
                    rows = []
    except Exception as exc:
        return {"available": False, "reason": str(exc), "immediate_history": immediate_history, "episodes": [], "history": history_trace}

    active_id = int(active[0]) if active else None
    active_summary = str(active[1] or "") if active else ""
    selected, used_chars = [], 0
    for row in rows:
        content = f"Question : {str(row[2] or '').strip()}\nRéponse : {str(row[3] or '').strip()}"
        if used_chars + len(content) > MEMORY_TOKEN_BUDGET_CHARS and selected:
            continue
        selected.append({"message_id": int(row[0]), "topic_id": int(row[1]) if row[1] is not None else None, "content": content[:1_500]})
        used_chars += len(content)
        if len(selected) >= MEMORY_MESSAGE_LIMIT:
            break

    matching_topics = [
        {"topic_id": int(row[0]), "summary": str(row[1] or ""), "distance": round(float(row[2]), 4)}
        for row in related_topics
        if float(row[2]) <= TOPIC_MATCH_MAX_COSINE_DISTANCE and str(row[1] or "").strip()
    ][:1]
    return {
        "available": True,
        "active_topic": active_summary,
        "active_topic_id": active_id,
        "related_topics": matching_topics,
        "immediate_history": immediate_history,
        "episodes": selected,
        "history": history_trace,
        "retrieval": {"source": "chat.messages+conversation_topics", "candidate_count": len(rows), "selected_count": len(selected), "related_topic_count": len(related_topics), "matching_topic_count": len(matching_topics), "match_max_cosine_distance": TOPIC_MATCH_MAX_COSINE_DISTANCE, "char_budget": MEMORY_TOKEN_BUDGET_CHARS, "used_chars": used_chars, "question_embedded": question_embedding is not None},
    }
