"""Persistent, topic-aware memory for chat reformulation.

The memory is deliberately separate from the RAG corpus: it stores compact
conversation episodes, ranks them with lexical, entity, semantic and recency
signals, and never lets an unavailable memory backend prevent an answer.
"""
from __future__ import annotations

import math
import json
import re
from collections.abc import Iterable
from typing import Any

from interface.backend.config import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL
from interface.backend.database import connect_database, fetch_conversation_history
from interface.backend.utilities import get_openai_client, normalize_text


IMMEDIATE_HISTORY_EXCHANGES = 2
MEMORY_EPISODE_LIMIT = 3
MEMORY_TOKEN_BUDGET_CHARS = 3_600
_STOPWORDS = {
    "avec", "dans", "pour", "plus", "moins", "quel", "quelle", "quels", "elles",
    "elle", "eux", "nous", "vous", "leur", "leurs", "deux", "video", "videos",
    "vues", "faire", "fait", "sont", "est", "une", "des", "les", "que", "qui",
}


def _json_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


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


def _embedding(text: str) -> list[float] | None:
    client = get_openai_client()
    if client is None or not text.strip():
        return None
    try:
        response = client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            input=text[:12_000],
        )
        return list(response.data[0].embedding)
    except Exception:
        return None


def _vector_literal(values: list[float] | None) -> str | None:
    if not values:
        return None
    return "[" + ",".join(str(value) for value in values) + "]"


def _cosine(left: Iterable[float] | None, right: Iterable[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    left_values, right_values = list(left), list(right)
    if len(left_values) != len(right_values) or not left_values:
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left_values)) * math.sqrt(sum(value * value for value in right_values))
    return sum(a * b for a, b in zip(left_values, right_values)) / denominator if denominator else 0.0


def _parse_vector(value: Any) -> list[float] | None:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str) and value.startswith("["):
        try:
            return [float(item) for item in value[1:-1].split(",") if item]
        except ValueError:
            return None
    return None


def _episode_text(question: str, answer: str) -> str:
    return f"Question : {question.strip()}\nRéponse : {answer.strip()}"


def _topic_score(entities: set[str], keywords: set[str], topic_entities: set[str], topic_keywords: set[str]) -> float:
    entity_overlap = len({normalize_text(item) for item in entities} & {normalize_text(item) for item in topic_entities})
    keyword_overlap = len(keywords & topic_keywords)
    return entity_overlap * 5 + keyword_overlap


def remember_conversation_turn(
    conversation_id: int,
    message_id: int,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """Persist one episode and update its compact topic state after an answer."""
    entities, keywords = extract_memory_signals(question, answer)
    episode = _episode_text(question, answer)
    embedding = _embedding(episode)
    try:
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, summary, entities, keywords
                    FROM chat.conversation_topics
                    WHERE conversation_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 12
                    """,
                    (conversation_id,),
                )
                topics = cursor.fetchall()
                best_topic = None
                best_topic_summary: dict[str, Any] = {}
                best_score = 0.0
                for topic_id, _summary, topic_entities, topic_keywords in topics:
                    score = _topic_score(set(entities), set(keywords), set(_json_list(topic_entities)), set(_json_list(topic_keywords)))
                    if score > best_score:
                        best_topic, best_score = int(topic_id), score
                        best_topic_summary = dict(_summary) if isinstance(_summary, dict) else {}

                prior_entities = _json_list(best_topic_summary.get("entities"))
                prior_decisions = _json_list(best_topic_summary.get("decisions"))
                summary = {
                    "objective": question.strip()[:500],
                    "latest_answer": answer.strip()[:1_200],
                    "decisions": (prior_decisions + [answer.strip()[:280]])[-4:],
                    "constraints": _json_list(best_topic_summary.get("constraints"))[-4:],
                    "entities": sorted(set(prior_entities) | set(entities)),
                    "open_questions": [],
                }
                if best_topic is None or best_score < 2:
                    cursor.execute(
                        """
                        INSERT INTO chat.conversation_topics (conversation_id, summary, entities, keywords)
                        VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb)
                        RETURNING id
                        """,
                        (conversation_id, json.dumps(summary), json.dumps(entities), json.dumps(keywords)),
                    )
                    best_topic = int(cursor.fetchone()[0])
                    decision = "new_topic"
                else:
                    cursor.execute(
                        """
                        UPDATE chat.conversation_topics
                        SET summary = %s::jsonb, entities = %s::jsonb, keywords = %s::jsonb,
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (json.dumps(summary), json.dumps(summary["entities"]), json.dumps(keywords), best_topic),
                    )
                    decision = "continued_topic"

                cursor.execute(
                    """
                    INSERT INTO chat.conversation_episodes
                        (conversation_id, topic_id, message_id, content, entities, keywords, embedding)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::vector)
                    ON CONFLICT (message_id) DO NOTHING
                    """,
                    (conversation_id, best_topic, message_id, episode, json.dumps(entities), json.dumps(keywords), _vector_literal(embedding)),
                )
            connection.commit()
        return {"available": True, "topic_id": best_topic, "decision": decision, "embedded": embedding is not None}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def load_reformulation_memory(
    conversation_id: int | None,
    question: str,
    *,
    include_episodes: bool = True,
) -> dict[str, Any]:
    """Return bounded immediate context, active topic and hybrid-ranked episodes."""
    if conversation_id is None:
        return {"available": False, "reason": "no_conversation_id", "immediate_history": [], "episodes": []}
    try:
        immediate_history, history_trace = fetch_conversation_history(
            conversation_id,
            limit=IMMEDIATE_HISTORY_EXCHANGES,
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc), "immediate_history": [], "episodes": []}
    query_entities, query_keywords = extract_memory_signals(question)
    query_embedding = _embedding(question) if include_episodes else None
    try:
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, summary, entities, keywords
                    FROM chat.conversation_topics
                    WHERE conversation_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (conversation_id,),
                )
                active = cursor.fetchone()
                if include_episodes:
                    cursor.execute(
                        """
                        SELECT id, topic_id, message_id, content, entities, keywords,
                               embedding::text, created_at
                        FROM chat.conversation_episodes
                        WHERE conversation_id = %s
                        ORDER BY message_id DESC
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
    active_summary = active[1] if active else {}
    active_entities = set(_json_list(active[2])) if active else set()
    active_keywords = set(_json_list(active[3])) if active else set()
    candidates = []
    total = len(rows)
    for index, row in enumerate(rows):
        episode_entities, episode_keywords = set(_json_list(row[4])), set(_json_list(row[5]))
        lexical = len(set(query_keywords) & episode_keywords)
        entity = len({normalize_text(item) for item in query_entities} & {normalize_text(item) for item in episode_entities})
        semantic = max(0.0, _cosine(query_embedding, _parse_vector(row[6])))
        recency = (total - index) / max(total, 1)
        topic_bonus = 1.5 if active_id is not None and int(row[1]) == active_id else 0.0
        score = semantic * 5 + entity * 4 + lexical * 1.5 + topic_bonus + recency
        candidates.append((score, row))

    selected, used_chars = [], 0
    for score, row in sorted(candidates, key=lambda item: item[0], reverse=True):
        content = str(row[3])
        if used_chars + len(content) > MEMORY_TOKEN_BUDGET_CHARS and selected:
            continue
        selected.append({"episode_id": int(row[0]), "topic_id": int(row[1]), "message_id": int(row[2]), "content": content[:1_500], "score": round(score, 3)})
        used_chars += len(content)
        if len(selected) >= MEMORY_EPISODE_LIMIT:
            break

    query_matches_active = _topic_score(set(query_entities), set(query_keywords), active_entities, active_keywords) > 0
    return {
        "available": True,
        "topic_decision": "continue_active" if query_matches_active else "possible_topic_switch_or_reference",
        "active_topic": active_summary,
        "active_topic_id": active_id,
        "immediate_history": immediate_history,
        "episodes": selected,
        "history": history_trace,
        "retrieval": {"hybrid": ["embedding", "keywords", "entities", "topic", "recency"], "candidate_count": len(rows), "selected_count": len(selected), "char_budget": MEMORY_TOKEN_BUDGET_CHARS, "used_chars": used_chars, "query_embedded": query_embedding is not None},
    }
