"""JSON-backed conversation memory for chat reformulation."""
from __future__ import annotations

import json
from typing import Any

from interface.backend.config import DEFAULT_GENERATION_MODEL
from interface.backend.database import connect_database, ensure_chat_schema


MEMORY_SUMMARY_MAX_CHARS = 900
TOPIC_COMPACTION_THRESHOLD = 10


MEMORY_SUMMARY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
MEMORY_SUMMARY_MAX_OUTPUT_TOKENS = 512

EMPTY_CONVERSATION_MEMORY: dict[str, Any] = {
    "current_topic": {"id": 1, "topic": "", "messages": [], "videos_discussed": []},
    "previous_topics": [],
}


def normalize_videos_discussed(value: Any) -> list[str]:
    """Keep topic video titles ordered, non-empty, and unique."""
    values = value if isinstance(value, list) else []
    return list(dict.fromkeys(
        str(item).strip() for item in values if str(item).strip()
    ))


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
                "videos_discussed": normalize_videos_discussed(
                    topic.get("videos_discussed", topic.get("video_titles"))
                ),
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
            "videos_discussed": normalize_videos_discussed(
                current.get("videos_discussed", current.get("video_titles"))
            ),
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
            max_output_tokens=MEMORY_SUMMARY_MAX_OUTPUT_TOKENS,
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
            max_output_tokens=MEMORY_SUMMARY_MAX_OUTPUT_TOKENS,
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
    videos_discussed: list[str] | None = None,
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
    selected_videos_discussed = normalize_videos_discussed(videos_discussed)
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
            {
                "id": current["id"],
                "topic": current["topic"],
                "summary": summary,
                "videos_discussed": current["videos_discussed"],
            }
        )
        memory["current_topic"] = {
            "id": current["id"] + 1,
            "topic": requested_topic,
            "messages": turn,
            "videos_discussed": selected_videos_discussed,
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
        current["videos_discussed"] = normalize_videos_discussed(
            [*current["videos_discussed"], *selected_videos_discussed]
        )
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
            "videos_discussed": memory["current_topic"]["videos_discussed"],
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


def load_reformulation_memory(
    conversation_id: int | None,
    _question: str,
    *,
    include_episodes: bool = True,
    embed_question: bool = False,
    exclude_topic_id: int | None = None,
) -> dict[str, Any]:
    """Expose JSON memory in the legacy reformulation-context shape.

    Topic IDs, embeddings and SQL similarity were retired with the normalized
    topic tables.  The complete memory JSON is now the only topic context.
    """
    if conversation_id is None:
        return {"available": False, "reason": "no_conversation_id", "immediate_history": [], "episodes": []}
    loaded = load_conversation_memory(conversation_id)
    if not loaded["available"]:
        return {"available": False, "reason": loaded.get("reason"), "immediate_history": [], "episodes": []}
    memory = loaded["memory"]
    current = memory["current_topic"]
    immediate_history = [
        {"role": str(message["role"]), "text": str(message["content"])}
        for message in current["messages"][-6:]
    ]
    episodes = (
        [
            {"topic": topic["topic"], "content": topic["summary"]}
            for topic in memory["previous_topics"]
        ]
        if include_episodes
        else []
    )
    return {
        "available": True,
        "active_topic": current["topic"],
        "related_topics": [],
        "immediate_history": immediate_history,
        "episodes": episodes,
        "history": {"source": "chat.conversations.memory_json"},
        "retrieval": {"source": "chat.conversations.memory_json", "question_embedded": False},
    }
