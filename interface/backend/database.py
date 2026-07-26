from __future__ import annotations

import json
import os
import threading
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


# État d'initialisation partagé par les accès SQL du backend.
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()


class ConversationNotFoundError(LookupError):
    pass


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL manquant dans l'environnement.")
    return database_url


def connect_database():
    return psycopg.connect(
        get_database_url(),
        options="-c search_path=data,public",
    )


def ensure_chat_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return

    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return

        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE SCHEMA IF NOT EXISTS chat")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat.conversations (
                        id BIGSERIAL PRIMARY KEY,
                        date TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat.messages (
                        id BIGSERIAL PRIMARY KEY,
                        conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
                        user_message TEXT NOT NULL,
                        question_reformulation_prompt TEXT,
                        question_reformulation_response_raw TEXT,
                        contextual_question TEXT,
                        planner_prompt TEXT,
                        planner_response_raw TEXT,
                        person_resolution_trace JSONB,
                        pydantic_verification BOOLEAN NOT NULL DEFAULT FALSE,
                        execution_plan_json JSONB,
                        sql_query JSONB,
                        prefilter_trace JSONB,
                        bm25_trace JSONB,
                        vector_trace JSONB,
                        rrf_trace JSONB,
                        rerank_trace JSONB,
                        source_evaluation_trace JSONB,
                        multi_source_actions JSONB,
                        answer_prompt TEXT,
                        answer_response_raw TEXT,
                        answer_message TEXT,
                        cited_chunks JSONB,
                        date TIMESTAMPTZ NOT NULL DEFAULT now(),
                        trace_id TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT
                        EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'chat'
                              AND table_name = 'messages'
                              AND column_name = 'source_evaluation_trace'
                        ),
                        (
                            SELECT ordinal_position
                            FROM information_schema.columns
                            WHERE table_schema = 'chat'
                              AND table_name = 'messages'
                              AND column_name = 'source_evaluation_trace'
                        ),
                        (
                            SELECT ordinal_position
                            FROM information_schema.columns
                            WHERE table_schema = 'chat'
                              AND table_name = 'messages'
                              AND column_name = 'multi_source_actions'
                        ),
                        (
                            SELECT ordinal_position
                            FROM information_schema.columns
                            WHERE table_schema = 'chat'
                              AND table_name = 'messages'
                              AND column_name = 'cited_chunks'
                        ),
                        (
                            SELECT ordinal_position
                            FROM information_schema.columns
                            WHERE table_schema = 'chat'
                              AND table_name = 'messages'
                              AND column_name = 'date'
                        )
                    """
                )
                (
                    source_trace_exists,
                    source_trace_position,
                    multi_source_actions_position,
                    cited_position,
                    date_position,
                ) = cursor.fetchone()
                if (
                    not source_trace_exists
                    or source_trace_position is None
                    or multi_source_actions_position is None
                    or multi_source_actions_position != source_trace_position + 1
                    or cited_position is None
                    or date_position is None
                    or cited_position != date_position - 1
                ):
                    # Migration volontaire : l'historique de chat est supprime
                    # pour reconstruire la table avec les colonnes dans le bon ordre.
                    cursor.execute("DROP TABLE chat.messages")
                    cursor.execute(
                        """
                        CREATE TABLE chat.messages (
                            id BIGSERIAL PRIMARY KEY,
                            conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
                            user_message TEXT NOT NULL,
                            question_reformulation_prompt TEXT,
                            question_reformulation_response_raw TEXT,
                            contextual_question TEXT,
                            planner_prompt TEXT,
                            planner_response_raw TEXT,
                            person_resolution_trace JSONB,
                            pydantic_verification BOOLEAN NOT NULL DEFAULT FALSE,
                            execution_plan_json JSONB,
                            sql_query JSONB,
                            prefilter_trace JSONB,
                            bm25_trace JSONB,
                            vector_trace JSONB,
                            rrf_trace JSONB,
                            rerank_trace JSONB,
                            source_evaluation_trace JSONB,
                            multi_source_actions JSONB,
                            answer_prompt TEXT,
                            answer_response_raw TEXT,
                            answer_message TEXT,
                            cited_chunks JSONB,
                            date TIMESTAMPTZ NOT NULL DEFAULT now(),
                            trace_id TEXT
                        )
                        """
                    )
                for statement in (
                    """DO $$ BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat' AND table_name = 'messages'
                              AND column_name = 'question_reformulation_trace'
                        ) AND NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat' AND table_name = 'messages'
                              AND column_name = 'question_reformulation_prompt'
                        ) THEN
                            ALTER TABLE chat.messages RENAME COLUMN question_reformulation_trace TO question_reformulation_prompt;
                        END IF;
                    END $$""",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS question_reformulation_prompt TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS question_reformulation_response_raw TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS contextual_question TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS planner_prompt TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS planner_response_raw TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS person_resolution_trace JSONB",
                    "ALTER TABLE chat.messages DROP COLUMN IF EXISTS intent_source",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS pydantic_verification BOOLEAN NOT NULL DEFAULT FALSE",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS execution_plan_json JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS sql_query JSONB",
                    """DO $$ BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat' AND table_name = 'messages'
                              AND column_name = 'sql_query' AND data_type = 'text'
                        ) THEN
                            ALTER TABLE chat.messages
                            ALTER COLUMN sql_query TYPE JSONB
                            USING CASE
                                WHEN sql_query IS NULL THEN NULL
                                ELSE jsonb_build_object(
                                    'sql', sql_query,
                                    'params', '[]'::jsonb,
                                    'source', 'legacy'
                                )
                            END;
                        END IF;
                    END $$""",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS prefilter_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS bm25_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS vector_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS rrf_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS rerank_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS source_evaluation_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS multi_source_actions JSONB",
                    """DO $$ BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat' AND table_name = 'messages'
                              AND column_name = 'retrieved_chunks'
                        ) AND NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat' AND table_name = 'messages'
                              AND column_name = 'cited_chunks'
                        ) THEN
                            ALTER TABLE chat.messages RENAME COLUMN retrieved_chunks TO cited_chunks;
                        END IF;
                    END $$""",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS cited_chunks JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS answer_prompt TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS answer_response_raw TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS answer_message TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS date TIMESTAMPTZ NOT NULL DEFAULT now()",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS trace_id TEXT",
                ):
                    cursor.execute(statement)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat.messages(conversation_id)"
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_date ON chat.messages(date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_trace_id ON chat.messages(trace_id)")
            connection.commit()

        _SCHEMA_READY = True


def ensure_conversation(connection: psycopg.Connection[Any], conversation_id: int | None) -> int:
    with connection.cursor() as cursor:
        if conversation_id is not None:
            cursor.execute("SELECT id FROM chat.conversations WHERE id = %s", (conversation_id,))
            row = cursor.fetchone()
            if row is None:
                raise ConversationNotFoundError(f"Conversation introuvable: {conversation_id}")
            return int(row[0])
        cursor.execute("INSERT INTO chat.conversations DEFAULT VALUES RETURNING id")
        return int(cursor.fetchone()[0])


def fetch_conversation_memory(conversation_id: int | None, limit: int = 8) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if conversation_id is None:
        return [], {"applied": False, "reason": "no_conversation_id", "message_count": 0}

    ensure_chat_schema()

    sql = """
        SELECT user_message, answer_message, cited_chunks
        FROM chat.messages
        WHERE conversation_id = %s
        ORDER BY id DESC
        LIMIT %s
    """
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (conversation_id, limit))
            rows = cursor.fetchall()

    items: list[dict[str, str]] = []
    for row in reversed(rows):
        user_message = str(row[0] or "").strip()
        answer_message = str(row[1] or "").strip()
        if user_message:
            items.append({"role": "user", "text": user_message})
        if answer_message:
            source_context = ""
            try:
                cited_chunks = row[2]
                if isinstance(cited_chunks, str):
                    cited_chunks = json.loads(cited_chunks)
                if isinstance(cited_chunks, list):
                    source_labels = []
                    for source in cited_chunks[:5]:
                        if not isinstance(source, dict):
                            continue
                        title = str(source.get("video_title") or "").strip()
                        persons = source.get("persons") or []
                        if title:
                            label = title
                            if persons:
                                label += f" (intervenants : {', '.join(map(str, persons))})"
                            source_labels.append(label)
                    if source_labels:
                        source_context = "\nSources de la réponse précédente : " + " ; ".join(source_labels)
            except (TypeError, ValueError, json.JSONDecodeError):
                source_context = ""
            items.append({"role": "assistant", "text": answer_message + source_context})

    return items, {
        "applied": True,
        "reason": None,
        "message_count": len(items),
        "sql": sql,
        "params": [conversation_id, limit],
    }


def sql_trace_for_storage(retrieval: dict[str, Any]) -> dict[str, Any] | None:
    candidate = retrieval.get("sql_query")
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str) or not candidate.strip():
        return None

    if retrieval.get("direct_lookup"):
        source = "structured_sql"
        params = retrieval["direct_lookup"].get("params", [])
    elif retrieval.get("prefilter"):
        source = "prefilter"
        params = retrieval["prefilter"].get("params", [])
    elif retrieval.get("memory"):
        source = "memory"
        params = retrieval["memory"].get("params", [])
    else:
        source = "unknown"
        params = []
    return {"sql": candidate, "params": params, "source": source}


def store_chat_message(
    conversation_id: int | None,
    user_message: str,
    answer_message: str,
    question_reformulation_response_raw: str | None,
    contextual_question: str,
    question_reformulation_prompt: str | None,
    planner_prompt: str | None,
    planner_response_raw: str | None,
    person_resolution_trace: dict[str, Any],
    pydantic_verification: bool,
    execution_plan_json: dict[str, Any],
    sql_query: dict[str, Any] | None,
    prefilter_trace: dict[str, Any],
    bm25_trace: dict[str, Any],
    vector_trace: dict[str, Any],
    rrf_trace: dict[str, Any],
    rerank_trace: dict[str, Any],
    source_evaluation_trace: dict[str, Any],
    multi_source_actions: list[dict[str, Any]],
    cited_chunks: list[dict[str, Any]],
    answer_prompt: str | None,
    answer_response_raw: str | None,
    trace_id: str | None,
) -> tuple[int, int]:
    ensure_chat_schema()

    with connect_database() as connection:
        resolved_conversation_id = ensure_conversation(connection, conversation_id)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat.messages (
                    conversation_id,
                    user_message,
                    question_reformulation_prompt,
                    question_reformulation_response_raw,
                    contextual_question,
                    planner_prompt,
                    planner_response_raw,
                    person_resolution_trace,
                    pydantic_verification,
                    execution_plan_json,
                    sql_query,
                    prefilter_trace,
                    bm25_trace,
                    vector_trace,
                    rrf_trace,
                    rerank_trace,
                    source_evaluation_trace,
                    multi_source_actions,
                    cited_chunks,
                    answer_prompt,
                    answer_response_raw,
                    answer_message,
                    trace_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    resolved_conversation_id,
                    user_message,
                    question_reformulation_prompt,
                    question_reformulation_response_raw,
                    contextual_question,
                    planner_prompt,
                    planner_response_raw,
                    Jsonb(person_resolution_trace),
                    pydantic_verification,
                    Jsonb(execution_plan_json),
                    Jsonb(sql_query) if sql_query is not None else None,
                    Jsonb(prefilter_trace),
                    Jsonb(bm25_trace),
                    Jsonb(vector_trace),
                    Jsonb(rrf_trace),
                    Jsonb(rerank_trace),
                    Jsonb(source_evaluation_trace),
                    Jsonb(multi_source_actions),
                    Jsonb(cited_chunks),
                    answer_prompt,
                    answer_response_raw,
                    answer_message,
                    trace_id,
                ),
            )
            message_id = int(cursor.fetchone()[0])
        connection.commit()
    return resolved_conversation_id, message_id
