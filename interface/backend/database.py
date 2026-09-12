from __future__ import annotations

import os
import threading
from typing import Any

import psycopg


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


def get_analytics_database_url() -> str:
    database_url = os.getenv("ANALYTICS_DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "ANALYTICS_DATABASE_URL manquant : le Text-to-SQL exige un compte PostgreSQL read-only."
        )
    return database_url


def _connection_url(database_url: str) -> str:
    """Avoid Windows/Docker localhost resolution stalls in local development."""
    # On Windows, libpq can spend more than two minutes trying the IPv6
    # localhost address before falling back to IPv4.  The host name used by
    # Compose in production is ``postgres``, so normalizing an explicit local
    # URL is safe regardless of how RAG_IONIS_ENV was inherited by Uvicorn.
    return database_url.replace("@localhost:", "@127.0.0.1:")


def connect_database():
    return psycopg.connect(
        _connection_url(get_database_url()),
        options="-c search_path=data,public",
        connect_timeout=5,
    )


def connect_analytics_database():
    return psycopg.connect(
        _connection_url(get_analytics_database_url()),
        options="-c search_path=data,public",
        connect_timeout=5,
    )


def ensure_chat_schema() -> None:
    """Create the JSON-backed chat storage and remove retired topic tables.

    ``memory_json`` is the single source of truth for topic state.  The cleanup
    is deliberately idempotent so existing installations converge safely.
    """
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
                        date TIMESTAMPTZ NOT NULL DEFAULT now(),
                        memory_json JSONB NOT NULL DEFAULT
                            '{"current_topic":{"id":1,"topic":"","messages":[]},"previous_topics":[]}'::jsonb
                    )
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE chat.conversations
                    ADD COLUMN IF NOT EXISTS memory_json JSONB NOT NULL DEFAULT
                        '{"current_topic":{"id":1,"topic":"","messages":[]},"previous_topics":[]}'::jsonb
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat.messages (
                        id BIGSERIAL PRIMARY KEY,
                        conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
                        user_message TEXT NOT NULL,
                        answer_message TEXT,
                        trace_id TEXT,
                        feedback BOOLEAN
                    )
                    """
                )
                cursor.execute(
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS answer_message TEXT"
                )
                cursor.execute(
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS trace_id TEXT"
                )
                cursor.execute(
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS feedback BOOLEAN"
                )
                cursor.execute(
                    """
                    DO $$
                    DECLARE
                        feedback_constraint RECORD;
                    BEGIN
                        IF EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'chat'
                              AND table_name = 'messages'
                              AND column_name = 'feedback'
                              AND data_type <> 'boolean'
                        ) THEN
                            FOR feedback_constraint IN
                                SELECT conname
                                FROM pg_constraint
                                WHERE conrelid = 'chat.messages'::regclass
                                  AND contype = 'c'
                                  AND pg_get_constraintdef(oid) LIKE '%feedback%'
                            LOOP
                                EXECUTE format(
                                    'ALTER TABLE chat.messages DROP CONSTRAINT %I',
                                    feedback_constraint.conname
                                );
                            END LOOP;
                            EXECUTE '
                                ALTER TABLE chat.messages
                                ALTER COLUMN feedback TYPE BOOLEAN
                                USING feedback::text::boolean
                            ';
                        END IF;
                    END
                    $$
                    """
                )
                cursor.execute(
                    """
                    DO $$
                    DECLARE
                        column_to_drop RECORD;
                    BEGIN
                        FOR column_to_drop IN
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = 'chat'
                              AND table_name = 'messages'
                              AND column_name NOT IN (
                                  'id',
                                  'conversation_id',
                                  'user_message',
                                  'answer_message',
                                  'trace_id',
                                  'feedback'
                              )
                        LOOP
                            EXECUTE format(
                                'ALTER TABLE chat.messages DROP COLUMN %I',
                                column_to_drop.column_name
                            );
                        END LOOP;
                    END
                    $$
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat.messages(conversation_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_trace_id ON chat.messages(trace_id)"
                )
                cursor.execute("DROP TABLE IF EXISTS chat.conversation_episodes")
                cursor.execute("DROP TABLE IF EXISTS chat.topic_messages")
                cursor.execute("ALTER TABLE chat.messages DROP CONSTRAINT IF EXISTS chat_messages_topic_id_fkey")
                cursor.execute("ALTER TABLE chat.messages DROP COLUMN IF EXISTS topic_id")
                cursor.execute("DROP TABLE IF EXISTS chat.conversation_topics")
                cursor.execute("DROP TABLE IF EXISTS chat.topics")
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


def create_conversation() -> int:
    """Create a conversation before the first LLM call when topic state is needed."""
    ensure_chat_schema()
    with connect_database() as connection:
        conversation_id = ensure_conversation(connection, None)
        connection.commit()
    return conversation_id


def fetch_conversation_history(
    conversation_id: int | None,
    limit: int = 8,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if conversation_id is None:
        return [], {"applied": False, "reason": "no_conversation_id", "message_count": 0}

    ensure_chat_schema()

    sql = """
        SELECT user_message, answer_message
        FROM chat.messages
        WHERE conversation_id = %s
        ORDER BY id DESC
        LIMIT %s
    """
    with connect_database() as connection:
        with connection.cursor() as cursor:
            parameters = (conversation_id, limit)
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()

    items: list[dict[str, str]] = []
    for row in reversed(rows):
        user_message = str(row[0] or "").strip()
        answer_message = str(row[1] or "").strip()
        if user_message:
            items.append({"role": "user", "text": user_message})
        if answer_message:
            items.append({"role": "assistant", "text": answer_message})

    return items, {
        "applied": True,
        "reason": None,
        "message_count": len(items),
        "sql": sql,
        "params": list(parameters),
        "latest_topic_only": False,
    }


def store_chat_message(
    conversation_id: int | None,
    user_message: str,
    answer_message: str,
    trace_id: str | None,
) -> tuple[int, int]:
    ensure_chat_schema()
    user_message = user_message.replace("\x00", "")
    answer_message = answer_message.replace("\x00", "")
    trace_id = trace_id.replace("\x00", "") if trace_id is not None else None

    with connect_database() as connection:
        resolved_conversation_id = ensure_conversation(connection, conversation_id)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat.messages (
                    conversation_id,
                    user_message,
                    answer_message,
                    trace_id
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    resolved_conversation_id,
                    user_message,
                    answer_message,
                    trace_id,
                ),
            )
            message_id = int(cursor.fetchone()[0])
        connection.commit()
    return resolved_conversation_id, message_id


def store_message_feedback(message_id: int, feedback: bool) -> bool:
    """Store a user's thumbs-up (true) or thumbs-down (false) for an answer."""
    ensure_chat_schema()
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE chat.messages
                SET feedback = %s
                WHERE id = %s AND answer_message IS NOT NULL
                """,
                (feedback, message_id),
            )
            updated = cursor.rowcount == 1
        connection.commit()
    return updated
