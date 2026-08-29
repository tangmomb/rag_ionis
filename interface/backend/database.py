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
                    DO $$
                    BEGIN
                        -- v1 stored topic content directly in conversation_topics.
                        -- Preserve its ids so messages and Phoenix traces remain valid.
                        IF to_regclass('chat.conversation_topics') IS NOT NULL
                           AND EXISTS (
                               SELECT 1 FROM information_schema.columns
                               WHERE table_schema = 'chat'
                                 AND table_name = 'conversation_topics'
                                 AND column_name = 'summary'
                           ) THEN
                            IF to_regclass('chat.topics') IS NOT NULL THEN
                                RAISE EXCEPTION 'Migration chat impossible: tables topics et conversation_topics (ancienne forme) coexistantes';
                            END IF;
                            ALTER TABLE chat.conversation_topics RENAME TO topics;
                        END IF;
                    END
                    $$
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat.topics (
                        id BIGSERIAL PRIMARY KEY,
                        summary TEXT NOT NULL DEFAULT '',
                        embedding vector(2000),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat.conversation_topics (
                        conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
                        topic_id BIGINT NOT NULL REFERENCES chat.topics(id) ON DELETE CASCADE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (conversation_id, topic_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat'
                              AND table_name = 'topics'
                              AND column_name = 'conversation_id'
                        ) THEN
                            INSERT INTO chat.conversation_topics (conversation_id, topic_id, created_at)
                            SELECT conversation_id, id, created_at
                            FROM chat.topics
                            ON CONFLICT (conversation_id, topic_id) DO NOTHING;
                            ALTER TABLE chat.topics DROP COLUMN conversation_id;
                        END IF;
                    END
                    $$
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat.messages (
                        id BIGSERIAL PRIMARY KEY,
                        conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
                        topic_id BIGINT,
                        user_message TEXT NOT NULL,
                        answer_message TEXT,
                        trace_id TEXT
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
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS topic_id BIGINT"
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
                                  'topic_id',
                                  'user_message',
                                  'answer_message',
                                  'trace_id'
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
                cursor.execute(
                    """
                    DO $$
                    BEGIN
                        ALTER TABLE chat.messages
                        DROP CONSTRAINT IF EXISTS chat_messages_topic_id_fkey;
                        ALTER TABLE chat.messages
                        ADD CONSTRAINT chat_messages_topic_id_fkey
                        FOREIGN KEY (topic_id) REFERENCES chat.topics(id)
                        ON DELETE SET NULL;
                    END
                    $$
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_topics_updated ON chat.topics(updated_at DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_conversation_topics_conversation ON chat.conversation_topics(conversation_id, topic_id DESC)")
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
            cursor.execute(sql, (conversation_id, limit))
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
        "params": [conversation_id, limit],
    }


def store_chat_message(
    conversation_id: int | None,
    user_message: str,
    answer_message: str,
    trace_id: str | None,
    topic_id: int | None = None,
) -> tuple[int, int]:
    ensure_chat_schema()

    with connect_database() as connection:
        resolved_conversation_id = ensure_conversation(connection, conversation_id)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat.messages (
                    conversation_id,
                    topic_id,
                    user_message,
                    answer_message,
                    trace_id
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    resolved_conversation_id,
                    topic_id,
                    user_message,
                    answer_message,
                    trace_id,
                ),
            )
            message_id = int(cursor.fetchone()[0])
        connection.commit()
    return resolved_conversation_id, message_id
