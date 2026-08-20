import os
import sys
from pathlib import Path

import psycopg


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline.support.environment import load_project_env


TABLES_BY_SCHEMA = {
    "data": (
        "comments",
        "chunks",
        "transcripts",
        "video_transcripts",
        "video_speakers",
        "speakers",
        "update_runs",
        "stats",
        "video_stats",
        "video_daily_stats",
        "videos",
    ),
    "chat": (
        "messages",
        "conversations",
    ),
}


def existing_tables(cursor, schema_name, tables):
    cursor.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = %s
          AND tablename = ANY(%s)
        """,
        (schema_name, list(tables)),
    )
    found = {row[0] for row in cursor.fetchall()}
    return [table for table in tables if table in found]


def main():
    load_project_env(PROJECT_DIR, override=False)

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            for schema_name, tables in TABLES_BY_SCHEMA.items():
                existing = existing_tables(cursor, schema_name, tables)
                if existing:
                    qualified = ", ".join(f"{schema_name}.{table}" for table in existing)
                    cursor.execute(f"TRUNCATE {qualified} RESTART IDENTITY CASCADE")

    print("Base videe")


if __name__ == "__main__":
    main()
