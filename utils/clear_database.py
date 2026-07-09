import os

import psycopg
from dotenv import load_dotenv


TABLES = (
    "comments",
    "transcripts",
    "video_transcripts",
    "stats",
    "video_stats",
    "video_daily_stats",
    "videos",
)


def existing_tables(cursor):
    cursor.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
          AND tablename = ANY(%s)
        """,
        (list(TABLES),),
    )
    found = {row[0] for row in cursor.fetchall()}
    return [table for table in TABLES if table in found]


def main():
    load_dotenv()

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            tables = existing_tables(cursor)
            if tables:
                cursor.execute(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE")

    print("Base videe")


if __name__ == "__main__":
    main()
