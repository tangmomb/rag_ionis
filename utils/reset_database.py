import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT_DIR / "docker" / "postgres" / "init" / "001_schema.sql"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def reset_database(cursor):
    cursor.execute("DROP SCHEMA IF EXISTS chat CASCADE")
    print("[reset] schema chat supprime")
    cursor.execute(
        """
        DROP TABLE IF EXISTS
            public.comments,
            public.chunks,
            public.transcripts,
            public.video_transcripts,
            public.stats,
            public.video_stats,
            public.video_daily_stats,
            public.videos
        CASCADE
        """
    )
    cursor.execute("DROP SCHEMA IF EXISTS data CASCADE")
    cursor.execute("CREATE SCHEMA data")
    cursor.execute("GRANT USAGE ON SCHEMA data TO PUBLIC")
    cursor.execute("GRANT ALL ON SCHEMA data TO CURRENT_USER")
    print("[reset] schema data recree")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    cursor.execute(schema_sql)
    print(f"[reset] schema applique: {SCHEMA_PATH}")


def main():
    load_dotenv()
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            reset_database(cursor)
        connection.commit()
    print("Base SQL recreee depuis le schema")


if __name__ == "__main__":
    main()
