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
    cursor.execute("DROP SCHEMA IF EXISTS public CASCADE")
    cursor.execute("CREATE SCHEMA public")
    cursor.execute("GRANT USAGE ON SCHEMA public TO PUBLIC")
    cursor.execute("GRANT ALL ON SCHEMA public TO CURRENT_USER")
    print("[reset] schema public recree")

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
