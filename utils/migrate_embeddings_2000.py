from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline.support.environment import load_project_env

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 2000
TEMPORARY_COLUMN = "embedding_2000"
BACKUP_COLUMN = "embedding_3072_backup"
HNSW_INDEX = "idx_chunks_embedding_hnsw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migre uniquement data.chunks.embedding de 3072 vers 2000 dimensions."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche l'etat de la base sans appel OpenAI ni modification.",
    )
    return parser.parse_args()


def column_type(cursor: psycopg.Cursor, column_name: str) -> str | None:
    cursor.execute(
        """
        SELECT format_type(attribute.atttypid, attribute.atttypmod)
        FROM pg_attribute attribute
        WHERE attribute.attrelid = 'data.chunks'::regclass
          AND attribute.attname = %s
          AND NOT attribute.attisdropped
        """,
        (column_name,),
    )
    row = cursor.fetchone()
    return str(row[0]) if row else None


def vector_literal(values: list[float]) -> str:
    if len(values) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"OpenAI a renvoye {len(values)} dimensions ; {EMBEDDING_DIMENSIONS} attendues."
        )
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def database_state(connection: psycopg.Connection) -> dict[str, object]:
    with connection.cursor() as cursor:
        current_type = column_type(cursor, "embedding")
        temporary_type = column_type(cursor, TEMPORARY_COLUMN)
        backup_type = column_type(cursor, BACKUP_COLUMN)
        cursor.execute("SELECT count(*) FROM data.chunks")
        total = int(cursor.fetchone()[0])
        cursor.execute("SELECT count(*) FROM data.chunks WHERE embedding IS NOT NULL")
        current_count = int(cursor.fetchone()[0])
        temporary_count = 0
        if temporary_type is not None:
            cursor.execute(f"SELECT count(*) FROM data.chunks WHERE {TEMPORARY_COLUMN} IS NOT NULL")
            temporary_count = int(cursor.fetchone()[0])
    return {
        "current_type": current_type,
        "temporary_type": temporary_type,
        "backup_type": backup_type,
        "total": total,
        "current_count": current_count,
        "temporary_count": temporary_count,
    }


def print_state(state: dict[str, object]) -> None:
    print(f"Colonne active: {state['current_type']} ({state['current_count']} embeddings)")
    print(f"Colonne temporaire: {state['temporary_type']} ({state['temporary_count']} embeddings)")
    print(f"Colonne de sauvegarde: {state['backup_type']}")
    print(f"Chunks: {state['total']}")


def prepare_temporary_column(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        active_type = column_type(cursor, "embedding")
        if active_type == "vector(2000)":
            return
        if active_type != "vector(3072)":
            raise RuntimeError(f"Type de colonne inattendu: {active_type}")
        if column_type(cursor, BACKUP_COLUMN) is not None:
            raise RuntimeError(
                f"La colonne {BACKUP_COLUMN} existe deja alors que la colonne active est en 3072."
            )
        cursor.execute(
            f"ALTER TABLE data.chunks ADD COLUMN IF NOT EXISTS {TEMPORARY_COLUMN} vector(2000)"
        )
        temporary_type = column_type(cursor, TEMPORARY_COLUMN)
        if temporary_type != "vector(2000)":
            raise RuntimeError(f"Type temporaire inattendu: {temporary_type}")
    connection.commit()


def migrate_batches(connection: psycopg.Connection, client: OpenAI, batch_size: int) -> None:
    while True:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, content
                FROM data.chunks
                WHERE {TEMPORARY_COLUMN} IS NULL
                ORDER BY id
                LIMIT %s
                """,
                (batch_size,),
            )
            rows = cursor.fetchall()
        if not rows:
            return

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            input=[str(row[1]) for row in rows],
        )
        embeddings = [item.embedding for item in response.data]
        if len(embeddings) != len(rows):
            raise RuntimeError(
                f"OpenAI a renvoye {len(embeddings)} embeddings pour {len(rows)} chunks."
            )
        updates = [
            (vector_literal(embedding), int(row[0]))
            for row, embedding in zip(rows, embeddings, strict=True)
        ]
        with connection.cursor() as cursor:
            cursor.executemany(
                f"UPDATE data.chunks SET {TEMPORARY_COLUMN} = %s::vector WHERE id = %s",
                updates,
            )
        connection.commit()
        print(f"[embed] {len(rows)} chunks migres (dernier id: {rows[-1][0]})", flush=True)


def validate_temporary_embeddings(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                count(*),
                count({TEMPORARY_COLUMN}),
                min(vector_dims({TEMPORARY_COLUMN})),
                max(vector_dims({TEMPORARY_COLUMN}))
            FROM data.chunks
            """
        )
        total, migrated, minimum, maximum = cursor.fetchone()
    if total != migrated or minimum != EMBEDDING_DIMENSIONS or maximum != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Validation impossible: total={total}, migres={migrated}, dimensions={minimum}..{maximum}"
        )
    print(f"[check] {migrated}/{total} embeddings valides en {minimum} dimensions")


def swap_columns(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        active_type = column_type(cursor, "embedding")
        if active_type == "vector(2000)":
            return
        if active_type != "vector(3072)":
            raise RuntimeError(f"Type actif inattendu avant permutation: {active_type}")
        if column_type(cursor, BACKUP_COLUMN) is not None:
            raise RuntimeError(f"La sauvegarde {BACKUP_COLUMN} existe deja.")
        cursor.execute(f"DROP INDEX IF EXISTS data.{HNSW_INDEX}")
        cursor.execute(f"ALTER TABLE data.chunks RENAME COLUMN embedding TO {BACKUP_COLUMN}")
        cursor.execute(f"ALTER TABLE data.chunks RENAME COLUMN {TEMPORARY_COLUMN} TO embedding")
        cursor.execute(
            "ALTER TABLE data.chunks ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER"
        )
        cursor.execute(
            """
            UPDATE data.chunks
            SET embedding_model = %s,
                embedding_dimensions = %s
            WHERE embedding IS NOT NULL
            """,
            (EMBEDDING_MODEL, EMBEDDING_DIMENSIONS),
        )
    connection.commit()
    print(f"[swap] embedding actif en 2000 ; sauvegarde conservee dans {BACKUP_COLUMN}")


def create_hnsw_index(database_url: str) -> None:
    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS {HNSW_INDEX}
                ON data.chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
    print(f"[index] {HNSW_INDEX} cree")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size doit etre strictement positif")
    load_project_env(PROJECT_DIR, override=True)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL manquante dans le fichier d'environnement selectionne")

    with psycopg.connect(database_url) as connection:
        state = database_state(connection)
        print_state(state)
        if args.dry_run:
            return 0
        if state["current_type"] == "vector(2000)":
            print("[skip] la colonne active est deja en 2000 dimensions")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY manquante dans .env")
            prepare_temporary_column(connection)
            migrate_batches(connection, OpenAI(api_key=api_key), args.batch_size)
            validate_temporary_embeddings(connection)
            swap_columns(connection)

    create_hnsw_index(database_url)
    with psycopg.connect(database_url) as connection:
        print_state(database_state(connection))
    print("[ok] migration terminee ; la sauvegarde 3072 n'a pas ete supprimee")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
