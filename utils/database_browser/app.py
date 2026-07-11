from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from psycopg import sql
from psycopg.rows import dict_row


PROJECT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

app = FastAPI(title="RAG IONIS — Database browser")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


def database_url() -> str:
    value = os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL manquant dans .env")
    return value


def connection():
    return psycopg.connect(database_url(), row_factory=dict_row)


def quote_table(schema: str, table: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))


def allowed_table(schema: str, table: str) -> bool:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
              AND table_type = 'BASE TABLE'
            """,
            (schema, table),
        )
        return cur.fetchone() is not None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT current_database() AS database, now() AS server_time")
            return {"ok": True, **cur.fetchone()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/api/tables")
def tables() -> list[dict[str, Any]]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.table_schema AS schema, t.table_name AS name,
                   COALESCE(s.n_live_tup, 0)::bigint AS estimated_rows
            FROM information_schema.tables t
            LEFT JOIN pg_stat_user_tables s
              ON s.schemaname = t.table_schema AND s.relname = t.table_name
            WHERE t.table_type = 'BASE TABLE'
              AND t.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY t.table_schema, t.table_name
            """
        )
        return cur.fetchall()


@app.get("/api/tables/{schema}/{table}")
def table_data(
    schema: str,
    table: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=200),
) -> dict[str, Any]:
    if not allowed_table(schema, table):
        raise HTTPException(status_code=404, detail="Table introuvable")

    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.column_name AS name, c.data_type AS type,
                   c.is_nullable = 'YES' AS nullable,
                   c.ordinal_position AS position
            FROM information_schema.columns c
            WHERE c.table_schema = %s AND c.table_name = %s
            ORDER BY c.ordinal_position
            """,
            (schema, table),
        )
        columns = cur.fetchall()
        if not columns:
            raise HTTPException(status_code=404, detail="Table vide ou introuvable")

        names = [column["name"] for column in columns]
        text_columns = [
            column["name"]
            for column in columns
            if column["type"] in {"text", "character varying", "character", "json", "jsonb"}
        ]
        where = sql.SQL("")
        params: list[Any] = []
        if q.strip() and text_columns:
            where = sql.SQL(" WHERE ") + sql.SQL(" OR ").join(
                sql.SQL("CAST({} AS text) ILIKE %s").format(sql.Identifier(name))
                for name in text_columns
            )
            params.extend([f"%{q.strip()}%"] * len(text_columns))

        cur.execute(
            sql.SQL("SELECT COUNT(*) AS count FROM {}{}").format(quote_table(schema, table), where),
            params,
        )
        total = cur.fetchone()["count"]

        cur.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
             AND tc.table_name = kcu.table_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %s AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
            LIMIT 1
            """,
            (schema, table),
        )
        primary_key = cur.fetchone()
        order_column = primary_key["column_name"] if primary_key else names[0]

        cur.execute(
            sql.SQL("SELECT * FROM {}{} ORDER BY {} LIMIT %s OFFSET %s").format(
                quote_table(schema, table), where, sql.Identifier(order_column)
            ),
            [*params, limit, offset],
        )
        rows = cur.fetchall()

    return {
        "schema": schema,
        "table": table,
        "columns": columns,
        "rows": rows,
        "total": total,
        "limit": limit,
        "offset": offset,
        "primary_key": order_column,
        "searchable": bool(text_columns),
    }

