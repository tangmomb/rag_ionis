from __future__ import annotations

import os
import json
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
DOWNLOAD_ROOT = PROJECT_DIR / "downloads" / "youtube"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_TEXT_CHARS = 250_000
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


@app.get("/videos")
def videos_index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "videos.html")


def read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > 50 * 1024 * 1024:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")[:MAX_TEXT_CHARS]
    except (OSError, UnicodeDecodeError):
        return ""


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def video_directories() -> list[tuple[Path, Path]]:
    if not DOWNLOAD_ROOT.is_dir():
        return []
    videos: list[tuple[Path, Path]] = []
    for run_dir in sorted(DOWNLOAD_ROOT.glob("*_init*"), reverse=True):
        if not run_dir.is_dir():
            continue
        for video_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            if any(path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS for path in video_dir.iterdir()):
                videos.append((run_dir, video_dir))
    return videos


def video_output_paths(video_dir: Path) -> dict[str, Path | None]:
    transcript_dirs = sorted(path for path in (video_dir / "outputs").glob("transcripts*") if path.is_dir())
    return {
        "analysis": first_existing(
            [video_dir / "metadata" / "pipeline_analysis.json", video_dir / "outputs" / "metadata" / "pipeline_analysis.json"]
        ),
        "summary": first_existing([path / "video_summary.md" for path in transcript_dirs]),
        "transcript": first_existing([path / "plain_transcript.txt" for path in transcript_dirs]),
        "ocr": first_existing(
            [
                video_dir / "outputs" / "ocr" / "03_reviewed_ocr_overlays.json",
                video_dir / "outputs" / "ocr" / "02_filtered_ocr_overlays.json",
                video_dir / "outputs" / "ocr" / "01_processed_ocr_items.json",
            ]
        ),
        "chunks": first_existing(
            [
                video_dir / "outputs" / "chunks" / "transcript_chunks_speaker_validated.json",
                video_dir / "outputs" / "chunks" / "transcript_chunks.json",
            ]
        ),
    }


def chunk_items(path: Path | None) -> list[dict[str, Any]]:
    payload = read_json(path) if path else None
    values = payload.get("chunks") if isinstance(payload, dict) else payload
    return [value for value in values if isinstance(value, dict)] if isinstance(values, list) else []


def ocr_groups(path: Path | None) -> dict[str, Any]:
    payload = read_json(path) if path else None
    if isinstance(payload, dict) and isinstance(payload.get("kinds"), dict):
        return payload["kinds"]
    if isinstance(payload, dict):
        return {"items": payload.get("items", payload)}
    if isinstance(payload, list):
        return {"items": payload}
    return {}


def video_summary(run_dir: Path, video_dir: Path) -> dict[str, Any]:
    metadata = read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
    paths = video_output_paths(video_dir)
    analysis = read_json(paths["analysis"]) if paths["analysis"] else {}
    analysis = analysis if isinstance(analysis, dict) else {}
    images = [path for path in (video_dir / "outputs" / "images").rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    chunks = chunk_items(paths["chunks"])
    embedding_count = len(list((video_dir / "outputs" / "chunks").glob("*_embedding.json")))
    video_file = next((path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS), None)
    if embedding_count:
        stage = "Prête"
    elif chunks:
        stage = "Chunks"
    elif paths["summary"]:
        stage = "Résumé"
    elif paths["transcript"]:
        stage = "Transcript"
    elif paths["ocr"]:
        stage = "OCR"
    elif images:
        stage = "Images"
    else:
        stage = "Téléchargée"
    preview = images[0].relative_to(video_dir).as_posix() if images else None
    return {
        "run": run_dir.name,
        "id": str(metadata.get("youtube_video_id") or video_dir.name),
        "directory": video_dir.name,
        "title": str(metadata.get("title") or video_dir.name),
        "duration_seconds": metadata.get("duration_seconds") or metadata.get("duration"),
        "published_at": metadata.get("published_at") or metadata.get("publishedAt"),
        "url": metadata.get("url") or f"https://www.youtube.com/watch?v={video_dir.name}",
        "video_type": analysis.get("video_type"),
        "has_subtitles": analysis.get("has_subtitles"),
        "stage": stage,
        "image_count": len(images),
        "chunk_count": len(chunks),
        "embedding_count": embedding_count,
        "has_summary": bool(paths["summary"]),
        "has_transcript": bool(paths["transcript"]),
        "has_ocr": bool(paths["ocr"]),
        "preview_path": preview,
        "video_path": video_file.relative_to(video_dir).as_posix() if video_file else None,
    }


def selected_video(run_name: str, video_id: str) -> tuple[Path, Path]:
    run_dir = (DOWNLOAD_ROOT / run_name).resolve()
    try:
        run_dir.relative_to(DOWNLOAD_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Chemin refusé") from exc
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run introuvable")
    direct = run_dir / video_id
    if direct.is_dir():
        return run_dir, direct
    for video_dir in (path for path in run_dir.iterdir() if path.is_dir()):
        metadata = read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
        if str(metadata.get("youtube_video_id") or "") == video_id:
            return run_dir, video_dir
    raise HTTPException(status_code=404, detail="Vidéo introuvable")


@app.get("/api/videos")
def videos() -> list[dict[str, Any]]:
    return [video_summary(run_dir, video_dir) for run_dir, video_dir in video_directories()]


@app.get("/api/videos/{run_name}/{video_id}")
def video_detail(run_name: str, video_id: str) -> dict[str, Any]:
    run_dir, video_dir = selected_video(run_name, video_id)
    result = video_summary(run_dir, video_dir)
    paths = video_output_paths(video_dir)
    groups = ocr_groups(paths["ocr"])
    images = [
        {
            "path": path.relative_to(video_dir).as_posix(),
            "name": path.name,
            "category": path.parent.name,
        }
        for path in sorted((video_dir / "outputs" / "images").rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ][:500]
    files = [
        {"path": path.relative_to(video_dir).as_posix(), "size": path.stat().st_size}
        for path in sorted(video_dir.rglob("*"))
        if path.is_file() and ".embedding_cache" not in path.parts and path.suffix.lower() not in VIDEO_EXTENSIONS
    ][:1000]
    result.update(
        {
            "summary": read_text(paths["summary"]),
            "transcript": read_text(paths["transcript"]),
            "ocr": groups,
            "chunks": chunk_items(paths["chunks"])[:500],
            "images": images,
            "files": files,
        }
    )
    return result


@app.get("/api/videos/{run_name}/{video_id}/media")
def video_media(run_name: str, video_id: str, path: str = Query(..., min_length=1)) -> FileResponse:
    _run_dir, video_dir = selected_video(run_name, video_id)
    target = (video_dir / Path(path)).resolve()
    try:
        target.relative_to(video_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Chemin refusé") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return FileResponse(target)


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


@app.delete("/api/tables/{schema}/{table}")
def clear_table(schema: str, table: str) -> dict[str, Any]:
    if schema in {"pg_catalog", "information_schema"} or not allowed_table(schema, table):
        raise HTTPException(status_code=404, detail="Table introuvable")

    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM {};").format(quote_table(schema, table)))
        deleted = cur.rowcount
        conn.commit()
    return {"schema": schema, "table": table, "deleted": deleted}
