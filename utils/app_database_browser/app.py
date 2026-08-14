from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Literal

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


TRANSCRIPT_VARIANTS = (
    (
        "plain",
        "Transcript plain",
        "transcripts_whisper",
        ("transcript_plain.txt", "plain_transcript.txt"),
    ),
    (
        "raw_timecoded",
        "1 — Transcript brut",
        "transcripts_whisper",
        ("transcript_1_brut.txt", "whisper_transcript_timecoded.txt"),
    ),
    (
        "corrected_timecoded",
        "2 — Transcript corrigé",
        "transcripts_whisper",
        (
            "transcript_2_corrected.txt",
            "whisper_transcript_timecoded_corrected.txt",
        ),
    ),
    (
        "enriched",
        "3 — Transcript enrichi avec speakers",
        "transcripts_whisper",
        (
            "transcript_3_enriched.txt",
            "transcript_enriched.txt",
            "transcript_3_with_speakers.txt",
            "whisper_transcript_timecoded_corrected_enriched.txt",
        ),
    ),
    (
        "ocr_plain",
        "OCR plain utilisé pour les corrections",
        "transcripts_ocr",
        ("plain_transcript.txt",),
    ),
)


def transcript_variant_paths(video_dir: Path) -> list[dict[str, Any]]:
    variants = []
    seen_paths = set()
    for key, label, directory_name, names in TRANSCRIPT_VARIANTS:
        directory = video_dir / "outputs" / directory_name
        candidates = [directory / name for name in names]
        path = first_existing(candidates)
        if path is None or path.resolve() in seen_paths:
            continue
        seen_paths.add(path.resolve())
        variants.append({"key": key, "label": label, "path": path})
    return variants


def video_directories() -> list[tuple[Path, Path]]:
    if not DOWNLOAD_ROOT.is_dir():
        return []
    videos: list[tuple[Path, Path]] = []
    run_dirs = [
        path
        for path in DOWNLOAD_ROOT.iterdir()
        if path.is_dir() and (path.name == "init" or path.name.endswith("_init"))
    ]
    for run_dir in sorted(run_dirs, key=lambda path: (path.name == "init", path.name), reverse=True):
        if not run_dir.is_dir():
            continue
        for video_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            if any(path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS for path in video_dir.iterdir()):
                videos.append((run_dir, video_dir))
    return videos


def video_output_paths(video_dir: Path) -> dict[str, Path | None]:
    transcript_variants = transcript_variant_paths(video_dir)
    return {
        "manifest": first_existing(
            [
                video_dir / "metadata" / "video_manifest.json",
                video_dir / "metadata" / "pipeline_analysis.json",
                video_dir / "outputs" / "metadata" / "pipeline_analysis.json",
            ]
        ),
        "transcript": next(
            (
                variant["path"]
                for variant in transcript_variants
                if variant["key"] == "plain"
            ),
            transcript_variants[0]["path"] if transcript_variants else None,
        ),
        "ocr": first_existing(
            [
                video_dir / "outputs" / "ocr" / "02_filtered_ocr_overlays.json",
                video_dir / "outputs" / "ocr" / "01_processed_ocr_items.json",
            ]
        ),
        "chunks": first_existing(
            [
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


def video_speaker_details(video_dir: Path) -> list[dict[str, str | None]]:
    path = video_dir / "outputs" / "speakers" / "speakers_validated.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        return []

    raw_details = payload.get("speaker_details")
    if not isinstance(raw_details, list):
        raw_details = [
            {"speaker": name, "title": ""}
            for name in payload.get("speakers", [])
        ]

    details = []
    seen = set()
    for value in raw_details:
        if not isinstance(value, dict):
            continue
        name = " ".join(
            str(value.get("speaker") or value.get("name") or "").split()
        ).strip()
        title = " ".join(str(value.get("title") or "").split()).strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        details.append({"name": name, "title": title or None})
    return details


def video_speakers(video_dir: Path) -> list[str]:
    return [detail["name"] for detail in video_speaker_details(video_dir)]


def video_chunk_profile(video_dir: Path) -> str:
    existing_chunks = read_json(
        video_dir / "outputs" / "chunks" / "transcript_chunks.json"
    )
    if isinstance(existing_chunks, dict):
        chunking = existing_chunks.get("chunking")
        profile = chunking.get("profile") if isinstance(chunking, dict) else None
        if profile in {"short", "long"}:
            return profile

    from pipeline.context import LONG_VIDEO_THRESHOLD_SECONDS

    metadata = read_json(
        video_dir / "metadata" / "youtube_video_metadata.json"
    )
    duration = metadata.get("duration_seconds", 0) if isinstance(metadata, dict) else 0
    try:
        return (
            "long"
            if float(duration) > LONG_VIDEO_THRESHOLD_SECONDS
            else "short"
        )
    except (TypeError, ValueError):
        return "short"


class SpeakerEdit(BaseModel):
    name: str
    title: str | None = None


class SpeakersEdit(BaseModel):
    speakers: list[SpeakerEdit]


class EnrichedTranscriptEdit(BaseModel):
    content: str


def video_overview(run_dir: Path, video_dir: Path) -> dict[str, Any]:
    metadata = read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
    paths = video_output_paths(video_dir)
    manifest = read_json(paths["manifest"]) if paths["manifest"] else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    routing_facts = manifest.get("routing_facts")
    if isinstance(routing_facts, dict):
        facts = routing_facts
    else:
        facts = manifest
    images = [path for path in (video_dir / "outputs" / "images").rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    chunks = chunk_items(paths["chunks"])
    embedding_count = len(list((video_dir / "outputs" / "chunks").glob("*_embedding.json")))
    video_file = next((path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS), None)
    if embedding_count:
        stage = "Prête"
    elif chunks:
        stage = "Chunks"
    elif paths["transcript"]:
        stage = "Transcript"
    elif paths["ocr"]:
        stage = "OCR"
    elif images:
        stage = "Images"
    else:
        stage = "Téléchargée"
    preview = images[0].relative_to(video_dir).as_posix() if images else None
    speaker_details = video_speaker_details(video_dir)
    return {
        "run": run_dir.name,
        "id": str(metadata.get("youtube_video_id") or video_dir.name),
        "directory": video_dir.name,
        "title": str(metadata.get("title") or video_dir.name),
        "duration_seconds": metadata.get("duration_seconds") or metadata.get("duration"),
        "published_at": metadata.get("published_at") or metadata.get("publishedAt"),
        "url": metadata.get("url") or f"https://www.youtube.com/watch?v={video_dir.name}",
        "video_type": facts.get("video_type"),
        "has_subtitles": facts.get("has_subtitles"),
        "speakers": [detail["name"] for detail in speaker_details],
        "speaker_details": speaker_details,
        "stage": stage,
        "image_count": len(images),
        "chunk_count": len(chunks),
        "embedding_count": embedding_count,
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
    return [video_overview(run_dir, video_dir) for run_dir, video_dir in video_directories()]


@app.get("/api/videos/{run_name}/{video_id}")
def video_detail(run_name: str, video_id: str) -> dict[str, Any]:
    run_dir, video_dir = selected_video(run_name, video_id)
    result = video_overview(run_dir, video_dir)
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
    transcripts = [
        {
            "key": variant["key"],
            "label": variant["label"],
            "path": variant["path"].relative_to(video_dir).as_posix(),
            "content": read_text(variant["path"]),
            "editable": (
                variant["key"] == "enriched"
                and variant["path"].stat().st_size <= MAX_TEXT_CHARS
            ),
        }
        for variant in transcript_variant_paths(video_dir)
    ]
    result.update(
        {
            "transcript": read_text(paths["transcript"]),
            "transcripts": transcripts,
            "ocr": groups,
            "chunks": chunk_items(paths["chunks"])[:500],
            "images": images,
            "files": files,
        }
    )
    return result


@app.put("/api/videos/{run_name}/{video_id}/speakers")
def update_video_speakers(
    run_name: str,
    video_id: str,
    edit: SpeakersEdit,
) -> dict[str, Any]:
    _run_dir, video_dir = selected_video(run_name, video_id)
    if len(edit.speakers) > 50:
        raise HTTPException(status_code=422, detail="Maximum 50 speakers")

    details: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for speaker in edit.speakers:
        name = " ".join(speaker.name.split()).strip()
        title = " ".join((speaker.title or "").split()).strip()
        if not name:
            raise HTTPException(
                status_code=422,
                detail="Le nom d'un speaker ne peut pas etre vide",
            )
        if len(name) > 200 or len(title) > 300:
            raise HTTPException(
                status_code=422,
                detail="Nom ou fonction de speaker trop long",
            )
        key = name.casefold()
        if key in seen:
            raise HTTPException(
                status_code=422,
                detail=f"Speaker en double: {name}",
            )
        seen.add(key)
        details.append({"name": name, "title": title or None})

    target = video_dir / "outputs" / "speakers" / "speakers_validated.json"
    payload = read_json(target)
    if not isinstance(payload, dict):
        payload = {}
    payload["speakers"] = [detail["name"] for detail in details]
    payload["speaker_details"] = [
        {
            "speaker": detail["name"],
            "title": detail["title"] or "",
        }
        for detail in details
    ]
    write_text_atomic(
        target,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "path": target.relative_to(video_dir).as_posix(),
        "speakers": payload["speakers"],
        "speaker_details": details,
    }


@app.put("/api/videos/{run_name}/{video_id}/transcripts/enriched")
def update_enriched_transcript(
    run_name: str,
    video_id: str,
    edit: EnrichedTranscriptEdit,
) -> dict[str, Any]:
    _run_dir, video_dir = selected_video(run_name, video_id)
    encoded_size = len(edit.content.encode("utf-8"))
    if encoded_size > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Transcript limite a {MAX_TEXT_CHARS} octets",
        )
    if not edit.content.strip():
        raise HTTPException(
            status_code=422,
            detail="Le transcript enrichi ne peut pas etre vide",
        )

    enriched = next(
        (
            variant
            for variant in transcript_variant_paths(video_dir)
            if variant["key"] == "enriched"
        ),
        None,
    )
    if enriched is None:
        raise HTTPException(
            status_code=404,
            detail="Transcript enrichi introuvable",
        )
    target = enriched["path"]
    if target.stat().st_size > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail="Transcript trop volumineux pour etre edite ici",
        )

    write_text_atomic(target, edit.content)
    from pipeline.steps.transcripts.create_plain_transcript import convert_file
    from pipeline.steps.chunks.create_transcript_chunks import create_chunks

    plain_target = convert_file(video_dir, target, force=True)
    chunk_profile = video_chunk_profile(video_dir)
    chunks_target = create_chunks(
        video_dir,
        force=True,
        profile=chunk_profile,
        transcripts_dir_name=target.parent.name,
    )
    if chunks_target is None:
        raise HTTPException(
            status_code=500,
            detail="Le transcript est enregistre, mais les chunks n'ont pas pu etre regeneres",
        )
    return {
        "key": "enriched",
        "path": target.relative_to(video_dir).as_posix(),
        "content": edit.content,
        "editable": True,
        "plain": {
            "path": plain_target.relative_to(video_dir).as_posix(),
            "content": read_text(plain_target),
        },
        "chunks": {
            "path": chunks_target.relative_to(video_dir).as_posix(),
            "profile": chunk_profile,
            "items": chunk_items(chunks_target),
        },
    }


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
    sort_by: str | None = Query(default=None, max_length=200),
    sort_order: Literal["asc", "desc"] = Query(default="asc"),
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
        primary_key_name = (
            primary_key["column_name"] if primary_key else names[0]
        )
        if sort_by is not None and sort_by not in names:
            raise HTTPException(
                status_code=400,
                detail="Colonne de tri introuvable",
            )
        order_column = sort_by or primary_key_name
        order_direction = (
            sql.SQL("DESC") if sort_order == "desc" else sql.SQL("ASC")
        )
        order_column_type = next(
            column["type"]
            for column in columns
            if column["name"] == order_column
        ).lower()
        order_value = sql.Identifier(order_column)
        if order_column_type in {"json", "xml", "user-defined"}:
            order_value = sql.SQL("CAST({} AS text)").format(order_value)
        order_expressions = [
            sql.SQL("{} {} NULLS LAST").format(
                order_value,
                order_direction,
            )
        ]
        if order_column != primary_key_name:
            order_expressions.append(
                sql.SQL("{} ASC").format(sql.Identifier(primary_key_name))
            )

        cur.execute(
            sql.SQL("SELECT * FROM {}{} ORDER BY {} LIMIT %s OFFSET %s").format(
                quote_table(schema, table),
                where,
                sql.SQL(", ").join(order_expressions),
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
        "primary_key": primary_key_name,
        "sort_by": order_column,
        "sort_order": sort_order,
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
