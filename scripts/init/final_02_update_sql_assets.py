import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from common.pipeline_paths import analysed_infos_path, consolidate_init_dir, existing_chunks_dir, existing_transcripts_dir, existing_youtube_api_infos_path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DEFAULT_S3_ROOT_PREFIX = "youtube"
DEFAULT_S3_BUCKET_NAME = ""
DATA_SCHEMA = "data"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 2000
SCHEMA_PATH = ROOT_DIR / "docker" / "postgres" / "init" / "001_schema.sql"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
PLAIN_TRANSCRIPT_NAME = "plain_transcript.txt"
WHISPER_TRANSCRIPT_TIMECODED_NAME = "whisper_transcript_timecoded.txt"
WHISPER_TRANSCRIPT_TIMECODED_CORRECTED_NAME = "whisper_transcript_timecoded_corrected.txt"
WHISPER_TRANSCRIPT_ENRICHED_NAME = "whisper_transcript_timecoded_corrected_enriched.txt"
OCR_SUBTITLE_NAME = "ocr_subtitles.txt"
OCR_SUBTITLE_TIMECODED_NAME = "ocr_subtitles_timecoded.txt"
OCR_SUBTITLE_TIMECODED_CORRECTED_NAME = "ocr_subtitles_timecoded_corrected.txt"
OCR_SUBTITLE_ENRICHED_NAME = "ocr_subtitles_timecoded_corrected_enriched.txt"
LEGACY_ENRICHED_TRANSCRIPT_SUFFIX = "_transcript_timecodes_corrected_enrichi.txt"
LEGACY_TIMECODED_TRANSCRIPT_SUFFIX = "_transcript_timecodes_corrected.txt"
LEGACY_UNCORRECTED_TIMECODED_TRANSCRIPT_SUFFIX = "_transcript_timecodes.txt"
LEGACY_PLAIN_TRANSCRIPT_SUFFIX = "_transcript.txt"
LEGACY_OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
LEGACY_OCR_SUBTITLE_TIMECODED_SUFFIX = "_ocr_subtitle_timecodes.txt"
LEGACY_OCR_SUBTITLE_TIMECODED_CORRECTED_SUFFIX = "_ocr_subtitle_timecodes_corrected.txt"
LEGACY_OCR_SUBTITLE_ENRICHED_SUFFIX = "_ocr_subtitle_timecodes_corrected_enrichi.txt"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def video_files(video_dir):
    direct_videos = []
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            direct_videos.append(path)

    if direct_videos:
        yield from direct_videos
        return

    for child in sorted(video_dir.iterdir()):
        if not child.is_dir():
            continue
        for path in sorted(child.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


def latest_video_dir(parent_dir):
    candidate = consolidate_init_dir(parent_dir)
    if not candidate.is_dir() or not any(video_files(candidate)):
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidate


def normalize_prefix(prefix):
    return prefix.strip("/") if prefix else ""


def default_prefix(video_dir):
    return normalize_prefix(f"{DEFAULT_S3_ROOT_PREFIX}/{video_dir.name}")


def ensure_schema(cursor):
    ensure_stats_table_name(cursor)
    ensure_transcripts_table_name(cursor)
    for table_name in ("videos", "stats", "transcripts", "chunks"):
        ensure_data_collected_date_column(cursor, table_name)

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id BIGSERIAL PRIMARY KEY,
            youtube_video_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            description TEXT,
            url TEXT NOT NULL,
            duration_seconds INTEGER,
            thumbnail_medium_url TEXT,
            has_subtitles BOOLEAN,
            video_type TEXT,
            speakers TEXT[],
            s3_uri TEXT,
            published_at TIMESTAMPTZ,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS thumbnail_medium_url TEXT")
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS has_subtitles BOOLEAN")
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS video_type TEXT")
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS speakers TEXT[]")
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS s3_uri TEXT")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS updated_at")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS s3_bucket")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS s3_prefix")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS video_summary")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS raw_json")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stats (
            id BIGSERIAL PRIMARY KEY,
            video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            view_count BIGINT,
            like_count BIGINT,
            comment_count BIGINT,
            snapshot_date DATE NOT NULL,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (video_id, snapshot_date)
        )
        """
    )
    cursor.execute("ALTER TABLE stats DROP COLUMN IF EXISTS raw_json")
    if table_exists(cursor, "comments"):
        cursor.execute("ALTER TABLE comments DROP COLUMN IF EXISTS raw_json")
    cursor.execute("DROP TABLE IF EXISTS video_elements CASCADE")
    cursor.execute(f"DROP TABLE IF EXISTS {legacy_video_elements_name()} CASCADE")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_snapshot_date ON stats(snapshot_date)")
    ensure_chunks_schema(cursor)
    ensure_transcripts_schema(cursor)


def table_columns(cursor, table_name):
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        """,
        (DATA_SCHEMA, table_name),
    )
    return {row[0] for row in cursor.fetchall()}


def table_exists(cursor, table_name):
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
        """,
        (DATA_SCHEMA, table_name),
    )
    return cursor.fetchone()[0]


def reset_data_schema(cursor):
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
    cursor.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def ensure_stats_table_name(cursor):
    if table_exists(cursor, "video_daily_stats") and not table_exists(cursor, "stats") and not table_exists(cursor, "video_stats"):
        cursor.execute("ALTER TABLE video_daily_stats RENAME TO stats")
    if table_exists(cursor, "video_stats") and not table_exists(cursor, "stats"):
        cursor.execute("ALTER TABLE video_stats RENAME TO stats")
    if constraint_exists(cursor, "video_daily_stats_video_id_snapshot_date_key"):
        cursor.execute(
            "ALTER TABLE stats RENAME CONSTRAINT video_daily_stats_video_id_snapshot_date_key TO stats_video_id_snapshot_date_key"
        )
    if constraint_exists(cursor, "video_stats_video_id_snapshot_date_key"):
        cursor.execute(
            "ALTER TABLE stats RENAME CONSTRAINT video_stats_video_id_snapshot_date_key TO stats_video_id_snapshot_date_key"
        )
    if constraint_exists(cursor, "video_daily_stats_pkey"):
        cursor.execute("ALTER TABLE stats RENAME CONSTRAINT video_daily_stats_pkey TO stats_pkey")
    if constraint_exists(cursor, "video_stats_pkey"):
        cursor.execute("ALTER TABLE stats RENAME CONSTRAINT video_stats_pkey TO stats_pkey")
    if constraint_exists(cursor, "video_daily_stats_video_id_fkey"):
        cursor.execute("ALTER TABLE stats RENAME CONSTRAINT video_daily_stats_video_id_fkey TO stats_video_id_fkey")
    if constraint_exists(cursor, "video_stats_video_id_fkey"):
        cursor.execute("ALTER TABLE stats RENAME CONSTRAINT video_stats_video_id_fkey TO stats_video_id_fkey")
    cursor.execute("DROP INDEX IF EXISTS idx_video_daily_stats_snapshot_date")
    cursor.execute("DROP INDEX IF EXISTS idx_video_stats_snapshot_date")


def ensure_transcripts_table_name(cursor):
    if table_exists(cursor, "video_transcripts") and not table_exists(cursor, "transcripts"):
        cursor.execute("ALTER TABLE video_transcripts RENAME TO transcripts")
    if constraint_exists(cursor, "video_transcripts_pkey"):
        cursor.execute("ALTER TABLE transcripts RENAME CONSTRAINT video_transcripts_pkey TO transcripts_pkey")
    if constraint_exists(cursor, "video_transcripts_video_id_fkey"):
        cursor.execute("ALTER TABLE transcripts RENAME CONSTRAINT video_transcripts_video_id_fkey TO transcripts_video_id_fkey")
    if constraint_exists(cursor, "video_transcripts_video_id_language_code_key"):
        cursor.execute(
            "ALTER TABLE transcripts RENAME CONSTRAINT video_transcripts_video_id_language_code_key TO transcripts_video_id_language_code_key"
        )
    if constraint_exists(cursor, "video_transcripts_video_id_language_code_transcript_type_key"):
        cursor.execute(
            "ALTER TABLE transcripts RENAME CONSTRAINT video_transcripts_video_id_language_code_transcript_type_key TO transcripts_video_id_language_code_transcript_type_key"
        )
    cursor.execute("DROP INDEX IF EXISTS idx_video_transcripts_video_id")


def ensure_chunks_schema(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY,
            video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            speakers TEXT[],
            embedding_model TEXT,
            embedding_dimensions INTEGER,
            embedding vector(2000),
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (video_id, chunk_index)
        )
        """
    )
    ensure_data_collected_date_column(cursor, "chunks")
    cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS speakers TEXT[]")
    cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT")
    cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER")
    cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(2000)")
    cursor.execute(
        """
        SELECT format_type(attribute.atttypid, attribute.atttypmod)
        FROM pg_attribute attribute
        WHERE attribute.attrelid = 'chunks'::regclass
          AND attribute.attname = 'embedding'
          AND NOT attribute.attisdropped
        """
    )
    embedding_type = cursor.fetchone()[0]
    if embedding_type != "vector(2000)":
        cursor.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
        populated_embeddings = int(cursor.fetchone()[0])
        if populated_embeddings:
            raise RuntimeError(
                f"La colonne chunks.embedding est en {embedding_type} avec "
                f"{populated_embeddings} embeddings. Regenere les embeddings en 2000 dimensions "
                "puis relance avec --reset-database."
            )
        cursor.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
        cursor.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(2000)")
    cursor.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS char_count")
    cursor.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS alert")
    cursor.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS alert_reason")
    cursor.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS source_file")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks(video_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index)")
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        """
    )


def legacy_video_elements_name():
    return "video_" + "assets"


def constraint_exists(cursor, constraint_name):
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_constraint constraint_info
            JOIN pg_namespace schema_info
              ON schema_info.oid = constraint_info.connamespace
            WHERE schema_info.nspname = %s
              AND constraint_info.conname = %s
        )
        """,
        (DATA_SCHEMA, constraint_name),
    )
    return cursor.fetchone()[0]


def ensure_data_collected_date_column(cursor, table_name):
    columns = table_columns(cursor, table_name)
    legacy_column = "created" + "_at"
    if legacy_column in columns and "data_collected_date" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} RENAME COLUMN {legacy_column} TO data_collected_date")
    elif legacy_column in columns and "data_collected_date" in columns:
        cursor.execute(
            f"""
            UPDATE {table_name}
            SET data_collected_date = COALESCE(data_collected_date, {legacy_column})
            """
        )
        cursor.execute(f"ALTER TABLE {table_name} DROP COLUMN {legacy_column}")


def ensure_transcripts_schema(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transcripts (
            id BIGSERIAL PRIMARY KEY,
            video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            language_code TEXT NOT NULL,
            transcript TEXT,
            transcript_timecodes TEXT,
            transcript_timecodes_enrichi TEXT,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    ensure_data_collected_date_column(cursor, "transcripts")
    cursor.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS transcript TEXT")
    cursor.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS transcript_timecodes TEXT")
    cursor.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS transcript_timecodes_enrichi TEXT")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS updated_at")

    columns = table_columns(cursor, "transcripts")
    if {"transcript_plain", "transcript_timecoded", "transcript_enriched"} & columns or {"transcript_type", "text"}.issubset(columns):
        plain_expression = "MAX(transcript)" if "transcript" in columns else "NULL::text"
        if "transcript_plain" in columns:
            plain_expression = f"COALESCE({plain_expression}, MAX(transcript_plain))"
        if {"transcript_type", "text"}.issubset(columns):
            plain_expression = f"COALESCE({plain_expression}, MAX(text) FILTER (WHERE transcript_type = 'plain'))"

        timecodes_expression = "MAX(transcript_timecodes)" if "transcript_timecodes" in columns else "NULL::text"
        if "transcript_timecoded" in columns:
            timecodes_expression = f"COALESCE({timecodes_expression}, MAX(transcript_timecoded))"
        if {"transcript_type", "text"}.issubset(columns):
            timecodes_expression = f"COALESCE({timecodes_expression}, MAX(text) FILTER (WHERE transcript_type = 'timecoded'))"

        enriched_expression = "MAX(transcript_timecodes_enrichi)" if "transcript_timecodes_enrichi" in columns else "NULL::text"
        if "transcript_enriched" in columns:
            enriched_expression = f"COALESCE({enriched_expression}, MAX(transcript_enriched))"
        if {"transcript_type", "text"}.issubset(columns):
            enriched_expression = f"COALESCE({enriched_expression}, MAX(text) FILTER (WHERE transcript_type = 'enriched'))"

        cursor.execute(
            f"""
            CREATE TEMP TABLE transcripts_merged ON COMMIT DROP AS
            SELECT
                MIN(id) AS id,
                video_id,
                language_code,
                {plain_expression} AS transcript,
                {timecodes_expression} AS transcript_timecodes,
                {enriched_expression} AS transcript_timecodes_enrichi,
                MIN(data_collected_date) AS data_collected_date
            FROM transcripts
            GROUP BY video_id, language_code
            """
        )
        cursor.execute("ALTER TABLE transcripts DROP CONSTRAINT IF EXISTS transcripts_video_id_language_code_transcript_type_key")
        cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_type")
        cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS text")
        cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS segments")
        cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_plain")
        cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_timecoded")
        cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_enriched")
        cursor.execute("TRUNCATE transcripts")
        cursor.execute(
            """
            INSERT INTO transcripts (
                id,
                video_id,
                language_code,
                transcript,
                transcript_timecodes,
                transcript_timecodes_enrichi,
                data_collected_date
            )
            SELECT
                id,
                video_id,
                language_code,
                transcript,
                transcript_timecodes,
                transcript_timecodes_enrichi,
                data_collected_date
            FROM transcripts_merged
            """
        )
        cursor.execute(
            """
            SELECT setval(
                pg_get_serial_sequence('transcripts', 'id'),
                COALESCE((SELECT MAX(id) FROM transcripts), 1),
                (SELECT COUNT(*) > 0 FROM transcripts)
            )
            """
        )

    cursor.execute("ALTER TABLE transcripts DROP CONSTRAINT IF EXISTS transcripts_video_id_language_code_key")
    cursor.execute("ALTER TABLE transcripts DROP CONSTRAINT IF EXISTS transcripts_video_id_language_code_transcript_type_key")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_type")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS text")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS segments")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_plain")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_timecoded")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_enriched")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS video_summary")
    cursor.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'transcripts_video_id_language_code_key'
            ) THEN
                ALTER TABLE transcripts
                    ADD CONSTRAINT transcripts_video_id_language_code_key
                    UNIQUE (video_id, language_code);
            END IF;
        END $$;
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id)")


def video_db_ids(cursor):
    cursor.execute("SELECT youtube_video_id, id FROM videos")
    return dict(cursor.fetchall())


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as error:
        print(f"[skip] json invalide: {path} ({error})")
        return None


def parse_datetime(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def parse_int(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_video_statistics(payload):
    if not isinstance(payload, dict):
        return None
    statistics = payload.get("statistics")
    if isinstance(statistics, dict):
        return statistics
    raw_json = payload.get("raw_json")
    if isinstance(raw_json, dict):
        fallback = raw_json.get("statistics")
        if isinstance(fallback, dict):
            return fallback
    return None


def extract_thumbnail_medium_url(payload):
    if not isinstance(payload, dict):
        return None
    thumbnail_medium_url = payload.get("thumbnail_medium_url")
    if thumbnail_medium_url:
        return thumbnail_medium_url
    raw_json = payload.get("raw_json")
    if isinstance(raw_json, dict):
        return (
            raw_json.get("snippet", {})
            .get("thumbnails", {})
            .get("medium", {})
            .get("url")
        )
    return None


def load_chunks_payload(video_path):
    chunks_dir = existing_chunks_dir(video_path)
    candidates = (
        chunks_dir / "transcript_chunks.json",
    )
    for candidate in candidates:
        payload = load_json(candidate)
        if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
            return payload, candidate
    return None, None


def load_chunk_embedding_payload(video_path, chunk_index):
    if chunk_index is None:
        return None
    chunks_dir = existing_chunks_dir(video_path)
    candidate = chunks_dir / f"chunk_{int(chunk_index):02d}_embedding.json"
    if candidate.exists():
        payload = load_json(candidate)
        if isinstance(payload, dict):
            return payload
    return None


def embedding_vector_literal(values):
    if not isinstance(values, list) or not values:
        return None
    if len(values) != DEFAULT_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding de {len(values)} dimensions recu ; "
            f"{DEFAULT_EMBEDDING_DIMENSIONS} attendues. Regenere les embeddings du pipeline."
        )
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def candidate_video_dirs(video_dir, video_ids=None):
    direct_videos = [path for path in sorted(video_dir.iterdir()) if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    if direct_videos:
        candidates = [video_dir]
    else:
        candidates = [path for path in sorted(video_dir.iterdir()) if path.is_dir()]
    selected_ids = set(video_ids or [])
    return [path for path in candidates if not selected_ids or path.name in selected_ids]


def local_video_identifier(video_path):
    path = Path(video_path)
    return path.name if path.is_dir() else path.stem


def load_video_metadata(video_path):
    target = existing_youtube_api_infos_path(video_path)
    if target.exists():
        return load_json(target)
    return None


def load_video_analysis(video_path):
    target = analysed_infos_path(video_path)
    if target.exists():
        return load_json(target)
    return None


def load_video_speakers(video_path):
    chunks_dir = video_path / "outputs" / "chunks"
    candidates = (
        chunks_dir / "transcript_chunks.json",
    )
    for candidate in candidates:
        payload = load_json(candidate)
        if not isinstance(payload, dict):
            continue
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            continue
        speakers = []
        seen = set()
        for chunk in chunks:
            meta_data = chunk.get("meta_data", {}) if isinstance(chunk, dict) else {}
            names = meta_data.get("speakers", []) if isinstance(meta_data, dict) else []
            if not isinstance(names, list):
                continue
            for name in names:
                normalized = " ".join(str(name).split()).strip()
                if not normalized or normalized.casefold() in seen:
                    continue
                seen.add(normalized.casefold())
                speakers.append(normalized)
        if speakers:
            return speakers
    return None


def upsert_chunk(cursor, video_id, chunk_payload, embedding_payload=None):
    chunk_index = parse_int(chunk_payload.get("chunk_index"))
    content = str(chunk_payload.get("content") or "").strip()
    if chunk_index is None or not content:
        return False

    meta_data = chunk_payload.get("meta_data", {}) if isinstance(chunk_payload.get("meta_data"), dict) else {}
    raw_speakers = meta_data.get("speakers", [])
    speakers = [str(name).strip() for name in raw_speakers if str(name).strip()] if isinstance(raw_speakers, list) else None
    embedding_model = embedding_payload.get("model") if isinstance(embedding_payload, dict) else None
    embedding_values = embedding_payload.get("embedding") if isinstance(embedding_payload, dict) else None
    embedding_literal = embedding_vector_literal(embedding_values)
    embedding_dimensions = len(embedding_values) if embedding_literal is not None else None
    if embedding_literal is not None and embedding_model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Modele d'embedding {embedding_model!r} recu ; {DEFAULT_EMBEDDING_MODEL!r} attendu."
        )

    if embedding_literal is not None:
        cursor.execute(
            """
            INSERT INTO chunks (
                video_id,
                chunk_index,
                content,
                speakers,
                embedding_model,
                embedding_dimensions,
                embedding,
                data_collected_date
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector, now())
            ON CONFLICT (video_id, chunk_index) DO UPDATE SET
                content = EXCLUDED.content,
                speakers = EXCLUDED.speakers,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dimensions = EXCLUDED.embedding_dimensions,
                embedding = EXCLUDED.embedding,
                data_collected_date = now()
            """,
            (
                video_id,
                chunk_index,
                content,
                speakers,
                embedding_model,
                embedding_dimensions,
                embedding_literal,
            ),
        )
    else:
        cursor.execute(
            """
            INSERT INTO chunks (
                video_id,
                chunk_index,
                content,
                speakers,
                embedding_model,
                data_collected_date
            )
            VALUES (%s, %s, %s, %s, %s, now())
            ON CONFLICT (video_id, chunk_index) DO UPDATE SET
                content = EXCLUDED.content,
                speakers = EXCLUDED.speakers,
                embedding_model = EXCLUDED.embedding_model,
                data_collected_date = now()
            """,
            (
                video_id,
                chunk_index,
                content,
                speakers,
                embedding_model,
            ),
        )
    return True


def video_storage_prefix(video_path, root_dir, prefix):
    relative_path = video_path.relative_to(root_dir).as_posix()
    if prefix:
        return f"{prefix}/{relative_path}"
    return relative_path


def upsert_video(cursor, video_path, root_dir, bucket, prefix):
    payload = load_video_metadata(video_path) or {}
    analysis = load_video_analysis(video_path) or {}
    youtube_video_id = payload.get("youtube_video_id") or local_video_identifier(video_path)
    title = payload.get("title") or youtube_video_id
    description = payload.get("description")
    url = payload.get("url") or f"https://www.youtube.com/watch?v={youtube_video_id}"
    published_at = parse_datetime(payload.get("published_at"))
    duration_seconds = parse_int(payload.get("duration_seconds"))
    thumbnail_medium_url = extract_thumbnail_medium_url(payload)
    has_subtitles = analysis.get("has_subtitles") if isinstance(analysis.get("has_subtitles"), bool) else None
    video_type = analysis.get("video_type") if isinstance(analysis.get("video_type"), str) else None
    speakers = load_video_speakers(video_path)
    s3_prefix = video_storage_prefix(video_path, root_dir, prefix)
    s3_uri = f"s3://{bucket}/{s3_prefix}" if bucket and s3_prefix else s3_prefix or None

    cursor.execute(
        """
        INSERT INTO videos (
            youtube_video_id,
            title,
            description,
            url,
            published_at,
            duration_seconds,
            thumbnail_medium_url,
            has_subtitles,
            video_type,
            speakers,
            s3_uri
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (youtube_video_id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            url = EXCLUDED.url,
            published_at = EXCLUDED.published_at,
            duration_seconds = EXCLUDED.duration_seconds,
            thumbnail_medium_url = EXCLUDED.thumbnail_medium_url,
            has_subtitles = EXCLUDED.has_subtitles,
            video_type = EXCLUDED.video_type,
            speakers = EXCLUDED.speakers,
            s3_uri = EXCLUDED.s3_uri
        RETURNING id
        """,
        (
            youtube_video_id,
            title,
            description,
            url,
            published_at,
            duration_seconds,
            thumbnail_medium_url,
            has_subtitles,
            video_type,
            speakers,
            s3_uri,
        ),
    )
    return youtube_video_id, cursor.fetchone()[0], payload


def upsert_video_stats(cursor, video_id, payload, snapshot_date=None):
    statistics = extract_video_statistics(payload)
    if not isinstance(statistics, dict):
        return False

    cursor.execute(
        """
        INSERT INTO stats (
            video_id,
            snapshot_date,
            view_count,
            like_count,
            comment_count
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id, snapshot_date) DO UPDATE SET
            view_count = EXCLUDED.view_count,
            like_count = EXCLUDED.like_count,
            comment_count = EXCLUDED.comment_count
        """,
        (
            video_id,
            snapshot_date or date.today(),
            parse_int(statistics.get("viewCount")),
            parse_int(statistics.get("likeCount")),
            parse_int(statistics.get("commentCount")),
        ),
    )
    return True


def video_folder_name(path, root_dir):
    relative_parts = path.relative_to(root_dir).parts
    if len(relative_parts) < 2:
        return path.stem
    return relative_parts[0]


def transcript_paths(video_dir):
    transcript_dir = existing_transcripts_dir(video_dir)
    if not transcript_dir.exists():
        return []

    def first_existing(names=(), patterns=()):
        for name in names:
            candidate = transcript_dir / name
            if candidate.exists():
                return candidate
        for pattern in patterns:
            matches = sorted(transcript_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    candidates = (
        ("plain", (PLAIN_TRANSCRIPT_NAME,), (f"*{LEGACY_PLAIN_TRANSCRIPT_SUFFIX}",)),
        (
            "timecoded",
            (
                WHISPER_TRANSCRIPT_TIMECODED_CORRECTED_NAME,
                OCR_SUBTITLE_TIMECODED_CORRECTED_NAME,
                WHISPER_TRANSCRIPT_TIMECODED_NAME,
                OCR_SUBTITLE_TIMECODED_NAME,
            ),
            (
                f"*{LEGACY_TIMECODED_TRANSCRIPT_SUFFIX}",
                f"*{LEGACY_OCR_SUBTITLE_TIMECODED_CORRECTED_SUFFIX}",
                f"*{LEGACY_OCR_SUBTITLE_TIMECODED_SUFFIX}",
                f"*{LEGACY_UNCORRECTED_TIMECODED_TRANSCRIPT_SUFFIX}",
            ),
        ),
        (
            "enriched",
            (WHISPER_TRANSCRIPT_ENRICHED_NAME, OCR_SUBTITLE_ENRICHED_NAME),
            (f"*{LEGACY_ENRICHED_TRANSCRIPT_SUFFIX}", f"*{LEGACY_OCR_SUBTITLE_ENRICHED_SUFFIX}"),
        ),
    )
    found = []
    for transcript_type, names, patterns in candidates:
        candidate = first_existing(names, patterns)
        if candidate:
            found.append((transcript_type, candidate))
    return found


def upsert_transcript(cursor, video_id, transcript_type, transcript_path):
    text = transcript_path.read_text(encoding="utf-8").strip()
    if not text:
        return False
    column_by_type = {
        "plain": "transcript",
        "timecoded": "transcript_timecodes",
        "enriched": "transcript_timecodes_enrichi",
    }
    column = column_by_type[transcript_type]

    cursor.execute(
        f"""
        INSERT INTO transcripts (
            video_id,
            language_code,
            {column}
        )
        VALUES (%s, 'fr', %s)
        ON CONFLICT (video_id, language_code) DO UPDATE SET
            {column} = EXCLUDED.{column}
        """,
        (video_id, text),
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Met a jour la base SQL avec les fichiers locaux/S3 et les transcripts generes."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier traite. Defaut: dernier sous-dossier de downloads/youtube",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent utilise si --video-dir est absent. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_S3_BUCKET_NAME,
        help="Nom du bucket S3. Defaut: aucun",
    )
    parser.add_argument(
        "--prefix",
        help="Prefixe S3. Defaut: youtube/nom_du_dossier_traite, comme la Step final_01",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Considere que les fichiers sont a la racine du bucket.",
    )
    parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="N'actualise pas la table transcripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les changements sans modifier la base.",
    )
    parser.add_argument(
        "--clean-init-assets",
        action="store_true",
        help="Option legacy sans effet. Les assets detailles ne sont plus stockes en SQL.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        help="Limite la mise a jour a cet ID video. Option repetable.",
    )
    parser.add_argument(
        "--reset-database",
        action="store_true",
        help="Supprime et recree le schema data avant la mise a jour.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable: {video_dir}")

    if args.no_prefix:
        prefix = ""
    else:
        prefix = normalize_prefix(args.prefix) if args.prefix is not None else default_prefix(video_dir)

    video_dirs = candidate_video_dirs(video_dir, video_ids=args.video_id)
    files = sorted(
        path
        for current_video_dir in video_dirs
        for path in current_video_dir.rglob("*")
        if path.is_file()
    )

    print(f"Dossier source: {video_dir}")
    print(f"Bucket S3: {args.bucket or '(aucun)'}")
    print(f"Prefixe S3: {prefix or '(racine)'}")

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            if args.reset_database:
                reset_data_schema(cursor)
                print("[clean] schema data supprime et recree.")
            else:
                cursor.execute("CREATE SCHEMA IF NOT EXISTS data")
            cursor.execute("SET search_path TO data, public")
            ensure_schema(cursor)

            videos_count = 0
            stats_count = 0
            for current_video_dir in video_dirs:
                youtube_video_id, video_id, metadata_payload = upsert_video(
                    cursor,
                    current_video_dir,
                    video_dir,
                    args.bucket,
                    prefix,
                )
                videos_count += 1
                print(f"[video] {youtube_video_id}")
                if upsert_video_stats(cursor, video_id, metadata_payload):
                    stats_count += 1

            ids_by_youtube_id = video_db_ids(cursor)
            assets_count = 0
            transcripts_count = 0
            chunks_count = 0
            skipped = []

            for path in files:
                youtube_video_id = video_folder_name(path, video_dir)
                video_id = ids_by_youtube_id.get(youtube_video_id)
                if not video_id:
                    skipped.append(path)
                    continue
                assets_count += 1

            if not args.skip_transcripts:
                for current_video_dir in video_dirs:
                    video_id = ids_by_youtube_id.get(current_video_dir.name)
                    if not video_id:
                        continue
                    chunks_payload, chunks_source = load_chunks_payload(current_video_dir)
                    if chunks_payload and chunks_source:
                        for chunk_payload in chunks_payload.get("chunks", []):
                            embedding_payload = load_chunk_embedding_payload(current_video_dir, chunk_payload.get("chunk_index"))
                            if not args.dry_run and upsert_chunk(
                                cursor,
                                video_id,
                                chunk_payload,
                                embedding_payload=embedding_payload,
                            ):
                                chunks_count += 1
                            elif args.dry_run:
                                chunks_count += 1
                    for transcript_type, transcript_path in transcript_paths(current_video_dir):
                        print(f"[transcript] {current_video_dir.name}: {transcript_type} - {transcript_path.name}")
                        if not args.dry_run and upsert_transcript(
                            cursor,
                            video_id,
                            transcript_type,
                            transcript_path,
                        ):
                            transcripts_count += 1
                        elif args.dry_run:
                            transcripts_count += 1

            if args.dry_run:
                connection.rollback()
            else:
                connection.commit()

    print(f"{videos_count} videos synchronisees, {stats_count} snapshots stats synchronises.")
    print(f"{assets_count} assets traites, {transcripts_count} transcripts synchronises, {chunks_count} chunks synchronises.")
    if skipped:
        print(f"{len(skipped)} fichiers ignores car video absente de la table videos.")


if __name__ == "__main__":
    main()
