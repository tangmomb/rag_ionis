import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
from pipeline.support.environment import load_project_env
from pipeline.support.analysis import load_routing_facts
from pipeline.support.paths import (
    CANONICAL_TRANSCRIPTS_DIR_NAME,
    consolidate_init_dir,
    existing_chunks_dir,
    existing_speakers_dir,
    existing_transcripts_dir,
    existing_youtube_api_infos_path,
    youtube_comments_path,
)
from pipeline.steps.transcripts.artifacts import (
    LEGACY_TRANSCRIPT_ENRICHED_NAMES,
    LEGACY_TRANSCRIPT_PLAIN_NAMES,
    TRANSCRIPT_3_ENRICHED_NAME,
    TRANSCRIPT_PLAIN_NAME,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DEFAULT_S3_ROOT_PREFIX = "youtube"
DEFAULT_S3_BUCKET_NAME = ""
DATA_SCHEMA = "data"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 2000
CHUNK_LEVELS = {"global", "section", "detail"}
DEFAULT_CHUNK_LEVEL = "detail"
SCHEMA_PATH = ROOT_DIR / "docker" / "postgres" / "init" / "001_schema.sql"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
PLAIN_TRANSCRIPT_NAME = TRANSCRIPT_PLAIN_NAME
WHISPER_TRANSCRIPT_ENRICHED_NAME = TRANSCRIPT_3_ENRICHED_NAME
LEGACY_ENRICHED_TRANSCRIPT_SUFFIX = "_transcript_timecodes_corrected_enrichi.txt"
LEGACY_PLAIN_TRANSCRIPT_SUFFIX = "_transcript.txt"
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
            is_long_video BOOLEAN GENERATED ALWAYS AS (
                COALESCE(duration_seconds > 600, FALSE)
            ) STORED,
            thumbnail_medium_url TEXT,
            has_subtitles BOOLEAN,
            video_type TEXT,
            s3_uri TEXT,
            published_at TIMESTAMPTZ,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    cursor.execute(
        """
        ALTER TABLE videos
        ADD COLUMN IF NOT EXISTS is_long_video BOOLEAN GENERATED ALWAYS AS (
            COALESCE(duration_seconds > 600, FALSE)
        ) STORED
        """
    )
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS thumbnail_medium_url TEXT")
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS has_subtitles BOOLEAN")
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS video_type TEXT")
    cursor.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS s3_uri TEXT")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS updated_at")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS s3_bucket")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS s3_prefix")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS video_summary")
    cursor.execute("ALTER TABLE videos DROP COLUMN IF EXISTS raw_json")
    ensure_speakers_schema(cursor)

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
    ensure_comments_schema(cursor)
    cursor.execute("DROP TABLE IF EXISTS video_elements CASCADE")
    cursor.execute(f"DROP TABLE IF EXISTS {legacy_video_elements_name()} CASCADE")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_snapshot_date ON stats(snapshot_date)")
    ensure_chunks_schema(cursor)
    ensure_transcripts_schema(cursor)


def ensure_comments_schema(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id BIGSERIAL PRIMARY KEY,
            video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            parent_comment_id BIGINT REFERENCES comments(id) ON DELETE CASCADE,
            youtube_comment_id TEXT NOT NULL UNIQUE,
            author_name TEXT,
            text TEXT NOT NULL,
            like_count BIGINT,
            published_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ,
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE
        )
        """
    )
    cursor.execute("ALTER TABLE comments ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ")
    cursor.execute(
        "ALTER TABLE comments ADD COLUMN IF NOT EXISTS "
        "first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()"
    )
    cursor.execute(
        "ALTER TABLE comments ADD COLUMN IF NOT EXISTS "
        "last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()"
    )
    cursor.execute(
        "ALTER TABLE comments ADD COLUMN IF NOT EXISTS "
        "is_deleted BOOLEAN NOT NULL DEFAULT FALSE"
    )
    cursor.execute("ALTER TABLE comments DROP COLUMN IF EXISTS raw_json")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_parent_comment_id "
        "ON comments(parent_comment_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_last_seen_at ON comments(last_seen_at)"
    )
    ensure_update_runs_schema(cursor)


def ensure_update_runs_schema(cursor):
    cursor.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('data.update_runs') IS NULL
               AND to_regclass('data.youtube_sync_runs') IS NOT NULL THEN
                ALTER TABLE data.youtube_sync_runs RENAME TO update_runs;
            END IF;
        END
        $$
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS update_runs (
            run_type TEXT NOT NULL CHECK (run_type IN ('stats', 'videos', 'all', 'legacy')),
            id BIGSERIAL PRIMARY KEY,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            archive_path TEXT,
            status TEXT NOT NULL,
            videos_discovered INTEGER NOT NULL DEFAULT 0,
            videos_updated INTEGER NOT NULL DEFAULT 0,
            videos_skipped INTEGER NOT NULL DEFAULT 0,
            stats_snapshots INTEGER NOT NULL DEFAULT 0,
            comments_seen INTEGER NOT NULL DEFAULT 0,
            comments_new INTEGER NOT NULL DEFAULT 0,
            comments_refreshed INTEGER NOT NULL DEFAULT 0,
            comments_deleted INTEGER NOT NULL DEFAULT 0,
            new_videos INTEGER NOT NULL DEFAULT 0,
            new_video_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            previous_archive_path TEXT,
            new_since_previous INTEGER NOT NULL DEFAULT 0,
            new_since_previous_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            pipeline_videos_started INTEGER NOT NULL DEFAULT 0,
            pipeline_videos_completed INTEGER NOT NULL DEFAULT 0,
            videos_with_new_comments INTEGER NOT NULL DEFAULT 0,
            new_comments_detected INTEGER NOT NULL DEFAULT 0,
            errors JSONB NOT NULL DEFAULT '[]'::jsonb
        )
        """
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "run_type TEXT NOT NULL DEFAULT 'legacy' "
        "CHECK (run_type IN ('stats', 'videos', 'all', 'legacy'))"
    )
    cursor.execute("ALTER TABLE update_runs ALTER COLUMN run_type DROP DEFAULT")
    cursor.execute(
        """
        UPDATE update_runs
        SET run_type = CASE
            WHEN archive_path LIKE '%update_stats%' THEN 'stats'
            WHEN archive_path LIKE '%update_videos%' THEN 'videos'
            ELSE 'all'
        END
        WHERE run_type = 'legacy'
        """
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS archive_path TEXT"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "new_videos INTEGER NOT NULL DEFAULT 0"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "new_video_ids JSONB NOT NULL DEFAULT '[]'::jsonb"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS previous_archive_path TEXT"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "new_since_previous INTEGER NOT NULL DEFAULT 0"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "new_since_previous_ids JSONB NOT NULL DEFAULT '[]'::jsonb"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "pipeline_videos_started INTEGER NOT NULL DEFAULT 0"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "pipeline_videos_completed INTEGER NOT NULL DEFAULT 0"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "videos_with_new_comments INTEGER NOT NULL DEFAULT 0"
    )
    cursor.execute(
        "ALTER TABLE update_runs ADD COLUMN IF NOT EXISTS "
        "new_comments_detected INTEGER NOT NULL DEFAULT 0"
    )
    cursor.execute(
        "DROP INDEX IF EXISTS data.idx_youtube_sync_runs_started_at"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_update_runs_started_at "
        "ON update_runs(started_at)"
    )


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


def ensure_speakers_schema(cursor):
    if table_exists(cursor, "speakers") and "video_id" in table_columns(cursor, "speakers"):
        cursor.execute(
            """
            CREATE TABLE speakers_global_migration (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                title TEXT,
                data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO speakers_global_migration (name, title, data_collected_date)
            SELECT DISTINCT ON (name)
                name,
                title,
                data_collected_date
            FROM speakers
            ORDER BY name, title IS NULL, id
            """
        )
        cursor.execute(
            """
            CREATE TABLE video_speakers (
                video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                speaker_id BIGINT NOT NULL
                    REFERENCES speakers_global_migration(id) ON DELETE CASCADE,
                data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (video_id, speaker_id)
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO video_speakers (video_id, speaker_id, data_collected_date)
            SELECT DISTINCT
                legacy.video_id,
                canonical.id,
                legacy.data_collected_date
            FROM speakers legacy
            JOIN speakers_global_migration canonical ON canonical.name = legacy.name
            ON CONFLICT (video_id, speaker_id) DO NOTHING
            """
        )
        cursor.execute("DROP TABLE speakers CASCADE")
        cursor.execute("ALTER TABLE speakers_global_migration RENAME TO speakers")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS speakers (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            title TEXT,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS video_speakers (
            video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            speaker_id BIGINT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (video_id, speaker_id)
        )
        """
    )
    video_columns = table_columns(cursor, "videos")
    if "speakers" in video_columns:
        cursor.execute(
            """
            INSERT INTO speakers (name, title)
            SELECT DISTINCT btrim(speaker_name), NULL
            FROM videos video
            CROSS JOIN LATERAL unnest(coalesce(video.speakers, ARRAY[]::text[]))
                AS speaker_name
            WHERE speaker_name IS NOT NULL
              AND btrim(speaker_name) <> ''
            ON CONFLICT (name) DO NOTHING
            """
        )
        cursor.execute(
            """
            INSERT INTO video_speakers (video_id, speaker_id)
            SELECT DISTINCT video.id, speaker.id
            FROM videos video
            CROSS JOIN LATERAL unnest(coalesce(video.speakers, ARRAY[]::text[]))
                AS speaker_name
            JOIN speakers speaker ON speaker.name = btrim(speaker_name)
            WHERE speaker_name IS NOT NULL
              AND btrim(speaker_name) <> ''
            ON CONFLICT (video_id, speaker_id) DO NOTHING
            """
        )
        cursor.execute("ALTER TABLE videos DROP COLUMN speakers")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_speakers_name ON speakers(name)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_video_speakers_speaker_id "
        "ON video_speakers(speaker_id)"
    )


def reset_data_schema(cursor):
    cursor.execute(
        """
        DROP TABLE IF EXISTS
            public.comments,
            public.chunks,
            public.transcripts,
            public.video_transcripts,
            public.video_speakers,
            public.speakers,
            public.update_runs,
            public.youtube_sync_runs,
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
            chunk_level TEXT NOT NULL DEFAULT 'detail'
                CONSTRAINT chunks_chunk_level_check
                CHECK (chunk_level IN ('global', 'section', 'detail')),
            chunk_parent_id BIGINT,
            content TEXT NOT NULL,
            embedding_model TEXT,
            embedding_dimensions INTEGER,
            embedding vector(2000),
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT chunks_video_id_id_key UNIQUE (video_id, id),
            CONSTRAINT chunks_video_id_chunk_level_chunk_index_key
                UNIQUE (video_id, chunk_level, chunk_index),
            CONSTRAINT chunks_video_id_chunk_parent_id_fkey
                FOREIGN KEY (video_id, chunk_parent_id)
                REFERENCES chunks(video_id, id) ON DELETE CASCADE,
            CONSTRAINT chunks_chunk_parent_not_self_check
                CHECK (chunk_parent_id IS NULL OR chunk_parent_id <> id)
        )
        """
    )
    ensure_data_collected_date_column(cursor, "chunks")
    cursor.execute(
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_level TEXT NOT NULL DEFAULT 'detail'"
    )
    cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_parent_id BIGINT")
    cursor.execute(
        """
        UPDATE chunks
        SET chunk_level = 'detail'
        WHERE chunk_level IS NULL
           OR chunk_level NOT IN ('global', 'section', 'detail')
        """
    )
    cursor.execute("ALTER TABLE chunks ALTER COLUMN chunk_level SET DEFAULT 'detail'")
    cursor.execute("ALTER TABLE chunks ALTER COLUMN chunk_level SET NOT NULL")
    cursor.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_video_id_chunk_index_key")
    if not constraint_exists(cursor, "chunks_chunk_level_check"):
        cursor.execute(
            """
            ALTER TABLE chunks
            ADD CONSTRAINT chunks_chunk_level_check
            CHECK (chunk_level IN ('global', 'section', 'detail'))
            """
        )
    if not constraint_exists(cursor, "chunks_video_id_id_key"):
        cursor.execute(
            """
            ALTER TABLE chunks
            ADD CONSTRAINT chunks_video_id_id_key
            UNIQUE (video_id, id)
            """
        )
    if constraint_exists(cursor, "chunks_chunk_parent_id_fkey"):
        cursor.execute("ALTER TABLE chunks DROP CONSTRAINT chunks_chunk_parent_id_fkey")
    if not constraint_exists(cursor, "chunks_video_id_chunk_parent_id_fkey"):
        cursor.execute(
            """
            ALTER TABLE chunks
            ADD CONSTRAINT chunks_video_id_chunk_parent_id_fkey
            FOREIGN KEY (video_id, chunk_parent_id)
            REFERENCES chunks(video_id, id) ON DELETE CASCADE
            """
        )
    if not constraint_exists(cursor, "chunks_chunk_parent_not_self_check"):
        cursor.execute(
            """
            ALTER TABLE chunks
            ADD CONSTRAINT chunks_chunk_parent_not_self_check
            CHECK (chunk_parent_id IS NULL OR chunk_parent_id <> id)
            """
        )
    if not constraint_exists(cursor, "chunks_video_id_chunk_level_chunk_index_key"):
        cursor.execute(
            """
            ALTER TABLE chunks
            ADD CONSTRAINT chunks_video_id_chunk_level_chunk_index_key
            UNIQUE (video_id, chunk_level, chunk_index)
            """
        )
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
    cursor.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS speakers")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks(video_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_video_level ON chunks(video_id, chunk_level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(chunk_parent_id)")
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
            transcript_enriched TEXT,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    ensure_data_collected_date_column(cursor, "transcripts")
    cursor.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS transcript TEXT")
    cursor.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS transcript_enriched TEXT")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS updated_at")

    cursor.execute("ALTER TABLE transcripts DROP CONSTRAINT IF EXISTS transcripts_video_id_language_code_key")
    cursor.execute("ALTER TABLE transcripts DROP CONSTRAINT IF EXISTS transcripts_video_id_language_code_transcript_type_key")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_type")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS text")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS segments")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_plain")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_timecoded")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_timecodes")
    cursor.execute("ALTER TABLE transcripts DROP COLUMN IF EXISTS transcript_timecodes_enrichi")
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


def load_chunk_embedding_payload(video_path, chunk_index, chunk_level=DEFAULT_CHUNK_LEVEL):
    if chunk_index is None:
        return None
    chunks_dir = existing_chunks_dir(video_path)
    normalized_level = str(chunk_level or DEFAULT_CHUNK_LEVEL).strip().lower()
    if normalized_level not in CHUNK_LEVELS:
        raise ValueError(f"Niveau de chunk invalide: {normalized_level!r}")
    if normalized_level == DEFAULT_CHUNK_LEVEL:
        candidates = [chunks_dir / f"chunk_{int(chunk_index):02d}_embedding.json"]
    else:
        candidates = [
            chunks_dir / f"chunk_{normalized_level}_{int(chunk_index):02d}_embedding.json"
        ]
    for candidate in candidates:
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
        candidates = [
            path
            for path in sorted(video_dir.iterdir())
            if path.is_dir()
            and path.name not in {"_00_info_videos", "_00_info_comments"}
        ]
    selected_ids = set(video_ids or [])
    return [path for path in candidates if not selected_ids or path.name in selected_ids]


def has_embeddings(video_dir):
    return any((Path(video_dir) / "outputs" / "chunks").glob("*_embedding.json"))


def ready_video_dirs(video_dirs):
    return [video_dir for video_dir in video_dirs if has_embeddings(video_dir)]


def local_video_identifier(video_path):
    path = Path(video_path)
    return path.name if path.is_dir() else path.stem


def load_video_metadata(video_path):
    target = existing_youtube_api_infos_path(video_path)
    if target.exists():
        return load_json(target)
    return None


def load_video_routing_facts(video_path):
    return load_routing_facts(video_path)


def load_video_speaker_details(video_path):
    candidate = existing_speakers_dir(video_path) / "speakers_validated.json"
    payload = load_json(candidate)
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
    for item in raw_details:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("speaker") or item.get("name") or "").split()).strip()
        title = " ".join(str(item.get("title") or "").split()).strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        details.append({"name": name, "title": title or None})
    return details


def load_video_speakers(video_path):
    return [detail["name"] for detail in load_video_speaker_details(video_path)] or None


def load_video_comments(video_path):
    payload = load_json(youtube_comments_path(video_path))
    if not isinstance(payload, dict) or not isinstance(payload.get("comments"), list):
        return None
    return payload


@dataclass(frozen=True)
class CommentSyncResult:
    seen: int
    created: int
    refreshed: int
    deleted: int


def normalized_cached_comments(payload):
    comments = []
    seen_ids = set()
    for item in payload.get("comments", []):
        if not isinstance(item, dict):
            continue
        youtube_comment_id = str(item.get("youtube_comment_id") or "").strip()
        if not youtube_comment_id or youtube_comment_id in seen_ids:
            continue
        seen_ids.add(youtube_comment_id)
        comments.append(
            {
                "youtube_comment_id": youtube_comment_id,
                "parent_youtube_comment_id": (
                    str(item.get("parent_youtube_comment_id") or "").strip()
                    or None
                ),
                "author_name": item.get("author_name"),
                "text": str(item.get("text") or ""),
                "like_count": parse_int(item.get("like_count")),
                "published_at": parse_datetime(item.get("published_at")),
                "updated_at": parse_datetime(item.get("updated_at")),
            }
        )
    return comments


def sync_video_comments_incremental(cursor, video_id, payload, *, collected_at=None):
    comments = normalized_cached_comments(payload)
    collected_at = (
        collected_at
        or parse_datetime(payload.get("fetched_at"))
        or datetime.now(timezone.utc)
    )

    cursor.execute(
        "SELECT youtube_comment_id FROM comments WHERE video_id = %s",
        (video_id,),
    )
    existing_ids = {str(row[0]) for row in cursor.fetchall()}

    database_ids = {}
    ordered = sorted(
        comments,
        key=lambda item: item["parent_youtube_comment_id"] is not None,
    )
    for item in ordered:
        parent_youtube_id = item["parent_youtube_comment_id"]
        parent_db_id = database_ids.get(parent_youtube_id)
        if parent_youtube_id and parent_db_id is None:
            raise ValueError(
                f"Commentaire parent {parent_youtube_id!r} absent du cache YouTube."
            )
        cursor.execute(
            """
            INSERT INTO comments (
                video_id,
                parent_comment_id,
                youtube_comment_id,
                author_name,
                text,
                like_count,
                published_at,
                updated_at,
                first_seen_at,
                last_seen_at,
                is_deleted
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            ON CONFLICT (youtube_comment_id) DO UPDATE SET
                video_id = EXCLUDED.video_id,
                parent_comment_id = EXCLUDED.parent_comment_id,
                author_name = EXCLUDED.author_name,
                text = EXCLUDED.text,
                like_count = EXCLUDED.like_count,
                published_at = EXCLUDED.published_at,
                updated_at = EXCLUDED.updated_at,
                last_seen_at = EXCLUDED.last_seen_at,
                is_deleted = FALSE
            RETURNING id
            """,
            (
                video_id,
                parent_db_id,
                item["youtube_comment_id"],
                item["author_name"],
                item["text"],
                item["like_count"],
                item["published_at"],
                item["updated_at"],
                collected_at,
                collected_at,
            ),
        )
        database_ids[item["youtube_comment_id"]] = cursor.fetchone()[0]

    current_ids = list(database_ids)
    cursor.execute(
        """
        UPDATE comments
        SET is_deleted = TRUE
        WHERE video_id = %s
          AND is_deleted = FALSE
          AND NOT (youtube_comment_id = ANY(%s::text[]))
        """,
        (video_id, current_ids),
    )
    deleted = max(0, int(cursor.rowcount or 0))
    current_id_set = set(current_ids)
    return CommentSyncResult(
        seen=len(current_ids),
        created=len(current_id_set - existing_ids),
        refreshed=len(current_id_set & existing_ids),
        deleted=deleted,
    )


def replace_video_comments(cursor, video_id, payload):
    return sync_video_comments_incremental(cursor, video_id, payload).seen


def replace_video_speakers(cursor, video_id, speaker_details):
    cursor.execute("DELETE FROM video_speakers WHERE video_id = %s", (video_id,))
    for detail in speaker_details:
        cursor.execute(
            """
            WITH canonical_speaker AS (
                INSERT INTO speakers (name, title, data_collected_date)
                VALUES (%s, %s, now())
                ON CONFLICT (name) DO UPDATE SET
                    title = EXCLUDED.title,
                    data_collected_date = now()
                RETURNING id
            )
            INSERT INTO video_speakers (video_id, speaker_id, data_collected_date)
            SELECT %s, id, now()
            FROM canonical_speaker
            ON CONFLICT (video_id, speaker_id) DO UPDATE SET
                data_collected_date = now()
            """,
            (detail["name"], detail.get("title"), video_id),
        )
    cursor.execute(
        """
        DELETE FROM speakers speaker
        WHERE NOT EXISTS (
            SELECT 1
            FROM video_speakers relation
            WHERE relation.speaker_id = speaker.id
        )
        """
    )
    return len(speaker_details)


def upsert_chunk(cursor, video_id, chunk_payload, embedding_payload=None):
    chunk_index = parse_int(chunk_payload.get("chunk_index"))
    chunk_level = str(chunk_payload.get("chunk_level") or DEFAULT_CHUNK_LEVEL).strip().lower()
    chunk_parent_id = parse_int(chunk_payload.get("chunk_parent_id"))
    content = str(chunk_payload.get("content") or "").strip()
    if chunk_index is None or not content:
        return False
    if chunk_level not in CHUNK_LEVELS:
        raise ValueError(
            f"Niveau de chunk {chunk_level!r} invalide ; valeurs attendues : "
            f"{', '.join(sorted(CHUNK_LEVELS))}."
        )

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
                chunk_level,
                chunk_parent_id,
                content,
                embedding_model,
                embedding_dimensions,
                embedding,
                data_collected_date
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, now())
            ON CONFLICT (video_id, chunk_level, chunk_index) DO UPDATE SET
                chunk_parent_id = EXCLUDED.chunk_parent_id,
                content = EXCLUDED.content,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dimensions = EXCLUDED.embedding_dimensions,
                embedding = EXCLUDED.embedding,
                data_collected_date = now()
            RETURNING id
            """,
            (
                video_id,
                chunk_index,
                chunk_level,
                chunk_parent_id,
                content,
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
                chunk_level,
                chunk_parent_id,
                content,
                embedding_model,
                data_collected_date
            )
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (video_id, chunk_level, chunk_index) DO UPDATE SET
                chunk_parent_id = EXCLUDED.chunk_parent_id,
                content = EXCLUDED.content,
                embedding_model = EXCLUDED.embedding_model,
                data_collected_date = now()
            RETURNING id
            """,
            (
                video_id,
                chunk_index,
                chunk_level,
                chunk_parent_id,
                content,
                embedding_model,
            ),
        )
    if hasattr(cursor, "fetchone"):
        row = cursor.fetchone()
        if row and row[0] is not None:
            return int(row[0])
    return True


def chunk_logical_key(chunk_payload):
    chunk_index = parse_int(chunk_payload.get("chunk_index"))
    chunk_level = str(chunk_payload.get("chunk_level") or DEFAULT_CHUNK_LEVEL).strip().lower()
    if chunk_index is None or chunk_level not in CHUNK_LEVELS:
        return None
    return chunk_level, chunk_index


def chunk_parent_key(chunk_payload):
    parent = chunk_payload.get("chunk_parent")
    if not isinstance(parent, dict):
        return None
    parent_index = parse_int(parent.get("chunk_index"))
    parent_level = str(parent.get("chunk_level") or "").strip().lower()
    if parent_index is None or parent_level not in CHUNK_LEVELS:
        raise ValueError(f"Reference parent de chunk invalide: {parent!r}")
    return parent_level, parent_index


def ordered_chunk_payloads(chunks):
    level_order = {"global": 0, "section": 1, "detail": 2}
    return sorted(
        (chunk for chunk in chunks if isinstance(chunk, dict)),
        key=lambda chunk: (
            level_order.get(
                str(chunk.get("chunk_level") or DEFAULT_CHUNK_LEVEL).strip().lower(),
                99,
            ),
            parse_int(chunk.get("chunk_index")) or 0,
        ),
    )


def upsert_video_chunks(cursor, video_id, video_path, chunks):
    database_ids = {}
    inserted = 0
    for original_payload in ordered_chunk_payloads(chunks):
        chunk_payload = dict(original_payload)
        parent_key = chunk_parent_key(chunk_payload)
        if parent_key is not None:
            parent_id = database_ids.get(parent_key)
            if parent_id is None:
                raise RuntimeError(
                    f"Parent logique introuvable pour le chunk {chunk_logical_key(chunk_payload)}: "
                    f"{parent_key}."
                )
            chunk_payload["chunk_parent_id"] = parent_id

        embedding_payload = load_chunk_embedding_payload(
            video_path,
            chunk_payload.get("chunk_index"),
            chunk_payload.get("chunk_level"),
        )
        result = upsert_chunk(
            cursor,
            video_id,
            chunk_payload,
            embedding_payload=embedding_payload,
        )
        if not result:
            continue
        inserted += 1
        logical_key = chunk_logical_key(chunk_payload)
        if logical_key is not None and isinstance(result, int) and not isinstance(result, bool):
            database_ids[logical_key] = result
        elif logical_key is not None and any(
            chunk_parent_key(candidate) == logical_key
            for candidate in chunks
            if isinstance(candidate, dict)
        ):
            raise RuntimeError(
                f"L'identifiant SQL du chunk parent {logical_key} n'a pas ete retourne."
            )
    return inserted


def video_storage_prefix(video_path, root_dir, prefix):
    relative_path = video_path.relative_to(root_dir).as_posix()
    if prefix:
        return f"{prefix}/{relative_path}"
    return relative_path


def upsert_video(cursor, video_path, root_dir, bucket, prefix):
    payload = load_video_metadata(video_path) or {}
    routing_facts = load_video_routing_facts(video_path)
    youtube_video_id = payload.get("youtube_video_id") or local_video_identifier(video_path)
    title = payload.get("title") or youtube_video_id
    description = payload.get("description")
    url = payload.get("url") or f"https://www.youtube.com/watch?v={youtube_video_id}"
    published_at = parse_datetime(payload.get("published_at"))
    duration_seconds = parse_int(payload.get("duration_seconds"))
    thumbnail_medium_url = extract_thumbnail_medium_url(payload)
    has_subtitles = (
        routing_facts.get("has_subtitles")
        if isinstance(routing_facts.get("has_subtitles"), bool)
        else None
    )
    video_type = (
        routing_facts.get("video_type")
        if isinstance(routing_facts.get("video_type"), str)
        else None
    )
    speaker_details = load_video_speaker_details(video_path)
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
            s3_uri
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (youtube_video_id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            url = EXCLUDED.url,
            published_at = EXCLUDED.published_at,
            duration_seconds = EXCLUDED.duration_seconds,
            thumbnail_medium_url = EXCLUDED.thumbnail_medium_url,
            has_subtitles = EXCLUDED.has_subtitles,
            video_type = EXCLUDED.video_type,
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
            s3_uri,
        ),
    )
    video_id = cursor.fetchone()[0]
    replace_video_speakers(cursor, video_id, speaker_details)
    return youtube_video_id, video_id, payload


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
            comment_count = EXCLUDED.comment_count,
            data_collected_date = now()
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
    # Seul WhisperX alimente le transcript publie. L'OCR reste un artefact
    # local de comparaison et ne doit jamais etre synchronise comme reference.
    transcript_dir = existing_transcripts_dir(
        video_dir,
        name=CANONICAL_TRANSCRIPTS_DIR_NAME,
    )
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
        (
            "plain",
            (PLAIN_TRANSCRIPT_NAME, *LEGACY_TRANSCRIPT_PLAIN_NAMES),
            (f"*{LEGACY_PLAIN_TRANSCRIPT_SUFFIX}",),
        ),
        (
            "enriched",
            (
                WHISPER_TRANSCRIPT_ENRICHED_NAME,
                *LEGACY_TRANSCRIPT_ENRICHED_NAMES,
            ),
            (f"*{LEGACY_ENRICHED_TRANSCRIPT_SUFFIX}",),
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
        "enriched": "transcript_enriched",
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
        "--skip-comments",
        action="store_true",
        help="N'actualise pas la table comments depuis le cache local.",
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
        "--snapshot-date",
        help="Date YYYY-MM-DD forcee pour le snapshot stats. Defaut: aujourd'hui.",
    )
    parser.add_argument(
        "--reset-database",
        action="store_true",
        help="Supprime et recree le schema data avant la mise a jour.",
    )
    return parser.parse_args()


def main():
    load_project_env(ROOT_DIR)
    args = parse_args()
    if not args.bucket:
        args.bucket = os.environ.get("S3_BUCKET_NAME", "")
    snapshot_date = date.fromisoformat(args.snapshot_date) if args.snapshot_date else None
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable: {video_dir}")

    if args.no_prefix:
        prefix = ""
    else:
        prefix = normalize_prefix(args.prefix) if args.prefix is not None else default_prefix(video_dir)

    candidate_dirs = candidate_video_dirs(video_dir, video_ids=args.video_id)
    video_dirs = ready_video_dirs(candidate_dirs)
    not_ready_dirs = [path for path in candidate_dirs if path not in video_dirs]
    files = sorted(
        path
        for current_video_dir in video_dirs
        for path in current_video_dir.rglob("*")
        if path.is_file()
    )

    print(f"Dossier source: {video_dir}")
    print(f"Bucket S3: {args.bucket or '(aucun)'}")
    print(f"Prefixe S3: {prefix or '(racine)'}")
    for skipped_video_dir in not_ready_dirs:
        print(f"[skip] {skipped_video_dir.name}: aucun embedding produit.")

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
            comments_count = 0
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
                if upsert_video_stats(
                    cursor,
                    video_id,
                    metadata_payload,
                    snapshot_date=snapshot_date,
                ):
                    stats_count += 1
                comments_payload = (
                    None
                    if args.skip_comments
                    else load_video_comments(current_video_dir)
                )
                if comments_payload is not None:
                    if args.dry_run:
                        comments_count += len(comments_payload.get("comments", []))
                    else:
                        comments_count += replace_video_comments(
                            cursor,
                            video_id,
                            comments_payload,
                        )

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
                        chunk_payloads = chunks_payload.get("chunks", [])
                        if not args.dry_run:
                            chunks_count += upsert_video_chunks(
                                cursor,
                                video_id,
                                current_video_dir,
                                chunk_payloads,
                            )
                        else:
                            chunks_count += len(
                                [chunk for chunk in chunk_payloads if isinstance(chunk, dict)]
                            )
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
    print(f"{comments_count} commentaires YouTube synchronises.")
    print(f"{assets_count} assets traites, {transcripts_count} transcripts synchronises, {chunks_count} chunks synchronises.")
    if not_ready_dirs:
        print(f"{len(not_ready_dirs)} videos ignorees sans embedding.")
    if skipped:
        print(f"{len(skipped)} fichiers ignores car video absente de la table videos.")


if __name__ == "__main__":
    main()
