import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DEFAULT_S3_ROOT_PREFIX = "youtube"
DEFAULT_S3_BUCKET_NAME = ""
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
ENRICHED_TRANSCRIPT_SUFFIX = "_transcript_timecodes_corrected_enrichi.txt"
TIMECODED_TRANSCRIPT_SUFFIX = "_transcript_timecodes_corrected.txt"
LEGACY_TIMECODED_TRANSCRIPT_SUFFIX = "_transcript_timecodes.txt"
PLAIN_TRANSCRIPT_SUFFIX = "_transcript.txt"
OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
OCR_SUBTITLE_TIMECODED_SUFFIX = "_ocr_subtitle_timecodes.txt"
OCR_SUBTITLE_TIMECODED_CORRECTED_SUFFIX = "_ocr_subtitle_timecodes_corrected.txt"
OCR_SUBTITLE_ENRICHED_SUFFIX = "_ocr_subtitle_timecodes_corrected_enrichi.txt"

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
    candidates = sorted(
        path
        for path in parent_dir.iterdir()
        if path.is_dir() and any(video_files(path))
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


def normalize_prefix(prefix):
    return prefix.strip("/") if prefix else ""


def default_prefix(video_dir):
    return normalize_prefix(f"{DEFAULT_S3_ROOT_PREFIX}/{video_dir.name}")


def s3_key_for(path, root_dir, prefix):
    relative_path = path.relative_to(root_dir).as_posix()
    if prefix:
        return f"{prefix}/{relative_path}"
    return relative_path


def ensure_schema(cursor):
    ensure_video_elements_table_name(cursor)
    for table_name in ("videos", "video_daily_stats", "video_transcripts", "video_elements"):
        ensure_data_collected_date_column(cursor, table_name)

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS video_elements (
            id BIGSERIAL PRIMARY KEY,
            video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            asset_type TEXT NOT NULL,
            s3_bucket TEXT,
            s3_key TEXT,
            s3_uri TEXT,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (video_id, s3_key)
        )
        """
    )
    ensure_data_collected_date_column(cursor, "video_elements")
    cursor.execute("ALTER TABLE video_elements DROP CONSTRAINT IF EXISTS " + legacy_video_elements_name() + "_video_id_local_path_key")
    cursor.execute("ALTER TABLE video_elements DROP CONSTRAINT IF EXISTS video_elements_video_id_local_path_key")
    cursor.execute("ALTER TABLE video_elements DROP COLUMN IF EXISTS local_path")
    cursor.execute("ALTER TABLE video_elements DROP COLUMN IF EXISTS modified_at")
    cursor.execute("ALTER TABLE video_elements DROP COLUMN IF EXISTS updated_at")
    cursor.execute("ALTER TABLE video_elements DROP COLUMN IF EXISTS size_bytes")
    cursor.execute("ALTER TABLE video_elements DROP COLUMN IF EXISTS content_type")
    cursor.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'video_elements_video_id_s3_key_key'
            ) THEN
                ALTER TABLE video_elements
                    ADD CONSTRAINT video_elements_video_id_s3_key_key UNIQUE (video_id, s3_key);
            END IF;
        END $$;
        """
    )
    legacy_index_prefix = "idx_" + legacy_video_elements_name()
    cursor.execute(f"DROP INDEX IF EXISTS {legacy_index_prefix}_video_id")
    cursor.execute(f"DROP INDEX IF EXISTS {legacy_index_prefix}_asset_type")
    cursor.execute(f"DROP INDEX IF EXISTS {legacy_index_prefix}_s3_key")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_elements_video_id ON video_elements(video_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_elements_asset_type ON video_elements(asset_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_elements_s3_key ON video_elements(s3_key)")
    ensure_transcripts_schema(cursor)


def table_columns(cursor, table_name):
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        """,
        (table_name,),
    )
    return {row[0] for row in cursor.fetchall()}


def table_exists(cursor, table_name):
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
        )
        """,
        (table_name,),
    )
    return cursor.fetchone()[0]


def constraint_exists(cursor, constraint_name):
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = %s
        )
        """,
        (constraint_name,),
    )
    return cursor.fetchone()[0]


def legacy_video_elements_name():
    return "video_" + "assets"


def ensure_video_elements_table_name(cursor):
    legacy_table = legacy_video_elements_name()
    if table_exists(cursor, legacy_table) and not table_exists(cursor, "video_elements"):
        cursor.execute(f"ALTER TABLE {legacy_table} RENAME TO video_elements")
    legacy_constraint = legacy_table + "_video_id_s3_key_key"
    if constraint_exists(cursor, legacy_constraint):
        cursor.execute(f"ALTER TABLE video_elements RENAME CONSTRAINT {legacy_constraint} TO video_elements_video_id_s3_key_key")
    legacy_primary_key = legacy_table + "_pkey"
    if constraint_exists(cursor, legacy_primary_key):
        cursor.execute(f"ALTER TABLE video_elements RENAME CONSTRAINT {legacy_primary_key} TO video_elements_pkey")
    legacy_foreign_key = legacy_table + "_video_id_fkey"
    if constraint_exists(cursor, legacy_foreign_key):
        cursor.execute(f"ALTER TABLE video_elements RENAME CONSTRAINT {legacy_foreign_key} TO video_elements_video_id_fkey")


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
        CREATE TABLE IF NOT EXISTS video_transcripts (
            id BIGSERIAL PRIMARY KEY,
            video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            language_code TEXT NOT NULL,
            transcript TEXT,
            transcript_timecodes TEXT,
            transcript_timecodes_enrichi TEXT,
            data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    ensure_data_collected_date_column(cursor, "video_transcripts")
    cursor.execute("ALTER TABLE video_transcripts ADD COLUMN IF NOT EXISTS transcript TEXT")
    cursor.execute("ALTER TABLE video_transcripts ADD COLUMN IF NOT EXISTS transcript_timecodes TEXT")
    cursor.execute("ALTER TABLE video_transcripts ADD COLUMN IF NOT EXISTS transcript_timecodes_enrichi TEXT")
    cursor.execute("ALTER TABLE video_transcripts ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()")

    columns = table_columns(cursor, "video_transcripts")
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
            CREATE TEMP TABLE video_transcripts_merged ON COMMIT DROP AS
            SELECT
                MIN(id) AS id,
                video_id,
                language_code,
                {plain_expression} AS transcript,
                {timecodes_expression} AS transcript_timecodes,
                {enriched_expression} AS transcript_timecodes_enrichi,
                MIN(data_collected_date) AS data_collected_date,
                MAX(updated_at) AS updated_at
            FROM video_transcripts
            GROUP BY video_id, language_code
            """
        )
        cursor.execute("ALTER TABLE video_transcripts DROP CONSTRAINT IF EXISTS video_transcripts_video_id_language_code_transcript_type_key")
        cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS transcript_type")
        cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS text")
        cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS segments")
        cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS transcript_plain")
        cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS transcript_timecoded")
        cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS transcript_enriched")
        cursor.execute("TRUNCATE video_transcripts")
        cursor.execute(
            """
            INSERT INTO video_transcripts (
                id,
                video_id,
                language_code,
                transcript,
                transcript_timecodes,
                transcript_timecodes_enrichi,
                data_collected_date,
                updated_at
            )
            SELECT
                id,
                video_id,
                language_code,
                transcript,
                transcript_timecodes,
                transcript_timecodes_enrichi,
                data_collected_date,
                updated_at
            FROM video_transcripts_merged
            """
        )
        cursor.execute(
            """
            SELECT setval(
                pg_get_serial_sequence('video_transcripts', 'id'),
                COALESCE((SELECT MAX(id) FROM video_transcripts), 1),
                (SELECT COUNT(*) > 0 FROM video_transcripts)
            )
            """
        )

    cursor.execute("ALTER TABLE video_transcripts DROP CONSTRAINT IF EXISTS video_transcripts_video_id_language_code_key")
    cursor.execute("ALTER TABLE video_transcripts DROP CONSTRAINT IF EXISTS video_transcripts_video_id_language_code_transcript_type_key")
    cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS transcript_type")
    cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS text")
    cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS segments")
    cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS transcript_plain")
    cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS transcript_timecoded")
    cursor.execute("ALTER TABLE video_transcripts DROP COLUMN IF EXISTS transcript_enriched")
    cursor.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'video_transcripts_video_id_language_code_key'
            ) THEN
                ALTER TABLE video_transcripts
                    ADD CONSTRAINT video_transcripts_video_id_language_code_key
                    UNIQUE (video_id, language_code);
            END IF;
        END $$;
        """
    )


def video_db_ids(cursor):
    cursor.execute("SELECT youtube_video_id, id FROM videos")
    return dict(cursor.fetchall())


def clean_init_assets(cursor, bucket, root_prefix):
    root = normalize_prefix(root_prefix)
    pattern = f"{root}/%_init/%"
    if bucket:
        cursor.execute(
            """
            DELETE FROM video_elements
            WHERE s3_bucket = %s
              AND s3_key LIKE %s
            """,
            (bucket, pattern),
        )
    else:
        cursor.execute(
            """
            DELETE FROM video_elements
            WHERE s3_key LIKE %s
            """,
            (pattern,),
        )
    return cursor.rowcount


def asset_type(path):
    suffix = path.suffix.lower()
    name = path.name
    parts = set(path.parts)

    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if name.endswith(OCR_SUBTITLE_ENRICHED_SUFFIX):
        return "ocr_subtitle_enriched"
    if name.endswith(OCR_SUBTITLE_TIMECODED_CORRECTED_SUFFIX) or name.endswith(OCR_SUBTITLE_TIMECODED_SUFFIX):
        return "ocr_subtitle_timecoded"
    if name.endswith(OCR_SUBTITLE_SUFFIX):
        return "ocr_subtitle"
    if name.endswith(ENRICHED_TRANSCRIPT_SUFFIX):
        return "transcript_enriched"
    if name.endswith(TIMECODED_TRANSCRIPT_SUFFIX) or name.endswith(LEGACY_TIMECODED_TRANSCRIPT_SUFFIX):
        return "transcript_timecoded"
    if name.endswith(PLAIN_TRANSCRIPT_SUFFIX):
        return "transcript"
    if "analyse" in parts and suffix == ".json":
        return "image_text_analysis_json"
    if "analyse" in parts and suffix == ".txt":
        return "image_text_analysis"
    if "images" in parts and suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "image"
    return "other"


def video_folder_name(path, root_dir):
    relative_parts = path.relative_to(root_dir).parts
    if len(relative_parts) < 2:
        return path.stem
    return relative_parts[0]


def upsert_asset(cursor, video_id, path, root_dir, bucket, prefix):
    s3_key = s3_key_for(path, root_dir, prefix) if bucket is not None else None
    s3_uri = f"s3://{bucket}/{s3_key}" if bucket and s3_key else None

    cursor.execute(
        """
        INSERT INTO video_elements (
            video_id,
            asset_type,
            s3_bucket,
            s3_key,
            s3_uri
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id, s3_key) DO UPDATE SET
            asset_type = EXCLUDED.asset_type,
            s3_bucket = EXCLUDED.s3_bucket,
            s3_key = EXCLUDED.s3_key,
            s3_uri = EXCLUDED.s3_uri
        """,
        (
            video_id,
            asset_type(path),
            bucket,
            s3_key,
            s3_uri,
        ),
    )


def transcript_paths(video_dir):
    transcript_dir = video_dir / "transcript"
    if not transcript_dir.exists():
        return []

    candidates = (
        ("plain", (f"*{PLAIN_TRANSCRIPT_SUFFIX}",)),
        ("timecoded", (f"*{TIMECODED_TRANSCRIPT_SUFFIX}", f"*{LEGACY_TIMECODED_TRANSCRIPT_SUFFIX}")),
        ("enriched", (f"*{ENRICHED_TRANSCRIPT_SUFFIX}",)),
    )
    found = []
    for transcript_type, patterns in candidates:
        for pattern in patterns:
            matches = sorted(transcript_dir.glob(pattern))
            if matches:
                found.append((transcript_type, matches[0]))
                break
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
        INSERT INTO video_transcripts (
            video_id,
            language_code,
            {column},
            updated_at
        )
        VALUES (%s, 'fr', %s, now())
        ON CONFLICT (video_id, language_code) DO UPDATE SET
            {column} = EXCLUDED.{column},
            updated_at = now()
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
        help="Prefixe S3. Defaut: youtube/nom_du_dossier_traite, comme la Step 13",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Considere que les fichiers sont a la racine du bucket.",
    )
    parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="N'actualise pas la table video_transcripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les changements sans modifier la base.",
    )
    parser.add_argument(
        "--clean-init-assets",
        action="store_true",
        help="Supprime les anciens assets SQL lies aux prefixes youtube/*_init avant l'upsert.",
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

    files = sorted(path for path in video_dir.rglob("*") if path.is_file())
    video_dirs = sorted(path for path in video_dir.iterdir() if path.is_dir())

    print(f"Dossier source: {video_dir}")
    print(f"Bucket S3: {args.bucket or '(aucun)'}")
    print(f"Prefixe S3: {prefix or '(racine)'}")

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            ensure_schema(cursor)
            if args.clean_init_assets and not args.no_prefix:
                deleted = clean_init_assets(cursor, args.bucket, DEFAULT_S3_ROOT_PREFIX)
                print(f"[clean] {deleted} anciens assets init supprimes.")

            ids_by_youtube_id = video_db_ids(cursor)
            assets_count = 0
            transcripts_count = 0
            skipped = []

            for path in files:
                youtube_video_id = video_folder_name(path, video_dir)
                video_id = ids_by_youtube_id.get(youtube_video_id)
                if not video_id:
                    skipped.append(path)
                    continue
                print(f"[asset] {youtube_video_id}: {path.relative_to(video_dir).as_posix()}")
                if not args.dry_run:
                    upsert_asset(cursor, video_id, path, video_dir, args.bucket, prefix)
                assets_count += 1

            if not args.skip_transcripts:
                for current_video_dir in video_dirs:
                    video_id = ids_by_youtube_id.get(current_video_dir.name)
                    if not video_id:
                        continue
                    for transcript_type, transcript_path in transcript_paths(current_video_dir):
                        print(f"[transcript] {current_video_dir.name}: {transcript_type} - {transcript_path.name}")
                        if not args.dry_run and upsert_transcript(cursor, video_id, transcript_type, transcript_path):
                            transcripts_count += 1
                        elif args.dry_run:
                            transcripts_count += 1

            if args.dry_run:
                connection.rollback()
            else:
                connection.commit()

    print(f"{assets_count} assets traites, {transcripts_count} transcripts synchronises.")
    if skipped:
        print(f"{len(skipped)} fichiers ignores car video absente de la table videos.")


if __name__ == "__main__":
    main()
