DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_ionis_analytics') THEN
        CREATE ROLE rag_ionis_analytics
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            NOINHERIT;
    END IF;
END
$$;

ALTER ROLE rag_ionis_analytics SET default_transaction_read_only = on;
ALTER ROLE rag_ionis_analytics SET statement_timeout = '5s';
ALTER ROLE rag_ionis_analytics SET search_path = data, public;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA data FROM rag_ionis_analytics;
GRANT USAGE ON SCHEMA data TO rag_ionis_analytics;

REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA data FROM rag_ionis_analytics;

GRANT SELECT (
    id,
    youtube_video_id,
    title,
    description,
    url,
    duration_seconds,
    is_long_video,
    thumbnail_medium_url,
    has_subtitles,
    video_type,
    published_at,
    data_collected_date
) ON data.videos TO rag_ionis_analytics;

GRANT SELECT (
    id,
    video_id,
    view_count,
    like_count,
    comment_count,
    snapshot_date,
    data_collected_date
) ON data.stats TO rag_ionis_analytics;

GRANT SELECT (
    id,
    name,
    title,
    data_collected_date
) ON data.speakers TO rag_ionis_analytics;

GRANT SELECT (
    video_id,
    speaker_id,
    data_collected_date
) ON data.video_speakers TO rag_ionis_analytics;

GRANT SELECT (
    id,
    video_id,
    parent_comment_id,
    author_name,
    text,
    like_count,
    published_at,
    updated_at,
    is_deleted
) ON data.comments TO rag_ionis_analytics;
