CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;
CREATE SCHEMA IF NOT EXISTS data;
CREATE SCHEMA IF NOT EXISTS chat;
SET search_path TO data, public;

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
);

CREATE TABLE IF NOT EXISTS speakers (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    title TEXT,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS video_speakers (
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    speaker_id BIGINT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (video_id, speaker_id)
);

CREATE TABLE IF NOT EXISTS stats (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    view_count BIGINT,
    like_count BIGINT,
    comment_count BIGINT,
    snapshot_date DATE NOT NULL,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS transcripts (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    language_code TEXT NOT NULL,
    transcript TEXT,
    transcript_enriched TEXT,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, language_code)
);

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
);

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
);

CREATE TABLE IF NOT EXISTS update_runs (
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
);

CREATE TABLE IF NOT EXISTS chat.conversations (
    id BIGSERIAL PRIMARY KEY,
    date TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat.messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    answer_message TEXT,
    trace_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at);
CREATE INDEX IF NOT EXISTS idx_speakers_name ON speakers(name);
CREATE INDEX IF NOT EXISTS idx_video_speakers_speaker_id ON video_speakers(speaker_id);
CREATE INDEX IF NOT EXISTS idx_stats_snapshot_date ON stats(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_video_level ON chunks(video_id, chunk_level);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(chunk_parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent_comment_id ON comments(parent_comment_id);
CREATE INDEX IF NOT EXISTS idx_comments_last_seen_at ON comments(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_update_runs_started_at ON update_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_date ON chat.conversations(date);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat.messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_trace_id ON chat.messages(trace_id);
