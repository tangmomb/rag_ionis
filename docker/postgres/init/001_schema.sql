CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE SCHEMA IF NOT EXISTS chat;

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
    transcript_timecodes TEXT,
    transcript_timecodes_enrichi TEXT,
    video_summary TEXT,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, language_code)
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    speakers TEXT[],
    embedding_model TEXT,
    embedding vector(3072),
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS comments (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    parent_comment_id BIGINT REFERENCES comments(id) ON DELETE CASCADE,
    youtube_comment_id TEXT NOT NULL UNIQUE,
    author_name TEXT,
    text TEXT NOT NULL,
    like_count BIGINT,
    published_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS chat.conversations (
    id BIGSERIAL PRIMARY KEY,
    date TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat.messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    planner_prompt TEXT,
    planner_response_raw TEXT,
    intent_source TEXT,
    pydantic_verification BOOLEAN NOT NULL DEFAULT FALSE,
    execution_plan_json JSONB,
    sql_query TEXT,
    prefilter_trace JSONB,
    bm25_trace JSONB,
    vector_trace JSONB,
    rrf_trace JSONB,
    rerank_trace JSONB,
    retrieved_chunks JSONB,
    answer_message TEXT,
    date TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at);
CREATE INDEX IF NOT EXISTS idx_stats_snapshot_date ON stats(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index);
CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent_comment_id ON comments(parent_comment_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_date ON chat.conversations(date);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat.messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_date ON chat.messages(date);
