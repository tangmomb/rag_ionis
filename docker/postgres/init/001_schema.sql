CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS videos (
    id BIGSERIAL PRIMARY KEY,
    youtube_video_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    raw_json JSONB,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS video_daily_stats (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    view_count BIGINT,
    like_count BIGINT,
    comment_count BIGINT,
    raw_json JSONB,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS video_transcripts (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    language_code TEXT NOT NULL,
    transcript TEXT,
    transcript_timecodes TEXT,
    transcript_timecodes_enrichi TEXT,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, language_code)
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
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS video_elements (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL,
    s3_bucket TEXT,
    s3_key TEXT,
    s3_uri TEXT,
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, s3_key)
);

CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at);
CREATE INDEX IF NOT EXISTS idx_video_daily_stats_snapshot_date ON video_daily_stats(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_video_transcripts_video_id ON video_transcripts(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent_comment_id ON comments(parent_comment_id);
CREATE INDEX IF NOT EXISTS idx_video_elements_video_id ON video_elements(video_id);
CREATE INDEX IF NOT EXISTS idx_video_elements_asset_type ON video_elements(asset_type);
CREATE INDEX IF NOT EXISTS idx_video_elements_s3_key ON video_elements(s3_key);
