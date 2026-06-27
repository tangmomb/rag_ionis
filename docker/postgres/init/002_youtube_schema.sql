CREATE TABLE IF NOT EXISTS videos (
    id BIGSERIAL PRIMARY KEY,
    youtube_video_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS video_transcripts (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    language_code TEXT NOT NULL,
    language_name TEXT,
    is_generated BOOLEAN,
    text TEXT NOT NULL,
    segments JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, language_code)
);

CREATE TABLE IF NOT EXISTS comments (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    parent_comment_id BIGINT REFERENCES comments(id) ON DELETE CASCADE,
    youtube_comment_id TEXT NOT NULL UNIQUE,
    author_name TEXT,
    author_channel_id TEXT,
    text TEXT NOT NULL,
    like_count BIGINT,
    published_at TIMESTAMPTZ,
    updated_at_youtube TIMESTAMPTZ,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at);
CREATE INDEX IF NOT EXISTS idx_video_daily_stats_snapshot_date ON video_daily_stats(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_video_transcripts_video_id ON video_transcripts(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent_comment_id ON comments(parent_comment_id);
