CREATE EXTENSION IF NOT EXISTS vector;

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
    data_collected_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS video_stats (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    view_count BIGINT,
    like_count BIGINT,
    comment_count BIGINT,
    snapshot_date DATE NOT NULL,
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
    video_summary TEXT,
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
    published_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at);
CREATE INDEX IF NOT EXISTS idx_video_stats_snapshot_date ON video_stats(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_video_transcripts_video_id ON video_transcripts(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent_comment_id ON comments(parent_comment_id);
