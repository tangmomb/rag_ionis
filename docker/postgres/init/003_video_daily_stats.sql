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

CREATE INDEX IF NOT EXISTS idx_video_daily_stats_snapshot_date ON video_daily_stats(snapshot_date);

ALTER TABLE videos
    DROP COLUMN IF EXISTS view_count,
    DROP COLUMN IF EXISTS like_count,
    DROP COLUMN IF EXISTS comment_count;
