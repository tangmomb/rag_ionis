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

CREATE INDEX IF NOT EXISTS idx_video_transcripts_video_id ON video_transcripts(video_id);
