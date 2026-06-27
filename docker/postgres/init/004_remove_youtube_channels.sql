DROP INDEX IF EXISTS idx_videos_channel_id;

ALTER TABLE videos
    DROP COLUMN IF EXISTS channel_id;

DROP TABLE IF EXISTS youtube_channels;
