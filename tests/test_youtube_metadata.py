from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.support.youtube_metadata import (
    YoutubeMetadataError,
    load_youtube_metadata,
    youtube_duration_seconds,
)


class YoutubeMetadataTests(unittest.TestCase):
    def make_video(self, root: Path) -> Path:
        video_dir = root / "abcdefghijk"
        video_dir.mkdir()
        video = video_dir / "abcdefghijk.mp4"
        video.touch()
        return video

    def test_duration_comes_from_the_youtube_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(Path(temporary_directory))
            metadata_dir = video.parent / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "youtube_video_metadata.json").write_text(
                json.dumps({"duration_seconds": 742}),
                encoding="utf-8",
            )

            metadata = load_youtube_metadata(video)

        self.assertEqual(youtube_duration_seconds(metadata), 742.0)

    def test_missing_youtube_json_is_an_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(Path(temporary_directory))

            with self.assertRaisesRegex(
                YoutubeMetadataError,
                "Metadonnees YouTube introuvables",
            ):
                load_youtube_metadata(video)

    def test_invalid_duration_is_rejected(self) -> None:
        for value in (None, True, 0, -1, "unknown"):
            with self.subTest(value=value):
                with self.assertRaises(YoutubeMetadataError):
                    youtube_duration_seconds({"duration_seconds": value})


if __name__ == "__main__":
    unittest.main()
