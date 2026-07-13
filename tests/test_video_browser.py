from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.database_browser import app as browser


class VideoBrowserTests(unittest.TestCase):
    def test_video_outputs_are_indexed_for_graphical_browsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_dir = root / "20260713_1200_init" / "video123"
            (video_dir / "metadata").mkdir(parents=True)
            (video_dir / "outputs" / "images" / "graphic").mkdir(parents=True)
            (video_dir / "outputs" / "ocr").mkdir(parents=True)
            (video_dir / "outputs" / "transcripts_ocr").mkdir(parents=True)
            (video_dir / "outputs" / "chunks").mkdir(parents=True)
            (video_dir / "video123.mp4").touch()
            (video_dir / "outputs" / "images" / "graphic" / "00_01.jpg").touch()
            (video_dir / "metadata" / "youtube_video_metadata.json").write_text(
                json.dumps({"youtube_video_id": "video123", "title": "Vidéo test", "duration_seconds": 42}),
                encoding="utf-8",
            )
            (video_dir / "metadata" / "pipeline_analysis.json").write_text(
                json.dumps({"video_type": "interview", "has_subtitles": True}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "ocr" / "03_reviewed_ocr_overlays.json").write_text(
                json.dumps({"kinds": {"subtitle": {"00:01": "Bonjour"}}}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "transcripts_ocr" / "plain_transcript.txt").write_text(
                "Bonjour depuis le transcript.", encoding="utf-8"
            )
            (video_dir / "outputs" / "transcripts_ocr" / "video_summary.md").write_text(
                "# Résumé", encoding="utf-8"
            )
            (video_dir / "outputs" / "chunks" / "transcript_chunks.json").write_text(
                json.dumps({"chunks": [{"chunk_index": 1, "content": "Premier chunk"}]}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "chunks" / "chunk_01_embedding.json").write_text("{}", encoding="utf-8")

            with patch.object(browser, "DOWNLOAD_ROOT", root):
                indexed = browser.videos()
                detail = browser.video_detail("20260713_1200_init", "video123")

            self.assertEqual(len(indexed), 1)
            self.assertEqual(indexed[0]["stage"], "Prête")
            self.assertEqual(indexed[0]["image_count"], 1)
            self.assertEqual(indexed[0]["embedding_count"], 1)
            self.assertEqual(detail["ocr"]["subtitle"]["00:01"], "Bonjour")
            self.assertEqual(detail["chunks"][0]["content"], "Premier chunk")
            self.assertIn("transcript", detail["transcript"])


if __name__ == "__main__":
    unittest.main()
