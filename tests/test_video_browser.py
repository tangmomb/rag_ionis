from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.database_browser import app as browser


class VideoBrowserTests(unittest.TestCase):
    def test_current_init_video_outputs_are_indexed_for_graphical_browsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_dir = root / "init" / "video123"
            (video_dir / "metadata").mkdir(parents=True)
            (video_dir / "outputs" / "images" / "graphic").mkdir(parents=True)
            (video_dir / "outputs" / "ocr").mkdir(parents=True)
            (video_dir / "outputs" / "transcripts_ocr").mkdir(parents=True)
            (video_dir / "outputs" / "speakers").mkdir(parents=True)
            (video_dir / "outputs" / "chunks").mkdir(parents=True)
            (video_dir / "video123.mp4").touch()
            (video_dir / "outputs" / "images" / "graphic" / "00_01.jpg").touch()
            (video_dir / "metadata" / "youtube_video_metadata.json").write_text(
                json.dumps({"youtube_video_id": "video123", "title": "Vidéo test", "duration_seconds": 42}),
                encoding="utf-8",
            )
            (video_dir / "metadata" / "video_manifest.json").write_text(
                json.dumps(
                    {
                        "routing_facts": {
                            "video_type": "interview",
                            "has_subtitles": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "ocr" / "03_reviewed_ocr_overlays.json").write_text(
                json.dumps({"kinds": {"subtitle": {"00:01": "Bonjour"}}}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "transcripts_ocr" / "plain_transcript.txt").write_text(
                "Bonjour depuis le transcript.", encoding="utf-8"
            )
            (video_dir / "outputs" / "transcripts_ocr" / "ocr_subtitles_timecoded.txt").write_text(
                "[00:01] Bonjour brut.", encoding="utf-8"
            )
            (
                video_dir
                / "outputs"
                / "transcripts_ocr"
                / "ocr_subtitles_timecoded_corrected.txt"
            ).write_text("[00:01] Bonjour corrigé.", encoding="utf-8")
            (
                video_dir
                / "outputs"
                / "transcripts_ocr"
                / "ocr_subtitles_timecoded_corrected_enriched.txt"
            ).write_text("[00:01] ANIMATIONS: Bonjour enrichi.", encoding="utf-8")
            (video_dir / "outputs" / "speakers" / "speakers_validated.json").write_text(
                json.dumps({"speakers": ["Alice Martin", "Bob Durand"]}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "chunks" / "transcript_chunks.json").write_text(
                json.dumps({"chunks": [{"chunk_index": 1, "content": "Premier chunk"}]}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "chunks" / "chunk_01_embedding.json").write_text("{}", encoding="utf-8")

            with patch.object(browser, "DOWNLOAD_ROOT", root):
                indexed = browser.videos()
                detail = browser.video_detail("init", "video123")

            self.assertEqual(len(indexed), 1)
            self.assertEqual(indexed[0]["stage"], "Prête")
            self.assertEqual(indexed[0]["image_count"], 1)
            self.assertEqual(indexed[0]["embedding_count"], 1)
            self.assertEqual(indexed[0]["speakers"], ["Alice Martin", "Bob Durand"])
            self.assertNotIn("has_summary", indexed[0])
            self.assertNotIn("summary", detail)
            self.assertEqual(detail["ocr"]["subtitle"]["00:01"], "Bonjour")
            self.assertEqual(detail["chunks"][0]["content"], "Premier chunk")
            self.assertIn("transcript", detail["transcript"])
            self.assertEqual(
                [transcript["key"] for transcript in detail["transcripts"]],
                ["plain", "raw_timecoded", "corrected_timecoded", "enriched"],
            )
            self.assertEqual(
                detail["transcripts"][2]["content"],
                "[00:01] Bonjour corrigé.",
            )
            self.assertEqual(detail["transcripts"][3]["label"], "Enrichi")
            self.assertEqual(detail["speakers"], ["Alice Martin", "Bob Durand"])

    def test_legacy_dated_init_directories_remain_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_dir = root / "20260713_1200_init" / "video123"
            video_dir.mkdir(parents=True)
            (video_dir / "video123.mp4").touch()

            with patch.object(browser, "DOWNLOAD_ROOT", root):
                indexed = browser.videos()

            self.assertEqual(len(indexed), 1)
            self.assertEqual(indexed[0]["run"], "20260713_1200_init")


if __name__ == "__main__":
    unittest.main()
