from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from utils.app_database_browser import app as browser


class VideoBrowserTests(unittest.TestCase):
    def test_current_init_video_outputs_are_indexed_for_graphical_browsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_dir = root / "init" / "video123"
            (video_dir / "metadata").mkdir(parents=True)
            (video_dir / "outputs" / "images" / "graphic").mkdir(parents=True)
            (video_dir / "outputs" / "ocr").mkdir(parents=True)
            (video_dir / "outputs" / "transcripts_ocr").mkdir(parents=True)
            (video_dir / "outputs" / "transcripts_whisper").mkdir(parents=True)
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
            (video_dir / "outputs" / "ocr" / "02_filtered_ocr_overlays.json").write_text(
                json.dumps({"kinds": {"subtitle": {"00:01": "Bonjour"}}}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "transcripts_ocr" / "plain_transcript.txt").write_text(
                "Comparaison OCR.", encoding="utf-8"
            )
            (
                video_dir
                / "outputs"
                / "transcripts_whisper"
                / "plain_transcript.txt"
            ).write_text("Bonjour depuis WhisperX.", encoding="utf-8")
            (
                video_dir
                / "outputs"
                / "transcripts_whisper"
                / "whisper_transcript_timecoded.txt"
            ).write_text("[00:01] Bonjour WhisperX brut.", encoding="utf-8")
            (
                video_dir
                / "outputs"
                / "transcripts_whisper"
                / "whisper_transcript_timecoded_corrected.txt"
            ).write_text("[00:01] Bonjour WhisperX corrigé.", encoding="utf-8")
            (
                video_dir
                / "outputs"
                / "transcripts_whisper"
                / "whisper_transcript_timecoded_corrected_enriched.txt"
            ).write_text("[00:01] Bonjour WhisperX enrichi.", encoding="utf-8")
            (video_dir / "outputs" / "speakers" / "speakers_validated.json").write_text(
                json.dumps(
                    {
                        "speakers": ["Alice Martin", "Bob Durand"],
                        "speaker_details": [
                            {
                                "speaker": "Alice Martin",
                                "title": "Directrice générale",
                            },
                            {
                                "speaker": "Bob Durand",
                                "title": "CTO",
                            },
                        ],
                    }
                ),
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
            self.assertEqual(
                indexed[0]["speaker_details"],
                [
                    {"name": "Alice Martin", "title": "Directrice générale"},
                    {"name": "Bob Durand", "title": "CTO"},
                ],
            )
            self.assertNotIn("has_summary", indexed[0])
            self.assertNotIn("summary", detail)
            self.assertEqual(detail["ocr"]["subtitle"]["00:01"], "Bonjour")
            self.assertEqual(detail["chunks"][0]["content"], "Premier chunk")
            self.assertEqual(detail["transcript"], "Bonjour depuis WhisperX.")
            self.assertEqual(
                [transcript["key"] for transcript in detail["transcripts"]],
                [
                    "plain",
                    "raw_timecoded",
                    "corrected_timecoded",
                    "enriched",
                    "ocr_plain",
                ],
            )
            self.assertEqual(
                detail["transcripts"][2]["content"],
                "[00:01] Bonjour WhisperX corrigé.",
            )
            self.assertEqual(
                detail["transcripts"][3]["label"],
                "3 — Transcript enrichi avec speakers",
            )
            self.assertTrue(detail["transcripts"][3]["editable"])
            self.assertFalse(detail["transcripts"][2]["editable"])
            self.assertEqual(detail["transcripts"][4]["content"], "Comparaison OCR.")
            self.assertEqual(
                detail["transcripts"][4]["label"],
                "OCR plain utilisé pour les corrections",
            )
            self.assertEqual(detail["speakers"], ["Alice Martin", "Bob Durand"])
            self.assertEqual(
                detail["speaker_details"],
                [
                    {"name": "Alice Martin", "title": "Directrice générale"},
                    {"name": "Bob Durand", "title": "CTO"},
                ],
            )

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

    def test_speaker_edits_update_only_the_validated_speakers_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_dir = root / "init" / "video123"
            speakers_dir = video_dir / "outputs" / "speakers"
            speakers_dir.mkdir(parents=True)
            target = speakers_dir / "speakers_validated.json"
            target.write_text(
                json.dumps(
                    {
                        "source": "speaker_candidates.json",
                        "speakers": ["Ancien nom"],
                        "speaker_details": [
                            {"speaker": "Ancien nom", "title": "Ancienne fonction"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(browser, "DOWNLOAD_ROOT", root):
                response = TestClient(browser.app).put(
                    "/api/videos/init/video123/speakers",
                    json={
                        "speakers": [
                            {
                                "name": "  Alice   Martin ",
                                "title": " Directrice générale ",
                            },
                            {"name": "Bob Durand", "title": ""},
                        ]
                    },
                )
            self.assertEqual(response.status_code, 200)
            result = response.json()

            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "speaker_candidates.json")
            self.assertEqual(payload["speakers"], ["Alice Martin", "Bob Durand"])
            self.assertEqual(
                payload["speaker_details"],
                [
                    {"speaker": "Alice Martin", "title": "Directrice générale"},
                    {"speaker": "Bob Durand", "title": ""},
                ],
            )
            self.assertEqual(
                result["path"],
                "outputs/speakers/speakers_validated.json",
            )

    def test_enriched_transcript_edit_updates_its_existing_file_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_dir = root / "init" / "video123"
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            transcript_dir.mkdir(parents=True)
            enriched = (
                transcript_dir
                / "whisper_transcript_timecoded_corrected_enriched.txt"
            )
            corrected = (
                transcript_dir
                / "whisper_transcript_timecoded_corrected.txt"
            )
            chunks_target = (
                video_dir / "outputs" / "chunks" / "transcript_chunks.json"
            )
            chunks_target.parent.mkdir(parents=True)
            chunks_target.write_text(
                json.dumps(
                    {
                        "chunking": {"profile": "short"},
                        "chunks": [
                            {"chunk_index": 1, "content": "Ancien chunk"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            enriched.write_text("[00:01] Ancien enrichi.", encoding="utf-8")
            corrected.write_text("[00:01] Corrigé inchangé.", encoding="utf-8")

            with patch.object(browser, "DOWNLOAD_ROOT", root):
                response = TestClient(browser.app).put(
                    "/api/videos/init/video123/transcripts/enriched",
                    json={"content": "[00:01] Nouveau transcript enrichi.\n"},
                )
            self.assertEqual(response.status_code, 200)
            result = response.json()

            self.assertEqual(
                enriched.read_text(encoding="utf-8"),
                "[00:01] Nouveau transcript enrichi.\n",
            )
            self.assertEqual(
                corrected.read_text(encoding="utf-8"),
                "[00:01] Corrigé inchangé.",
            )
            plain = transcript_dir / "transcript_plain.txt"
            self.assertEqual(
                plain.read_text(encoding="utf-8"),
                "Nouveau transcript enrichi.\n",
            )
            self.assertEqual(
                result["path"],
                (
                    "outputs/transcripts_whisper/"
                    "whisper_transcript_timecoded_corrected_enriched.txt"
                ),
            )
            self.assertEqual(
                result["plain"],
                {
                    "path": "outputs/transcripts_whisper/transcript_plain.txt",
                    "content": "Nouveau transcript enrichi.\n",
                },
            )
            chunks_payload = json.loads(
                chunks_target.read_text(encoding="utf-8")
            )
            self.assertEqual(
                chunks_payload["chunks"][0]["content"],
                "Nouveau transcript enrichi.",
            )
            self.assertEqual(result["chunks"]["profile"], "short")
            self.assertEqual(
                result["chunks"]["path"],
                "outputs/chunks/transcript_chunks.json",
            )
            self.assertEqual(
                result["chunks"]["items"][0]["content"],
                "Nouveau transcript enrichi.",
            )


if __name__ == "__main__":
    unittest.main()
