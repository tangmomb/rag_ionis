from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.chunks import create_transcript_chunks as chunk_creation
from pipeline.steps.transcripts import correct_whisper_transcript as combined_correction
from pipeline.steps.speakers import correct_speaker_transcripts as speaker_correction


class CorrectSpeakerTranscriptsTests(unittest.TestCase):
    def test_combined_step_creates_ocr_corrected_transcript_with_gpt_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            video_dir.mkdir()
            video = video_dir / "video123.mp4"
            video.touch()
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            ocr_dir = video_dir / "outputs" / "ocr"
            speakers_dir = video_dir / "outputs" / "speakers"
            transcript_dir.mkdir(parents=True)
            ocr_dir.mkdir(parents=True)
            speakers_dir.mkdir(parents=True)
            (transcript_dir / "whisper_transcript_timecoded.txt").write_text(
                "[00:00-00:05] SPEAKER_00: Bonjour, je m'appelle Lucie Ouyaya.\n",
                encoding="utf-8",
            )
            (ocr_dir / "01_processed_ocr_items.json").write_text(
                '{"items":[]}',
                encoding="utf-8",
            )
            (speakers_dir / "speaker_candidates.json").write_text(
                json.dumps(
                    {
                        "source": "outputs/transcripts_whisper/whisper_transcript_timecoded.txt",
                        "speakers": ["Lucie Ouyaya"],
                        "candidates": [
                            {
                                "name": "Lucie Ouyaya",
                                "methods": ["transcript_je_m_appelle"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (speakers_dir / "speakers_validated.json").write_text(
                json.dumps(
                    {
                        "source": "outputs/speakers/speaker_candidates.json",
                        "speakers": ["Loucif Ouyahia"],
                    }
                ),
                encoding="utf-8",
            )

            result = combined_correction.correct_file(video, force=True, mode="balanced")

            self.assertEqual(
                result,
                transcript_dir / "whisper_transcript_timecoded_corrected.txt",
            )
            text = result.read_text(encoding="utf-8")
            self.assertIn("Loucif Ouyahia", text)
            self.assertNotIn("Lucie Ouyaya", text)
            correction_log = json.loads(
                (speakers_dir / "speaker_transcript_corrections.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                correction_log["files"][0]["path"],
                "outputs/transcripts_whisper/whisper_transcript_timecoded_corrected.txt",
            )

    def test_speaker_mapping_also_replaces_a_partially_ocr_corrected_name(self) -> None:
        mappings = [
            {
                "source_name": "Lucie Ouyaya",
                "validated_name": "Loucif Ouyahia",
                "changed": True,
            }
        ]

        corrected, counts = speaker_correction.apply_mappings(
            "Bonjour, je m'appelle Lucie Ouyahia.",
            mappings,
        )

        self.assertEqual(corrected, "Bonjour, je m'appelle Loucif Ouyahia.")
        self.assertEqual(counts, {"Lucie Ouyaya": 1})

    def test_source_only_does_not_modify_existing_enriched_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            video_dir.mkdir()
            video = video_dir / "video123.mp4"
            video.touch()
            transcript_dir = video_dir / "outputs" / "transcripts_ocr"
            speakers_dir = video_dir / "outputs" / "speakers"
            transcript_dir.mkdir(parents=True)
            speakers_dir.mkdir(parents=True)
            source = transcript_dir / "ocr_subtitles_timecoded_corrected.txt"
            source.write_text("[00:01] Lucie Ouyaya\n", encoding="utf-8")
            enriched = transcript_dir / "ocr_subtitles_timecoded_corrected_enriched.txt"
            enriched.write_text("[00:01] Lucie Ouyaya\n", encoding="utf-8")
            (speakers_dir / "speaker_candidates.json").write_text(
                json.dumps(
                    {
                        "source": "outputs/transcripts_ocr/ocr_subtitles_timecoded_corrected.txt",
                        "speakers": ["Lucie Ouyaya"],
                        "candidates": [
                            {
                                "name": "Lucie Ouyaya",
                                "methods": ["transcript_je_m_appelle"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (speakers_dir / "speakers_validated.json").write_text(
                json.dumps(
                    {
                        "source": "outputs/speakers/speaker_candidates.json",
                        "speakers": ["Loucif Ouyahia"],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(speaker_correction, "update_analysed_infos"):
                speaker_correction.correct_speaker_transcripts(
                    video,
                    force=True,
                    source_only=True,
                )

            self.assertIn("Loucif Ouyahia", source.read_text(encoding="utf-8"))
            self.assertIn("Lucie Ouyaya", enriched.read_text(encoding="utf-8"))

    def test_validated_speaker_replaces_transcript_candidate_in_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "4hYeJ7cq-tM"
            video_dir.mkdir()
            video = video_dir / "4hYeJ7cq-tM.mp4"
            video.touch()
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            speakers_dir = video_dir / "outputs" / "speakers"
            transcript_dir.mkdir(parents=True)
            speakers_dir.mkdir(parents=True)

            plain = transcript_dir / "plain_transcript.txt"
            timecoded = transcript_dir / "whisper_transcript_timecoded_corrected.txt"
            plain.write_text("Bonjour, je m'appelle Lucie Ouyaya.", encoding="utf-8")
            timecoded.write_text(
                "[00:00-00:10] SPEAKER_00: Bonjour, je m'appelle Lucie Ouyaya.\n",
                encoding="utf-8",
            )
            candidates = speakers_dir / "speaker_candidates.json"
            candidates.write_text(
                json.dumps(
                    {
                        "source": "outputs/transcripts_whisper/plain_transcript.txt",
                        "speakers": ["Lucie Ouyaya"],
                        "candidates": [
                            {
                                "name": "Lucie Ouyaya",
                                "methods": ["transcript_je_m_appelle"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            validated = speakers_dir / "speakers_validated.json"
            validated.write_text(
                json.dumps(
                    {
                        "source": "outputs/speakers/speaker_candidates.json",
                        "speakers": ["Loucif Ouyahia"],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(speaker_correction, "update_analysed_infos"):
                result = speaker_correction.correct_speaker_transcripts(video, force=True)

            self.assertEqual(result, speakers_dir / "speaker_transcript_corrections.json")
            self.assertIn("Loucif Ouyahia", plain.read_text(encoding="utf-8"))
            self.assertIn("Loucif Ouyahia", timecoded.read_text(encoding="utf-8"))
            self.assertIn("SPEAKER_00:", timecoded.read_text(encoding="utf-8"))
            self.assertEqual(list(transcript_dir.glob("*_speaker_corrected.txt")), [])

            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["replacement_count"], 2)
            self.assertEqual(
                payload["mappings"][0],
                {
                    "source_name": "Lucie Ouyaya",
                    "validated_name": "Loucif Ouyahia",
                    "methods": ["transcript_je_m_appelle"],
                    "changed": True,
                },
            )

            chunks = video_dir / "outputs" / "chunks" / "transcript_chunks.json"
            with (
                patch.object(chunk_creation, "validated_speakers_path", return_value=validated),
                patch.object(chunk_creation, "chunks_path", return_value=chunks),
                patch.object(chunk_creation, "update_analysed_infos"),
            ):
                chunks_path = chunk_creation.create_chunks(video, force=True)

            chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.assertEqual(
                chunks_payload["source"],
                "outputs/transcripts_whisper/plain_transcript.txt",
            )
            self.assertIn("Loucif Ouyahia", chunks_payload["chunks"][0]["content"])

    def test_chunks_are_obsolete_when_the_corrected_transcript_is_newer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transcript = root / "plain_transcript.txt"
            validated = root / "speakers_validated.json"
            chunks = root / "transcript_chunks.json"
            transcript.write_text("Nouveau transcript", encoding="utf-8")
            validated.write_text('{"speakers":["Speaker Test"]}', encoding="utf-8")
            chunks.write_text("{}", encoding="utf-8")
            os.utime(chunks, (100, 100))
            os.utime(validated, (200, 200))
            os.utime(transcript, (300, 300))

            self.assertFalse(
                chunk_creation.output_is_current(chunks, (transcript, validated))
            )
            os.utime(chunks, (400, 400))
            self.assertTrue(
                chunk_creation.output_is_current(chunks, (transcript, validated))
            )


if __name__ == "__main__":
    unittest.main()
