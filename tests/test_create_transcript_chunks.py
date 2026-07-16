from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
INIT_DIR = ROOT_DIR / "scripts" / "init"
if str(INIT_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_DIR))

from common import used_by_pre21a_propose_speakers as speaker_proposal
from common import used_by_pre21b_validate_speakers as speaker_validation
from common import used_by_hs21_ns21_create_transcript_chunks as chunk_creation


class CreateTranscriptChunksTests(unittest.TestCase):
    def test_speaker_proposal_prefers_raw_whisper_before_plain_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            transcript_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            raw = transcript_dir / "whisper_transcript_timecoded.txt"
            raw.write_text(
                "[00:00-00:05] SPEAKER_00: Bonjour, je m'appelle Raw Speaker.",
                encoding="utf-8",
            )
            (transcript_dir / "plain_transcript.txt").write_text(
                "Bonjour, je m'appelle Old Speaker.",
                encoding="utf-8",
            )

            self.assertEqual(speaker_proposal.source_text_path(video), raw)

    def test_has_sub_speaker_proposal_uses_corrected_ocr_before_plain_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            transcript_dir = video_dir / "outputs" / "transcripts_ocr"
            transcript_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            corrected_ocr = transcript_dir / "ocr_subtitles_timecoded_corrected.txt"
            corrected_ocr.write_text(
                "[00:01] Bonjour, je m'appelle OCR Speaker.",
                encoding="utf-8",
            )
            (transcript_dir / "plain_transcript.txt").write_text(
                "Bonjour, je m'appelle Old Speaker.",
                encoding="utf-8",
            )
            self.assertEqual(speaker_proposal.source_text_path(video), corrected_ocr)

    def test_transcript_speakers_do_not_require_an_ocr_match(self) -> None:
        transcript = (
            "Je m'appelle Hugo Géradain. "
            "Je m'appelle Clara De Almeida."
        )
        speakers = speaker_proposal.propose_speakers(transcript, [])["speakers"]

        self.assertEqual(speakers, ["Hugo Géradain", "Clara De Almeida"])

    def test_je_suis_detects_a_speaker(self) -> None:
        transcript = (
            "Je suis Sophie Vanderpol Je suis la Fondatrice d'Olidi. "
            "Je suis Responsable d'Affaires."
        )

        speakers = speaker_proposal.propose_speakers(transcript, [])["speakers"]

        self.assertIn("Sophie Vanderpol", speakers)
        self.assertNotIn("la Fondatrice d'Olidi", speakers)

    def test_each_speaker_stage_writes_its_own_json_before_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            video_dir.mkdir()
            video = video_dir / "video123.mp4"
            video.touch()
            transcript = video_dir / "plain_transcript.txt"
            transcript.write_text("Je suis Sophie Vanderpol.", encoding="utf-8")
            candidates = video_dir / "outputs" / "speakers" / "speaker_candidates.json"
            validated = video_dir / "outputs" / "speakers" / "speakers_validated.json"
            chunks = video_dir / "outputs" / "chunks" / "transcript_chunks.json"

            with (
                patch.object(speaker_proposal, "source_text_path", return_value=transcript),
                patch.object(speaker_proposal, "load_ocr_speaker_candidates", return_value=([], None)),
                patch.object(speaker_proposal, "candidates_path", return_value=candidates),
                patch.object(speaker_proposal, "video_title", return_value="Sophie Vanderpol témoigne"),
            ):
                speaker_proposal.propose_for_video(video, force=True)

            candidate_payload = json.loads(candidates.read_text(encoding="utf-8"))
            self.assertEqual(candidate_payload["speakers"], ["Sophie Vanderpol"])
            self.assertEqual(candidate_payload["video_title"], "Sophie Vanderpol témoigne")

            speaker_validation.write_validation_output(
                "test-model",
                video,
                candidates,
                validated,
                ["Sophie Vanderpol"],
                ["Sophie Vanderpol"],
                '{"valid_speakers":["Sophie Vanderpol"]}',
                {"api": "test"},
            )

            with (
                patch.object(chunk_creation, "source_text_path", return_value=transcript),
                patch.object(chunk_creation, "validated_speakers_path", return_value=validated),
                patch.object(chunk_creation, "chunks_path", return_value=chunks),
            ):
                chunk_creation.create_chunks(video, force=True)

            validated_payload = json.loads(validated.read_text(encoding="utf-8"))
            chunks_payload = json.loads(chunks.read_text(encoding="utf-8"))
            self.assertEqual(validated_payload["speakers"], ["Sophie Vanderpol"])
            self.assertEqual(
                chunks_payload["chunks"][0]["meta_data"]["speakers"],
                ["Sophie Vanderpol"],
            )


if __name__ == "__main__":
    unittest.main()
