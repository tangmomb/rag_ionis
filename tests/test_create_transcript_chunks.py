from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.speakers import propose_speakers as speaker_proposal
from pipeline.steps.speakers import validate_speakers as speaker_validation
from pipeline.steps.chunks import create_transcript_chunks as chunk_creation


class CreateTranscriptChunksTests(unittest.TestCase):
    def test_speaker_proposal_uses_corrected_whisper_instead_of_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            transcript_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            raw = transcript_dir / "transcript_1_brut.txt"
            raw.write_text(
                "[00:00-00:05] SPEAKER_00: Bonjour, je m'appelle Raw Speaker.",
                encoding="utf-8",
            )
            corrected = transcript_dir / "transcript_2_corrected.txt"
            corrected.write_text(
                "[00:00-00:05] SPEAKER_00: Bonjour, je m'appelle Corrected Speaker.",
                encoding="utf-8",
            )
            (transcript_dir / "plain_transcript.txt").write_text(
                "Bonjour, je m'appelle Old Speaker.",
                encoding="utf-8",
            )

            self.assertEqual(speaker_proposal.source_text_path(video), corrected)

    def test_speaker_proposal_does_not_fall_back_to_raw_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            transcript_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            (transcript_dir / "transcript_1_brut.txt").write_text(
                "[00:01] Bonjour, je m'appelle Raw Speaker.",
                encoding="utf-8",
            )
            (transcript_dir / "plain_transcript.txt").write_text(
                "Bonjour, je m'appelle Old Speaker.",
                encoding="utf-8",
            )
            self.assertIsNone(speaker_proposal.source_text_path(video))

    def test_long_video_speaker_context_uses_first_1000_plain_characters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            transcript_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            transcript_text = "A" * 1000 + "B" * 500
            plain = transcript_dir / "transcript_plain.txt"
            plain.write_text(transcript_text, encoding="utf-8")

            target = speaker_proposal.propose_for_video(
                video,
                force=True,
                prefer_plain=True,
                transcript_excerpt_chars=1000,
            )
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertTrue(payload["source"].endswith("transcript_plain.txt"))
        self.assertEqual(payload["transcript_excerpt"], "A" * 1000)
        self.assertEqual(payload["transcript_excerpt_chars"], 1000)

    def test_transcript_introductions_do_not_create_speaker_candidates(self) -> None:
        transcript = (
            "Je suis Sophie Vanderpol, Fondatrice. "
            "Je m'appelle Hugo Jarguin, Responsable. "
            "Moi c'est Clara Dalmada."
        )

        speakers = speaker_proposal.propose_speakers(transcript, [])["speakers"]

        self.assertEqual(speakers, [])

    def test_transcript_speaker_count_uses_distinct_diarization_labels(self) -> None:
        transcript = (
            "[00:00-00:05] SPEAKER_00: Bonjour.\n"
            "[00:05-00:10] SPEAKER_01: Bienvenue.\n"
            "[00:10-00:15] SPEAKER_00: Suite.\n"
        )

        self.assertEqual(speaker_proposal.transcript_speaker_count(transcript), 2)

    def test_visual_lower_third_name_is_detected_from_current_ocr_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            ocr_dir = video_dir / "outputs" / "ocr"
            ocr_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            (ocr_dir / "01_processed_ocr_items.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "image": "footage/00_19.jpg",
                                "text": "Hugo Géradin",
                                "kind": "others",
                                "second": 19,
                                "box": [
                                    [30, 455],
                                    [248, 455],
                                    [248, 489],
                                    [30, 489],
                                ],
                            },
                            {
                                "image": "footage/00_19.jpg",
                                "text": "Responsable d'Affaires",
                                "kind": "others",
                                "second": 19,
                                "box": [
                                    [29, 499],
                                    [238, 499],
                                    [238, 521],
                                    [29, 521],
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            names, source = speaker_proposal.load_ocr_speaker_candidates(video)

            self.assertEqual(names, ["Hugo Géradin"])
            self.assertEqual(source, ocr_dir / "01_processed_ocr_items.json")

    def test_filtered_ocr_texts_include_every_kind_except_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            ocr_dir = video_dir / "outputs" / "ocr"
            ocr_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            filtered = ocr_dir / "02_filtered_ocr_overlays.json"
            filtered.write_text(
                json.dumps(
                    {
                        "kinds": {
                            "graphic": {
                                "00:01": "IONIS",
                            },
                            "others": {
                                "00:02": " Alice   Martin / Directrice ",
                                "00:03": "Bob Dupont / CTO",
                            },
                            "subtitle": {
                                "00:04": "Je m'appelle Alice Martin",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            texts, source = speaker_proposal.load_filtered_ocr_texts(video)

            self.assertEqual(
                texts,
                [
                    "IONIS",
                    "Alice Martin / Directrice",
                    "Bob Dupont / CTO",
                ],
            )
            self.assertEqual(source, filtered)

    def test_chunks_do_not_require_or_store_validated_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "video123.mp4"
            transcript = root / "plain_transcript.txt"
            target = root / "transcript_chunks.json"
            video.touch()
            transcript.write_text("Contenu canonique WhisperX.", encoding="utf-8")

            with (
                patch.object(chunk_creation, "source_text_path", return_value=transcript),
                patch.object(chunk_creation, "chunks_path", return_value=target),
            ):
                result = chunk_creation.create_chunks(video, force=True)

            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertNotIn("speakers_source", payload)
            self.assertNotIn("speakers", payload["chunks"][0])
            self.assertNotIn("meta_data", payload["chunks"][0])

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
                patch.object(
                    speaker_proposal,
                    "load_ocr_speaker_candidates",
                    return_value=(["Sophie Vanderpol"], None),
                ),
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
                patch.object(chunk_creation, "chunks_path", return_value=chunks),
            ):
                chunk_creation.create_chunks(video, force=True)

            validated_payload = json.loads(validated.read_text(encoding="utf-8"))
            chunks_payload = json.loads(chunks.read_text(encoding="utf-8"))
            self.assertEqual(validated_payload["speakers"], ["Sophie Vanderpol"])
            self.assertNotIn("speakers_source", chunks_payload)
            self.assertNotIn("speakers", chunks_payload["chunks"][0])
            self.assertNotIn("meta_data", chunks_payload["chunks"][0])
            self.assertEqual(chunks_payload["chunks"][0]["chunk_level"], "detail")
            self.assertIsNone(chunks_payload["chunks"][0]["chunk_parent_id"])


if __name__ == "__main__":
    unittest.main()
