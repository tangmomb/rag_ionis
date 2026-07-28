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
from pipeline.steps.transcripts import enrich_transcripts as enrichment
from pipeline.steps.speakers import correct_speaker_transcripts as speaker_correction


class CorrectSpeakerTranscriptsTests(unittest.TestCase):
    def test_enrichment_applies_speakers_without_intermediate_file(self) -> None:
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
            (transcript_dir / "transcript_1_brut.txt").write_text(
                "[00:00-00:05] SPEAKER_00: Bonjour, je m'appelle Lucie Ouyaya.\n",
                encoding="utf-8",
            )
            (ocr_dir / "01_processed_ocr_items.json").write_text(
                '{"items":[]}',
                encoding="utf-8",
            )
            (ocr_dir / "02_filtered_ocr_overlays.json").write_text(
                '{"kinds":{"graphic":{}}}',
                encoding="utf-8",
            )
            (speakers_dir / "speaker_candidates.json").write_text(
                json.dumps(
                    {
                        "source": "outputs/transcripts_whisper/transcript_1_brut.txt",
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
                transcript_dir / "transcript_2_corrected.txt",
            )
            text = result.read_text(encoding="utf-8")
            self.assertIn("Lucie Ouyaya", text)
            self.assertNotIn("Loucif Ouyahia", text)

            enriched = enrichment.enrich_transcript(video, force=True)

            self.assertEqual(
                enriched,
                transcript_dir / "transcript_3_enriched.txt",
            )
            self.assertIn(
                "Loucif Ouyahia",
                enriched.read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "SPEAKER_00",
                enriched.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Lucie Ouyaya",
                result.read_text(encoding="utf-8"),
            )
            self.assertFalse(
                (speakers_dir / "speaker_transcript_corrections.json").exists()
            )
            self.assertFalse(
                (transcript_dir / "transcript_3_with_speakers.txt").exists()
            )

    def test_single_validated_speaker_replaces_every_whisper_label(self) -> None:
        corrected, counts = speaker_correction.apply_speaker_labels(
            (
                "[00:00-00:05] SPEAKER_00: Bonjour.\n"
                "[00:05-00:10] SPEAKER_01: Suite.\n"
            ),
            ["Matthieu"],
        )

        self.assertEqual(
            corrected,
            (
                "[00:00-00:05] Matthieu: Bonjour.\n"
                "[00:05-00:10] Matthieu: Suite.\n"
            ),
        )
        self.assertEqual(
            counts,
            {"SPEAKER_00": 1, "SPEAKER_01": 1},
        )

    def test_multiple_speakers_are_mapped_from_their_introductions(self) -> None:
        corrected, counts = speaker_correction.apply_speaker_labels(
            (
                "[00:00-00:01] SPEAKER_00: Introduction.\n"
                "[00:01-00:07] SPEAKER_01: Je suis Sophie Vanderpol, Fondatrice.\n"
                "[00:16-00:21] SPEAKER_00: Je m'appelle Hugo Jarguin.\n"
                "[00:33-00:40] SPEAKER_02: Je m'appelle Clara Dalmada.\n"
            ),
            ["Sophie Vanderpol", "Hugo Jarguin", "Clara Dalmada"],
        )

        self.assertIn("Hugo Jarguin: Introduction.", corrected)
        self.assertIn("Sophie Vanderpol: Je suis Sophie Vanderpol", corrected)
        self.assertIn("Hugo Jarguin: Je m'appelle Hugo Jarguin", corrected)
        self.assertIn("Clara Dalmada: Je m'appelle Clara Dalmada", corrected)
        self.assertEqual(
            counts,
            {"SPEAKER_00": 2, "SPEAKER_01": 1, "SPEAKER_02": 1},
        )

    def test_ocr_name_timecodes_override_numeric_label_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            video_dir.mkdir()
            video = video_dir / "video123.mp4"
            video.touch()
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            speakers_dir = video_dir / "outputs" / "speakers"
            ocr_dir = video_dir / "outputs" / "ocr"
            transcript_dir.mkdir(parents=True)
            speakers_dir.mkdir(parents=True)
            ocr_dir.mkdir(parents=True)

            enriched = transcript_dir / "transcript_3_enriched.txt"
            enriched.write_text(
                "[00:12-00:20] SPEAKER_01: Présentation de la rencontre.\n"
                "[00:25-00:36] SPEAKER_03: Premier témoignage.\n"
                "[00:36-00:47] SPEAKER_02: Deuxième témoignage.\n"
                "[00:51-01:00] SPEAKER_00: Troisième témoignage.\n",
                encoding="utf-8",
            )
            (speakers_dir / "speaker_candidates.json").write_text(
                json.dumps({"speakers": []}),
                encoding="utf-8",
            )
            (speakers_dir / "speakers_validated.json").write_text(
                json.dumps(
                    {
                        "speakers": [
                            "Valérie Pham-Trong",
                            "Eric Modesto",
                            "Cyril Morcrette",
                            "Jérôme Hannebelle",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (ocr_dir / "01_processed_ocr_items.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "kind": "others",
                                "text": "Valérie Pham-Trong",
                                "second": 17,
                            },
                            {
                                "kind": "others",
                                "text": "Eric Modesto",
                                "second": 29.5,
                            },
                            {
                                "kind": "others",
                                "text": "Cyril Morcrette",
                                "second": 41,
                            },
                            {
                                "kind": "others",
                                "text": "Jérôme Hannebelle",
                                "second": 55,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = speaker_correction.correct_speaker_files(
                video,
                [enriched],
                force=True,
                replace_speaker_labels=True,
            )

            self.assertEqual(result, 4)
            self.assertEqual(
                enriched.read_text(encoding="utf-8"),
                "[00:12-00:20] Valérie Pham-Trong: Présentation de la rencontre.\n"
                "[00:25-00:36] Eric Modesto: Premier témoignage.\n"
                "[00:36-00:47] Cyril Morcrette: Deuxième témoignage.\n"
                "[00:51-01:00] Jérôme Hannebelle: Troisième témoignage.\n",
            )

    def test_ocr_hint_has_priority_over_conflicting_introduction(self) -> None:
        corrected, counts = speaker_correction.apply_speaker_labels(
            (
                "[00:00-00:05] SPEAKER_01: Je suis Bob Durand.\n"
                "[00:05-00:10] SPEAKER_00: Suite.\n"
            ),
            ["Alice Martin", "Bob Durand"],
            label_hints={"SPEAKER_01": "Alice Martin"},
        )

        self.assertEqual(
            corrected,
            (
                "[00:00-00:05] Alice Martin: Je suis Bob Durand.\n"
                "[00:05-00:10] Bob Durand: Suite.\n"
            ),
        )
        self.assertEqual(
            counts,
            {"SPEAKER_01": 1, "SPEAKER_00": 1},
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
            source = transcript_dir / "ocr_subtitles_timecoded.txt"
            source.write_text("[00:01] Lucie Ouyaya\n", encoding="utf-8")
            enriched = transcript_dir / "ocr_subtitles_timecoded_enriched.txt"
            enriched.write_text("[00:01] Lucie Ouyaya\n", encoding="utf-8")
            (speakers_dir / "speaker_candidates.json").write_text(
                json.dumps(
                    {
                        "source": "outputs/transcripts_ocr/ocr_subtitles_timecoded.txt",
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

            result = speaker_correction.correct_speaker_transcripts(video, force=True)

            self.assertEqual(result, 2)
            self.assertIn("Loucif Ouyahia", plain.read_text(encoding="utf-8"))
            self.assertIn("Loucif Ouyahia", timecoded.read_text(encoding="utf-8"))
            self.assertIn("SPEAKER_00:", timecoded.read_text(encoding="utf-8"))
            self.assertEqual(list(transcript_dir.glob("*_speaker_corrected.txt")), [])
            self.assertFalse(
                (speakers_dir / "speaker_transcript_corrections.json").exists()
            )

            chunks = video_dir / "outputs" / "chunks" / "transcript_chunks.json"
            with patch.object(chunk_creation, "chunks_path", return_value=chunks):
                chunks_path = chunk_creation.create_chunks(video, force=True)

            chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.assertEqual(
                chunks_payload["source"],
                "outputs/transcripts_whisper/plain_transcript.txt",
            )
            self.assertIn("Loucif Ouyahia", chunks_payload["chunks"][0]["content"])

    def test_chunks_are_obsolete_only_when_the_plain_transcript_is_newer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transcript = root / "plain_transcript.txt"
            chunks = root / "transcript_chunks.json"
            transcript.write_text("Nouveau transcript", encoding="utf-8")
            chunks.write_text("{}", encoding="utf-8")
            os.utime(chunks, (100, 100))
            os.utime(transcript, (300, 300))

            self.assertFalse(
                chunk_creation.output_is_current(chunks, (transcript,))
            )
            os.utime(chunks, (400, 400))
            self.assertTrue(
                chunk_creation.output_is_current(chunks, (transcript,))
            )


if __name__ == "__main__":
    unittest.main()
