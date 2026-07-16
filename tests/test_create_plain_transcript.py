from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.transcripts import create_plain_transcript as plain_transcript


class CreatePlainTranscriptTests(unittest.TestCase):
    def test_empty_motion_design_whisper_prefixes_ocr_plain_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory)
            video = video_dir / "video.mp4"
            video.touch()
            source = video_dir / "whisper_transcript_timecoded_corrected.txt"
            source.write_text("", encoding="utf-8")
            enriched = video_dir / "whisper_transcript_timecoded_corrected_enriched.txt"
            enriched.write_text(
                "[00:01] Premier texte\n[00:05] Deuxième texte\n",
                encoding="utf-8",
            )
            raw_whisper = video_dir / "whisper_transcript_timecoded.txt"
            raw_whisper.write_text("", encoding="utf-8")

            with (
                patch.object(plain_transcript, "analysed_video_type", return_value="motion_design"),
                patch.object(plain_transcript, "whisper_timecoded_path", return_value=raw_whisper),
                patch.object(plain_transcript, "update_analysed_infos"),
            ):
                target = plain_transcript.convert_file(video, source, force=True)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "Textes présents sur la vidéo :\nPremier texte Deuxième texte\n",
            )

    def test_regular_whisper_plain_transcript_has_no_ocr_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory)
            video = video_dir / "video.mp4"
            video.touch()
            source = video_dir / "whisper_transcript_timecoded_corrected.txt"
            source.write_text("[00:00-00:05] SPEAKER_00: Bonjour.\n", encoding="utf-8")
            raw_whisper = video_dir / "whisper_transcript_timecoded.txt"
            raw_whisper.write_text("[00:00-00:05] SPEAKER_00: Bonjour.\n", encoding="utf-8")

            with (
                patch.object(plain_transcript, "analysed_video_type", return_value="motion_design"),
                patch.object(plain_transcript, "whisper_timecoded_path", return_value=raw_whisper),
                patch.object(plain_transcript, "update_analysed_infos"),
            ):
                target = plain_transcript.convert_file(video, source, force=True)

            self.assertEqual(target.read_text(encoding="utf-8"), "Bonjour.\n")

    def test_validated_person_name_prefix_is_removed_from_plain_transcript(self) -> None:
        text = (
            "[00:01] Alice Martin: Bonjour.\n"
            "[00:03] Alice Martin: Voici mon parcours.\n"
        )

        self.assertEqual(
            plain_transcript.strip_timecodes(text, ["Alice Martin"]),
            "Bonjour. Voici mon parcours.",
        )


if __name__ == "__main__":
    unittest.main()
