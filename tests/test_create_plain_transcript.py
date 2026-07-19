from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.transcripts import create_plain_transcript as plain_transcript


class CreatePlainTranscriptTests(unittest.TestCase):
    def test_empty_whisper_stays_empty_even_when_an_enriched_ocr_file_exists(self) -> None:
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
            target = plain_transcript.convert_file(video, source, force=True)

            self.assertEqual(target.read_text(encoding="utf-8"), "\n")

    def test_regular_whisper_plain_transcript_has_no_ocr_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory)
            video = video_dir / "video.mp4"
            video.touch()
            source = video_dir / "whisper_transcript_timecoded_corrected.txt"
            source.write_text("[00:00-00:05] SPEAKER_00: Bonjour.\n", encoding="utf-8")
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

    def test_plain_transcript_excludes_intercalaires(self) -> None:
        text = (
            "[00:01] Alice Martin: Bonjour.\n"
            "[00:03] INTERCALAIRE: COMMENT TU T'APPELLES ?\n"
            "[00:05] Alice Martin: Voici mon parcours.\n"
        )

        self.assertEqual(
            plain_transcript.strip_timecodes(text, ["Alice Martin"]),
            "Bonjour. Voici mon parcours.",
        )

    def test_canonical_plain_prefers_transcript_with_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            transcript_dir = (
                Path(temporary_directory)
                / "outputs"
                / plain_transcript.CANONICAL_TRANSCRIPTS_DIR_NAME
            )
            transcript_dir.mkdir(parents=True)
            with_speakers = transcript_dir / plain_transcript.TRANSCRIPT_3_WITH_SPEAKERS_NAME
            enriched = transcript_dir / plain_transcript.TRANSCRIPT_ENRICHED_NAME
            with_speakers.write_text("[00:01] Alice Martin: Bonjour.\n", encoding="utf-8")
            enriched.write_text(
                "[00:01] Alice Martin: Bonjour.\n"
                "[00:03] INTERCALAIRE: COMMENT TU T'APPELLES ?\n",
                encoding="utf-8",
            )

            self.assertEqual(
                plain_transcript.timecoded_inputs(transcript_dir),
                [with_speakers],
            )

    def test_ocr_directory_contains_only_plain_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            video_dir.mkdir()
            video = video_dir / "video123.mp4"
            video.touch()
            internal_ocr = video_dir / "outputs" / "ocr"
            public_ocr = video_dir / "outputs" / "transcripts_ocr"
            internal_ocr.mkdir(parents=True)
            public_ocr.mkdir(parents=True)
            source = internal_ocr / "ocr_subtitles_timecoded_corrected.txt"
            source.write_text(
                "[00:01] Bonjour Ionis-STM.\n[00:03] Suite.\n",
                encoding="utf-8",
            )
            (public_ocr / "ocr_subtitles_timecoded.txt").write_text(
                "obsolete",
                encoding="utf-8",
            )
            (public_ocr / "ocr_subtitles_timecoded_corrected.txt").write_text(
                "obsolete",
                encoding="utf-8",
            )

            target = plain_transcript.convert_ocr_correction_file(
                video,
                source,
                force=True,
            )

            self.assertEqual(
                [path.name for path in public_ocr.iterdir()],
                ["plain_transcript.txt"],
            )
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "Bonjour Ionis-STM. Suite.\n",
            )


if __name__ == "__main__":
    unittest.main()
