from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.transcripts import create_plain_transcript as plain_transcript
from pipeline.steps.transcripts import normalize_ionis_stm as brand_normalization


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

    def test_empty_whisper_uses_graphic_and_others_from_filtered_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            ocr_dir = video_dir / "outputs" / "ocr"
            transcript_dir.mkdir(parents=True)
            ocr_dir.mkdir(parents=True)
            source = transcript_dir / plain_transcript.TRANSCRIPT_3_WITH_SPEAKERS_NAME
            source.write_text("\n", encoding="utf-8")
            (ocr_dir / "02_filtered_ocr_overlays.json").write_text(
                """
                {
                  "kinds": {
                    "graphic": {"00:05": "Titre graphique"},
                    "others": {
                      "00:03": "Premier texte",
                      "00:08": "Dernier texte"
                    },
                    "subtitle": {"00:01": "Sous-titre exclu"}
                  }
                }
                """,
                encoding="utf-8",
            )

            target = plain_transcript.convert_file(video_dir, source, force=True)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                (
                    "texte présent dans la vidéo : Premier texte "
                    "Titre graphique Dernier texte\n"
                ),
            )

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

    def test_long_video_plain_uses_raw_even_if_corrected_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            transcript_dir = (
                Path(temporary_directory)
                / "outputs"
                / plain_transcript.CANONICAL_TRANSCRIPTS_DIR_NAME
            )
            transcript_dir.mkdir(parents=True)
            raw = transcript_dir / plain_transcript.TRANSCRIPT_1_BRUT_NAME
            corrected = transcript_dir / plain_transcript.TRANSCRIPT_2_CORRECTED_NAME
            raw.write_text("[00:01-00:03] Texte brut.\n", encoding="utf-8")
            corrected.write_text(
                "[00:01-00:03] Texte corrige.\n",
                encoding="utf-8",
            )

            sources = plain_transcript.timecoded_inputs(
                transcript_dir,
                raw_only=True,
            )
            target = plain_transcript.convert_file(
                transcript_dir.parents[1],
                sources[0],
                force=True,
            )
            target_text = target.read_text(encoding="utf-8")

        self.assertEqual(sources, [raw])
        self.assertEqual(target_text, "Texte brut.\n")

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
            source = internal_ocr / "ocr_subtitles_timecoded.txt"
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

            target = plain_transcript.convert_ocr_file(
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

    def test_brand_normalization_uses_raw_ocr_and_removes_spacing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            video_dir.mkdir()
            video = video_dir / "video123.mp4"
            video.touch()
            ocr_dir = video_dir / "outputs" / "ocr"
            ocr_dir.mkdir(parents=True)
            raw = ocr_dir / "ocr_subtitles_timecoded.txt"
            obsolete = ocr_dir / "ocr_subtitles_timecoded_corrected.txt"
            raw.write_text("[00:01] Bonjour Ionis STM.\n", encoding="utf-8")
            obsolete.write_text("ancien résultat GPT", encoding="utf-8")

            result = brand_normalization.process_video(video, force=True)

            self.assertTrue(result)
            self.assertEqual(
                raw.read_text(encoding="utf-8"),
                "[00:01] Bonjour Ionis-STM.\n",
            )
            self.assertFalse(obsolete.exists())


if __name__ == "__main__":
    unittest.main()
