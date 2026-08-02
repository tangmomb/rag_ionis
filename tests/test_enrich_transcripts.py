from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.transcripts import enrich_transcripts as enrichment


class EnrichTranscriptsTests(unittest.TestCase):
    def test_overlay_label_is_only_intercalaire(self) -> None:
        self.assertEqual(
            enrichment.format_intercalaire_line(
                {"second": 5, "text": "Titre"}
            ),
            "[00:05] INTERCALAIRE: Titre",
        )

    def test_only_graphic_ocr_items_are_loaded_as_intercalaires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            overlays = Path(temporary_directory) / "overlays.json"
            overlays.write_text(
                """
                {
                  "kinds": {
                    "graphic": {
                      "00:05": "Chapitre",
                      "00:05#2": "Sous-chapitre",
                      "00:10": "Conclusion"
                    },
                    "name": {"00:02": "Alice Martin"},
                    "title": {"00:03": "Texte animé"},
                    "subtitle": {"00:04": "Sous-titre"}
                  }
                }
                """,
                encoding="utf-8",
            )

            items = enrichment.load_intercalaires(overlays)

        self.assertEqual(
            items,
            [
                {"second": 5, "text": "Chapitre"},
                {"second": 5, "text": "Sous-chapitre"},
                {"second": 10, "text": "Conclusion"},
            ],
        )

    def test_duplicate_timecode_suffix_is_ignored_when_parsing(self) -> None:
        self.assertEqual(enrichment.parse_timecode("00:03#2"), 3)
        self.assertEqual(enrichment.parse_timecode("01:02:03#12"), 3723)

    def test_enriched_file_combines_speakers_and_intercalaires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            video_dir.mkdir()
            video = video_dir / "video123.mp4"
            video.touch()
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            ocr_dir = video_dir / "outputs" / "ocr"
            transcript_dir.mkdir(parents=True)
            ocr_dir.mkdir(parents=True)
            source = transcript_dir / "transcript_2_corrected.txt"
            source.write_text(
                "[00:00-00:05] Alice Martin: Bonjour.\n"
                "[00:06-00:10] Bob Durand: Suite.\n",
                encoding="utf-8",
            )
            overlays = ocr_dir / "02_filtered_ocr_overlays.json"
            overlays.write_text(
                """
                {
                  "kinds": {
                    "graphic": {"00:05": "Chapitre 1"},
                    "name": {"00:02": "Alice Martin"},
                    "title": {"00:03": "Titre animé"},
                    "subtitle": {"00:04": "Sous-titre"}
                  }
                }
                """,
                encoding="utf-8",
            )

            with (
                patch.object(
                    enrichment,
                    "corrected_timecodes_path",
                    return_value=source,
                ),
                patch.object(
                    enrichment,
                    "enriched_ocr_source_path",
                    return_value=overlays,
                ),
                patch.object(
                    enrichment,
                    "correct_speaker_files",
                    return_value=0,
                ),
            ):
                target = enrichment.enrich_transcript(video, force=True)

            self.assertEqual(target.name, "transcript_3_enriched.txt")
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "[00:00-00:05] Alice Martin: Bonjour.\n"
                "[00:06-00:10] Bob Durand: Suite.\n"
                "\n"
                "[00:05] INTERCALAIRE: Chapitre 1\n",
            )

    def test_missing_ocr_produces_an_enriched_speaker_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            video_dir.mkdir()
            video = video_dir / "video123.mp4"
            video.touch()
            transcript_dir = video_dir / "outputs" / "transcripts_whisper"
            transcript_dir.mkdir(parents=True)
            source = transcript_dir / "transcript_2_corrected.txt"
            source.write_text(
                "[00:00-00:05] Alice Martin: Bonjour.\n",
                encoding="utf-8",
            )
            missing_overlays = video_dir / "outputs" / "ocr" / "missing.json"

            with (
                patch.object(
                    enrichment,
                    "corrected_timecodes_path",
                    return_value=source,
                ),
                patch.object(
                    enrichment,
                    "enriched_ocr_source_path",
                    return_value=missing_overlays,
                ),
                patch.object(
                    enrichment,
                    "correct_speaker_files",
                    return_value=0,
                ),
            ):
                target = enrichment.enrich_transcript(video, force=True)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "[00:00-00:05] Alice Martin: Bonjour.\n",
            )


if __name__ == "__main__":
    unittest.main()
