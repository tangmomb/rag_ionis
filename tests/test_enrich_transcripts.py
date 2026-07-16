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

from common import used_by_hs19_ns18_enrich_transcripts as enrichment


class EnrichTranscriptsTests(unittest.TestCase):
    def test_blocks_are_grouped_by_type_and_chronological_inside_each_group(self) -> None:
        lines = enrichment.grouped_lines(
            [
                {"group": "speaker", "line": "[00:00] Alice: Bonjour."},
                {"group": "animations", "line": "[00:03] ANIMATIONS: Titre"},
                {"group": "intercalaire", "line": "[00:05] INTERCALAIRE: Chapitre"},
                {"group": "speaker", "line": "[00:06] Alice: Suite."},
                {"group": "animations", "line": "[00:08] ANIMATIONS: Sous-titre"},
                {"group": "intercalaire", "line": "[00:10] INTERCALAIRE: Conclusion"},
            ]
        )

        self.assertEqual(
            lines,
            [
                "[00:00] Alice: Bonjour.",
                "[00:06] Alice: Suite.",
                "",
                "[00:03] ANIMATIONS: Titre",
                "[00:08] ANIMATIONS: Sous-titre",
                "",
                "[00:05] INTERCALAIRE: Chapitre",
                "[00:10] INTERCALAIRE: Conclusion",
            ],
        )

    def test_overlay_labels_use_intercalaire_and_animations(self) -> None:
        self.assertEqual(
            enrichment.format_overlay_line(
                {"kind": "graphic", "second": 5, "text": "Titre"}
            ),
            "[00:05] INTERCALAIRE: Titre",
        )
        self.assertEqual(
            enrichment.format_overlay_line(
                {"kind": "name", "second": 8, "text": "Texte animé"}
            ),
            "[00:08] ANIMATIONS: Texte animé",
        )

    def test_one_validated_speaker_replaces_speaker_00(self) -> None:
        text, count = enrichment.replace_speaker_labels(
            "SPEAKER_00: Bonjour. SPEAKER_01: Inconnu.",
            ["Loucif Ouyahia"],
        )

        self.assertEqual(text, "Loucif Ouyahia: Bonjour. SPEAKER_01: Inconnu.")
        self.assertEqual(count, 1)

    def test_multiple_validated_speakers_replace_labels_in_order(self) -> None:
        text, count = enrichment.replace_speaker_labels(
            "SPEAKER_01: Deux. SPEAKER_00: Un. SPEAKER_02: Trois.",
            ["Alice Martin", "Bob Durand"],
        )

        self.assertEqual(
            text,
            "Bob Durand: Deux. Alice Martin: Un. SPEAKER_02: Trois.",
        )
        self.assertEqual(count, 2)

    def test_enriched_file_uses_validated_speaker_name(self) -> None:
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
            source = transcript_dir / "whisper_transcript_timecoded_corrected.txt"
            source.write_text(
                "[00:00-00:05] SPEAKER_00: Bonjour.\n",
                encoding="utf-8",
            )
            overlays = ocr_dir / "03_reviewed_ocr_overlays.json"
            overlays.write_text('{"kinds":{}}', encoding="utf-8")
            (speakers_dir / "speakers_validated.json").write_text(
                json.dumps({"speakers": ["Alice Martin"]}),
                encoding="utf-8",
            )

            with (
                patch.object(enrichment, "timecodes_path", return_value=source),
                patch.object(enrichment, "enriched_ocr_source_path", return_value=overlays),
                patch.object(enrichment, "analysed_video_type", return_value="interview"),
                patch.object(enrichment, "update_analysed_infos"),
            ):
                target = enrichment.enrich_transcript(video, force=True)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "[00:00-00:05] Alice Martin: Bonjour.\n",
            )


if __name__ == "__main__":
    unittest.main()
