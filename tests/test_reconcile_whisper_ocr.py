from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.steps.transcripts import reconcile_whisper_with_ocr as reconciliation


class ReconcileWhisperOcrTests(unittest.TestCase):
    def test_close_compound_is_replaced_with_the_ocr_form(self) -> None:
        whisper = (
            "[00:00-00:06] SPEAKER_00: "
            "Bienvenue a ionis stm aujourd'hui.\n"
        )
        ocr = "[00:02] Bienvenue à Ionis-STM aujourd'hui.\n"

        corrected, changes = reconciliation.reconcile_transcripts(
            whisper,
            ocr,
        )

        self.assertEqual(
            corrected,
            (
                "[00:00-00:06] SPEAKER_00: "
                "Bienvenue a Ionis-STM aujourd'hui.\n"
            ),
        )
        self.assertIn(("ionis stm", "Ionis-STM"), changes)

    def test_fuzzy_cousin_is_corrected(self) -> None:
        corrected, changes = reconciliation.reconcile_transcripts(
            "[00:10-00:15] SPEAKER_00: Je rejoins ionos stm.\n",
            "[00:12] Je rejoins Ionis-STM.\n",
        )

        self.assertIn("Ionis-STM", corrected)
        self.assertIn(("ionos stm", "Ionis-STM"), changes)

    def test_single_proper_name_inside_a_subtitle_can_be_corrected(self) -> None:
        corrected, changes = reconciliation.reconcile_transcripts(
            "[00:10-00:15] SPEAKER_00: Je suis Lucif.\n",
            "[00:12] Je suis Loucif.\n",
        )

        self.assertIn("Je suis Loucif.", corrected)
        self.assertIn(("Lucif", "Loucif"), changes)

    def test_plain_ocr_transcript_can_correct_whisper_without_timecodes(self) -> None:
        corrected, changes = reconciliation.reconcile_transcripts(
            "[00:10-00:15] SPEAKER_00: Je rejoins ionis stm.\n",
            "Bienvenue à Ionis-STM. Présentation des formations.\n",
        )

        self.assertIn("Je rejoins Ionis-STM.", corrected)
        self.assertIn(("ionis stm", "Ionis-STM"), changes)

    def test_plain_ocr_never_rewrites_the_whisper_speaker_prefix(self) -> None:
        corrected, changes = reconciliation.reconcile_transcripts(
            (
                "[00:37-00:44] SPEAKER_00: C'est pourquoi "
                "j'ai rejoint IONIS STM.\n"
            ),
            "C'est pourquoi j'ai rejoint Ionis-STM.\n",
        )

        self.assertIn(
            "[00:37-00:44] SPEAKER_00: C'est pourquoi",
            corrected,
        )
        self.assertNotIn("SPEAKER_C'est", corrected)
        self.assertIn(("IONIS STM", "Ionis-STM"), changes)

    def test_plain_ocr_cannot_drop_words_from_whisper(self) -> None:
        whisper = (
            "[00:37-00:44] SPEAKER_00: C'est pourquoi j'ai rejoint IONIS STM.\n"
            "[01:01-01:03] SPEAKER_00: J'espère toutefois être fier du chemin parcouru.\n"
        )

        corrected, changes = reconciliation.reconcile_transcripts(
            whisper,
            (
                "C'est pourquoi rejoint Ionis-STM. "
                "J'espère toutefois fier du chemin parcouru.\n"
            ),
        )

        self.assertIn("C'est pourquoi j'ai rejoint Ionis-STM.", corrected)
        self.assertIn("J'espère toutefois être fier", corrected)
        self.assertNotIn(
            ("j'ai rejoint IONIS STM", "rejoint Ionis-STM"),
            changes,
        )

    def test_distant_ocr_line_does_not_modify_whisper(self) -> None:
        whisper = "[00:00-00:04] SPEAKER_00: Je rejoins ionis stm.\n"

        corrected, changes = reconciliation.reconcile_transcripts(
            whisper,
            "[00:30] Je rejoins Ionis-STM.\n",
        )

        self.assertEqual(corrected, whisper)
        self.assertEqual(changes, [])

    def test_ordinary_sentence_casing_is_not_copied_from_ocr(self) -> None:
        whisper = "[00:00-00:04] SPEAKER_00: nous sommes ici.\n"

        corrected, changes = reconciliation.reconcile_transcripts(
            whisper,
            "[00:01] Nous sommes ici.\n",
        )

        self.assertEqual(corrected, whisper)
        self.assertEqual(changes, [])

    def test_file_step_uses_the_single_plain_ocr_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            whisper_dir = video_dir / "outputs" / "transcripts_whisper"
            ocr_dir = video_dir / "outputs" / "transcripts_ocr"
            whisper_dir.mkdir(parents=True)
            ocr_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            raw_whisper = whisper_dir / reconciliation.WHISPER_SOURCE_NAME
            raw_whisper.write_text(
                "[00:00-00:05] SPEAKER_00: ionis stm.\n",
                encoding="utf-8",
            )
            raw_ocr = ocr_dir / "plain_transcript.txt"
            raw_ocr.write_text("Ionis-STM.\n", encoding="utf-8")

            corrected = reconciliation.reconcile_file(video, force=True)

            self.assertEqual(
                raw_whisper.read_text(encoding="utf-8"),
                "[00:00-00:05] SPEAKER_00: ionis stm.\n",
            )
            self.assertEqual(
                raw_ocr.read_text(encoding="utf-8"),
                "Ionis-STM.\n",
            )
            self.assertEqual(
                corrected.read_text(encoding="utf-8"),
                "[00:00-00:05] SPEAKER_00: Ionis-STM.\n",
            )
            self.assertEqual(
                reconciliation.corrections_path(video).read_text(
                    encoding="utf-8"
                ),
                "ionis stm\tIonis-STM\n",
            )


if __name__ == "__main__":
    unittest.main()
