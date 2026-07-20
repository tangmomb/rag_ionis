from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.steps.transcripts import reconcile_whisper_with_ocr as reconciliation


class ReconcileWhisperOcrTests(unittest.TestCase):
    def test_luna_corrects_bodies_while_python_preserves_line_structure(self) -> None:
        class Responses:
            def __init__(self) -> None:
                self.request = None

            def create(self, **request):
                self.request = request
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "segments": [
                                {
                                    "index": 0,
                                    "text": (
                                        "Avant de rejoindre Ionis-STM pour un "
                                        "pré-MSc."
                                    ),
                                },
                                {
                                    "index": 1,
                                    "text": (
                                        "Je travaille chez Bouygues Énergies "
                                        "et Services."
                                    ),
                                },
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

        responses = Responses()
        client = SimpleNamespace(responses=responses)
        whisper = (
            "[00:00-00:05] SPEAKER_00: "
            "Avant de rejoindre UNISSTM pour un pré-MSc.\n"
            "[00:05-00:10] SPEAKER_01: "
            "Je travaille chez Bougainé RG Services.\n"
        )

        corrected, changes = reconciliation.reconcile_transcripts_with_luna(
            client,
            "gpt-5.6-luna",
            whisper,
            (
                "Avant de rejoindre Ionis-STM pour un pré-MSc. "
                "Je travaille chez Bouygues Énergies et Services."
            ),
        )

        self.assertIn(
            "[00:00-00:05] SPEAKER_00: "
            "Avant de rejoindre Ionis-STM pour un pré-MSc.",
            corrected,
        )
        self.assertIn(
            "[00:05-00:10] SPEAKER_01: "
            "Je travaille chez Bouygues Énergies et Services.",
            corrected,
        )
        self.assertIn(("UNISSTM", "Ionis-STM"), changes)
        self.assertIn(("Bougainé RG", "Bouygues Énergies et"), changes)
        self.assertEqual(responses.request["model"], "gpt-5.6-luna")

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

            with patch.object(
                reconciliation,
                "reconcile_transcripts_with_luna",
                return_value=(
                    "[00:00-00:05] SPEAKER_00: Ionis-STM.\n",
                    [("ionis stm", "Ionis-STM")],
                ),
            ):
                corrected = reconciliation.reconcile_file(
                    video,
                    force=True,
                    client=object(),
                )

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

    def test_file_step_can_finalize_a_batch_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            whisper_dir = video_dir / "outputs" / "transcripts_whisper"
            ocr_dir = video_dir / "outputs" / "transcripts_ocr"
            whisper_dir.mkdir(parents=True)
            ocr_dir.mkdir(parents=True)
            video = video_dir / "video123.mp4"
            video.touch()
            (whisper_dir / reconciliation.WHISPER_SOURCE_NAME).write_text(
                "[00:00-00:05] SPEAKER_00: ionis stm.\n",
                encoding="utf-8",
            )
            (ocr_dir / "plain_transcript.txt").write_text(
                "Ionis-STM.\n",
                encoding="utf-8",
            )
            output_path = (
                whisper_dir / reconciliation.BATCH_OUTPUT_NAME
            )
            output_path.write_text(
                json.dumps(
                    {
                        "custom_id": "transcript-reconciliation",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "output_text": json.dumps(
                                    {
                                        "segments": [
                                            {
                                                "index": 0,
                                                "text": "Ionis-STM.",
                                            }
                                        ]
                                    }
                                )
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            state = {
                "status": "completed",
                "output_file_id": "output-file",
            }
            with (
                patch("openai.OpenAI", return_value=object()),
                patch.object(
                    reconciliation,
                    "load_batch_state",
                    return_value=state,
                ),
                patch.object(
                    reconciliation,
                    "batch_state_matches",
                    return_value=True,
                ),
                patch.object(
                    reconciliation,
                    "poll_batch_state",
                    return_value=state,
                ),
                patch.object(
                    reconciliation,
                    "download_batch_files",
                ),
            ):
                corrected = reconciliation.reconcile_file(
                    video,
                    mode="batch",
                )

            corrected_text = corrected.read_text(encoding="utf-8")

        self.assertIn("Ionis-STM.", corrected_text)


if __name__ == "__main__":
    unittest.main()
