from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.transcripts import transcribe_with_whisper as transcription


class WhisperXDiarizationTests(unittest.TestCase):
    def test_timestamped_transcript_includes_speaker_ids(self) -> None:
        text = transcription.format_timestamped_transcript(
            [
                {
                    "start": 1,
                    "end": 4,
                    "speaker": "SPEAKER_00",
                    "text": "Bonjour.",
                },
                {
                    "start": 5,
                    "end": 8,
                    "speaker": "SPEAKER_01",
                    "text": "Bienvenue.",
                },
            ]
        )

        self.assertEqual(
            text,
            "[00:01-00:04] SPEAKER_00: Bonjour.\n"
            "[00:05-00:08] SPEAKER_01: Bienvenue.",
        )

    def test_timestamped_transcript_still_supports_segments_without_speaker(self) -> None:
        text = transcription.format_timestamped_transcript(
            [{"start": 1, "end": 4, "text": "Bonjour."}]
        )

        self.assertEqual(text, "[00:01-00:04] Bonjour.")

    def test_transcription_runs_diarization_and_assigns_speakers(self) -> None:
        class FakeModel:
            def transcribe(self, _audio_path, **_kwargs):
                return {
                    "language": "fr",
                    "segments": [{"start": 1, "end": 4, "text": "Bonjour."}],
                }

        class FakeWhisperX:
            @staticmethod
            def load_align_model(**_kwargs):
                return object(), {}

            @staticmethod
            def align(segments, *_args, **_kwargs):
                return {"segments": segments}

            @staticmethod
            def assign_word_speakers(diarized_segments, aligned):
                self.assertEqual(diarized_segments, "DIARIZED")
                aligned["segments"][0]["speaker"] = "SPEAKER_00"
                return aligned

        diarization_calls = []

        def diarization_pipeline(audio_path, **kwargs):
            diarization_calls.append((audio_path, kwargs))
            return "DIARIZED"

        with patch.object(transcription, "torch", None):
            text, speakers = transcription.transcribe_with_whisperx(
                FakeWhisperX(),
                FakeModel(),
                Path("audio.wav"),
                "cpu",
                diarization_pipeline=diarization_pipeline,
                min_speakers=1,
                max_speakers=3,
            )

        self.assertEqual(text, "[00:01-00:04] SPEAKER_00: Bonjour.")
        self.assertEqual(speakers, ["SPEAKER_00"])
        self.assertEqual(
            diarization_calls,
            [("audio.wav", {"min_speakers": 1, "max_speakers": 3})],
        )

    def test_gated_model_error_explains_how_to_unlock_access(self) -> None:
        from huggingface_hub.errors import GatedRepoError

        diarize_module = ModuleType("whisperx.diarize")

        class RefusedPipeline:
            def __init__(self, **_kwargs):
                raise GatedRepoError("access refused")

        diarize_module.DiarizationPipeline = RefusedPipeline
        with (
            patch.dict(sys.modules, {"whisperx.diarize": diarize_module}),
            patch.object(transcription, "huggingface_token", return_value="hf_test"),
            patch.object(transcription, "resolved_device", return_value="cpu"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Accepte l'acces avec le meme compte",
            ):
                transcription.load_diarization_pipeline("cpu")


if __name__ == "__main__":
    unittest.main()
