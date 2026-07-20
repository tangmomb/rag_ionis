import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import transcribe_mp4_openai as transcription


class TranscribeMp4OpenAiTests(unittest.TestCase):
    def test_zero_selects_all_models(self):
        with patch("builtins.input", return_value="0"):
            selected = transcription.prompt_model_choices()

        self.assertEqual(selected, transcription.MODEL_OPTIONS)

    def test_outputs_are_named_per_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "transcriptions_openai"
            video_path = Path(temporary_directory) / "interview.mp4"
            payload = {"text": "Bonjour."}

            with patch.object(transcription, "OUTPUT_DIR", output_dir):
                first_text, _ = transcription.write_outputs(
                    video_path, payload, transcription.MODEL_OPTIONS[0]
                )
                second_text, _ = transcription.write_outputs(
                    video_path, payload, transcription.MODEL_OPTIONS[1]
                )

            self.assertNotEqual(first_text, second_text)
            self.assertTrue(output_dir.is_dir())
            self.assertTrue(first_text.exists())
            self.assertTrue(second_text.exists())
