from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.speakers import assign_ocr_speakers as assignment


class AssignOcrSpeakersTests(unittest.TestCase):
    def build_video(self, root, speakers, transcript, ocr_items=None):
        video_dir = Path(root) / "video123"
        video_dir.mkdir()
        video = video_dir / "video123.mp4"
        video.touch()
        transcript_dir = video_dir / "outputs" / "transcripts_ocr"
        speakers_dir = video_dir / "outputs" / "speakers"
        ocr_dir = video_dir / "outputs" / "ocr"
        transcript_dir.mkdir(parents=True)
        speakers_dir.mkdir(parents=True)
        ocr_dir.mkdir(parents=True)
        (transcript_dir / "ocr_subtitles_timecoded.txt").write_text(
            transcript,
            encoding="utf-8",
        )
        (speakers_dir / "speakers_validated.json").write_text(
            json.dumps({"speakers": speakers}),
            encoding="utf-8",
        )
        (ocr_dir / "01_processed_ocr_items.json").write_text(
            json.dumps({"items": ocr_items or []}),
            encoding="utf-8",
        )
        return video

    def test_single_validated_speaker_is_applied_without_diarization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.build_video(
                temporary_directory,
                ["Alice Martin"],
                "[00:01] Bonjour.\n[00:03] Je vous presente mon parcours.\n",
            )

            target = assignment.assign_ocr_speakers(video, force=True)

            transcript = (
                video.parent
                / "outputs"
                / "transcripts_ocr"
                / "ocr_subtitles_timecoded.txt"
            ).read_text(encoding="utf-8")
            self.assertEqual(
                transcript,
                "[00:01] Alice Martin: Bonjour.\n"
                "[00:03] Alice Martin: Je vous presente mon parcours.\n",
            )
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertFalse(payload["diarization"]["enabled"])
            self.assertEqual(payload["speaker_mapping"][0]["method"], "single_validated_speaker")

    def test_multiple_speakers_use_introduction_and_ocr_name_timecodes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.build_video(
                temporary_directory,
                ["Alice Martin", "Bob Durand"],
                "[00:01] Bonjour, je m'appelle Alice Martin.\n"
                "[00:11] Bonjour, voici mon parcours.\n",
                ocr_items=[
                    {
                        "kind": "name",
                        "text": "Bob Durand",
                        "second": 11,
                    }
                ],
            )

            def fake_pipeline(_audio_path, min_speakers=None, max_speakers=None):
                self.assertEqual(min_speakers, 2)
                self.assertEqual(max_speakers, 2)
                return [
                    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
                    {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01"},
                ]

            original_extract_audio = assignment.extract_audio

            def fake_extract_audio(_video_path, audio_dir):
                audio = audio_dir / "video123.wav"
                audio.touch()
                return audio

            assignment.extract_audio = fake_extract_audio
            try:
                target = assignment.assign_ocr_speakers(
                    video,
                    force=True,
                    diarization_pipeline=fake_pipeline,
                    diarization_device="cpu",
                )
            finally:
                assignment.extract_audio = original_extract_audio

            transcript = (
                video.parent
                / "outputs"
                / "transcripts_ocr"
                / "ocr_subtitles_timecoded.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("[00:01] Alice Martin:", transcript)
            self.assertIn("[00:11] Bob Durand:", transcript)
            payload = json.loads(target.read_text(encoding="utf-8"))
            methods = {
                item["validated_name"]: item["method"]
                for item in payload["speaker_mapping"]
            }
            self.assertEqual(methods["Alice Martin"], "spoken_introduction")
            self.assertEqual(methods["Bob Durand"], "ocr_name")


if __name__ == "__main__":
    unittest.main()
