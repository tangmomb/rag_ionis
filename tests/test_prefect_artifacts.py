from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "init"
sys.path.insert(0, str(SCRIPTS_DIR))

from common.prefect_artifacts import artifact_records, progress_for_label  # noqa: E402


class PrefectArtifactTests(unittest.TestCase):
    def test_video_snapshot_contains_progressive_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            (video_dir / "metadata").mkdir(parents=True)
            (video_dir / "outputs" / "images" / "footage").mkdir(parents=True)
            (video_dir / "outputs" / "ocr").mkdir(parents=True)
            (video_dir / "outputs" / "transcripts_whisper").mkdir(parents=True)
            (video_dir / "outputs" / "chunks").mkdir(parents=True)
            (video_dir / "video123.mp4").touch()
            (video_dir / "outputs" / "images" / "footage" / "00_01.jpg").touch()

            (video_dir / "metadata" / "youtube_video_metadata.json").write_text(
                json.dumps(
                    {
                        "youtube_video_id": "video123",
                        "title": "Video de test",
                        "duration_seconds": 42,
                    }
                ),
                encoding="utf-8",
            )
            (video_dir / "metadata" / "pipeline_analysis.json").write_text(
                json.dumps({"has_subtitles": False, "video_type": "interview"}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "ocr" / "reviewed.json").write_text(
                json.dumps({"kinds": {"subtitle": {"00:01": "Bonjour Prefect"}}}),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "transcripts_whisper" / "plain_transcript.txt").write_text(
                "Bonjour depuis le transcript.",
                encoding="utf-8",
            )
            (video_dir / "outputs" / "chunks" / "transcript_chunks.json").write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "content": "Premier chunk",
                                "meta_data": {"speakers": ["Alice"]},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (video_dir / "outputs" / "chunks" / "chunk_01_embedding.json").write_text(
                "{}",
                encoding="utf-8",
            )

            records = artifact_records(
                "Step 23 - Create Chunk Embeddings",
                ["python", "step.py", "--video-dir", str(video_dir)],
                1.25,
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["progress"], 92.0)
            markdown = records[0]["markdown"]
            for expected in (
                "Video de test",
                "## Images",
                "## OCR",
                "Bonjour Prefect",
                "## Transcripts",
                "## Chunks et embeddings",
                "Alice",
            ):
                self.assertIn(expected, markdown)

    def test_progress_is_derived_from_step_number(self) -> None:
        self.assertEqual(progress_for_label("Step 25 - Update SQL Assets"), 100.0)
        self.assertIsNone(progress_for_label("Initial cleanup"))


if __name__ == "__main__":
    unittest.main()
