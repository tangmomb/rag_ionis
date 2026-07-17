import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipeline.catalog import TASKS, TaskSpec
from pipeline.context import PipelineContext
from pipeline.contracts import PlannedTask, TaskResult
from pipeline.executor import execute_tasks
from pipeline.support.paths import existing_transcripts_dir, transcripts_dir


class PipelineTranscriptPathsTests(unittest.TestCase):
    def test_transcripts_dir_does_not_read_process_environment(self) -> None:
        with TemporaryDirectory() as raw_directory:
            video = Path(raw_directory) / "video.mp4"
            with patch.dict(
                os.environ,
                {"PIPELINE_TRANSCRIPTS_DIR_NAME": "transcripts_whisper"},
            ):
                self.assertEqual(
                    transcripts_dir(video),
                    video.parent / "outputs" / "transcripts",
                )

    def test_explicit_branch_never_selects_other_output_branch(self) -> None:
        with TemporaryDirectory() as raw_directory:
            video = Path(raw_directory) / "video.mp4"
            whisper = video.parent / "outputs" / "transcripts_whisper"
            whisper.mkdir(parents=True)

            self.assertEqual(
                existing_transcripts_dir(video, name="transcripts_ocr"),
                video.parent / "outputs" / "transcripts_ocr",
            )
            self.assertEqual(
                existing_transcripts_dir(video),
                whisper,
            )

    def test_explicit_branch_keeps_compatible_legacy_fallback(self) -> None:
        with TemporaryDirectory() as raw_directory:
            video = Path(raw_directory) / "video.mp4"
            legacy_ocr = video.parent / "transcript_ocr"
            legacy_ocr.mkdir()

            self.assertEqual(
                existing_transcripts_dir(video, name="transcripts_ocr"),
                legacy_ocr,
            )

    def test_explicit_branch_ignores_other_legacy_branch(self) -> None:
        with TemporaryDirectory() as raw_directory:
            video = Path(raw_directory) / "video.mp4"
            (video.parent / "transcript_whisper").mkdir()

            self.assertEqual(
                existing_transcripts_dir(video, name="transcripts_ocr"),
                video.parent / "outputs" / "transcripts_ocr",
            )


class PipelineEnvironmentTests(unittest.TestCase):
    def test_executor_does_not_override_process_environment(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            artifact = root / "outputs" / "environment.txt"
            seen_environment = {}

            def handler(context: PipelineContext) -> TaskResult:
                seen_environment.update(
                    {
                        "openai": os.environ.get("PIPELINE_OPENAI_MODE"),
                        "transcripts": os.environ.get(
                            "PIPELINE_TRANSCRIPTS_DIR_NAME"
                        ),
                    }
                )
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text("ok", encoding="utf-8")
                return TaskResult.succeeded(artifacts=(artifact,))

            task_id = "test.environment"
            spec = TaskSpec(
                task_id,
                "processing",
                "Test environment",
                handler,
                postcondition=lambda _context: artifact.exists(),
            )
            context = PipelineContext(
                video_path=video,
                media={"duration_seconds": 1.0},
            )

            with (
                patch.dict(TASKS, {task_id: spec}),
                patch.dict(
                    os.environ,
                    {
                        "PIPELINE_OPENAI_MODE": "external-openai",
                        "PIPELINE_TRANSCRIPTS_DIR_NAME": "external-transcripts",
                    },
                ),
            ):
                context.set_plan([PlannedTask(task_id, "test")])
                execute_tasks(context)
                self.assertEqual(
                    seen_environment,
                    {
                        "openai": "external-openai",
                        "transcripts": "external-transcripts",
                    },
                )
                self.assertEqual(
                    os.environ["PIPELINE_OPENAI_MODE"],
                    "external-openai",
                )
                self.assertEqual(
                    os.environ["PIPELINE_TRANSCRIPTS_DIR_NAME"],
                    "external-transcripts",
                )
                self.assertFalse(context._task_force)


if __name__ == "__main__":
    unittest.main()
