from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.catalog import TASKS, TaskSpec
from pipeline.context import LONG_VIDEO_THRESHOLD_SECONDS, PipelineContext
from pipeline.executor import execute_tasks
from pipeline.manifest import build_manifest
from pipeline.options import PipelineOptions
from pipeline.planner import inspection_plan, processing_plan


class PipelineRoutingTests(unittest.TestCase):
    def make_video(
        self,
        root: Path,
        *,
        has_subtitles: bool | None,
        video_type: str | None,
    ) -> Path:
        video_dir = root / "abcdefghijk"
        metadata_dir = video_dir / "metadata"
        metadata_dir.mkdir(parents=True)
        video = video_dir / "abcdefghijk.mp4"
        video.touch()
        analysis = {}
        if has_subtitles is not None:
            analysis["has_subtitles"] = has_subtitles
        if video_type is not None:
            analysis["video_type"] = video_type
        (metadata_dir / "pipeline_analysis.json").write_text(
            json.dumps(analysis),
            encoding="utf-8",
        )
        return video

    def context(
        self,
        video: Path,
        duration_seconds: float,
    ) -> PipelineContext:
        media = {
            "path": video.resolve().as_posix(),
            "filename": video.name,
            "duration_seconds": duration_seconds,
            "width": 1280,
            "height": 720,
            "fps": 25.0,
            "video_codec": "h264",
            "audio_codec": "aac",
            "has_audio": True,
        }
        with patch("pipeline.context.probe_video", return_value=media):
            return PipelineContext.inspect(video)

    def options(self) -> PipelineOptions:
        return PipelineOptions(
            image_review_model="image-model",
            review_scope="duo",
            speaker_validation_model="speaker-model",
            correction_mode="balanced",
        )

    def test_more_than_ten_minutes_selects_long_ocr_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=True,
                video_type="video_recording",
            )
            context = self.context(video, LONG_VIDEO_THRESHOLD_SECONDS + 1)
            tasks = processing_plan(context)

        self.assertTrue(context.is_long_video)
        self.assertEqual(context.routing()["pipeline_id"], "long.ocr.video_recording")
        task_ids = [task.id for task in tasks]
        self.assertIn("transcript.extract_ocr", task_ids)
        self.assertIn("chunks.summarize_sections", task_ids)
        self.assertIn("chunks.summarize_video", task_ids)

    def test_exactly_ten_minutes_selects_short_whisper_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=False,
                video_type="video_recording",
            )
            context = self.context(video, LONG_VIDEO_THRESHOLD_SECONDS)
            task_ids = [task.id for task in processing_plan(context)]

        self.assertFalse(context.is_long_video)
        self.assertEqual(context.routing()["pipeline_id"], "short.whisper.video_recording")
        self.assertIn("transcript.whisper", task_ids)
        self.assertNotIn("chunks.summarize_sections", task_ids)

    def test_motion_design_is_a_routing_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=True,
                video_type="motion_design",
            )
            context = self.context(video, 90)
            manifest = build_manifest(
                context,
                self.options(),
                processing_plan(context),
            )

        self.assertTrue(manifest["features"]["motion_design"])
        self.assertTrue(manifest["questions"]["is_motion_design"]["answer"])
        self.assertEqual(manifest["routing"]["pipeline_id"], "short.ocr.motion_design")

    def test_pending_characteristics_produce_an_inspection_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=None,
                video_type=None,
            )
            context = self.context(video, 120)
            manifest = build_manifest(
                context,
                self.options(),
                inspection_plan(),
            )

        self.assertEqual(
            manifest["routing"]["status"],
            "needs_content_inspection",
        )
        self.assertIsNone(manifest["features"]["has_subtitles"]["value"])
        self.assertIn("has_subtitles", manifest["routing"]["missing_features"])

    def test_plan_records_python_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=False,
                video_type="interview",
            )
            context = self.context(video, 180)
            manifest = build_manifest(
                context,
                self.options(),
                processing_plan(context),
            )

        tasks = {task["id"]: task for task in manifest["plan"]["tasks"]}
        self.assertEqual(
            tasks["transcript.whisper"]["handler"],
            "pipeline.step_handlers.transcribe_whisper",
        )
        self.assertNotIn("command", tasks["transcript.whisper"])

    def test_executor_passes_one_mutable_context_and_checkpoints_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=False,
                video_type="interview",
            )
            context = self.context(video, 180)
            artifact = video.parent / "outputs" / "test.txt"

            def handler(pipeline_context):
                self.assertIs(pipeline_context, context)
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text("ok", encoding="utf-8")
                pipeline_context.analysis["handler_ran"] = True
                pipeline_context.record_artifacts("test.handler", artifact)

            spec = TaskSpec(
                "test.handler",
                "processing",
                "Test handler",
                handler,
            )
            context.set_plan(
                [
                    {
                        "id": spec.id,
                        "phase": spec.phase,
                        "title": spec.title,
                        "reason": "test",
                        "handler": spec.entrypoint,
                    }
                ]
            )
            with patch.dict(TASKS, {spec.id: spec}):
                execute_tasks(context)

            manifest = json.loads(
                context.manifest_path.read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["execution"]["status"], "completed")
        self.assertEqual(
            manifest["execution"]["tasks"]["test.handler"]["status"],
            "completed",
        )
        self.assertEqual(
            manifest["artifacts"]["by_task"]["test.handler"],
            ["outputs/test.txt"],
        )

    def test_executor_checkpoints_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=False,
                video_type="interview",
            )
            context = self.context(video, 180)

            def failing_handler(pipeline_context):
                raise RuntimeError(f"boom:{pipeline_context.video_id}")

            spec = TaskSpec(
                "test.failure",
                "processing",
                "Test failure",
                failing_handler,
            )
            context.set_plan(
                [
                    {
                        "id": spec.id,
                        "phase": spec.phase,
                        "title": spec.title,
                        "reason": "test",
                        "handler": spec.entrypoint,
                    }
                ]
            )
            with patch.dict(TASKS, {spec.id: spec}):
                with self.assertRaisesRegex(RuntimeError, "boom:abcdefghijk"):
                    execute_tasks(context)

            manifest = json.loads(
                context.manifest_path.read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["execution"]["status"], "failed")
        failure = manifest["execution"]["tasks"]["test.failure"]
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["error"], "boom:abcdefghijk")


if __name__ == "__main__":
    unittest.main()
