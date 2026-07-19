from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from typing import get_type_hints
from unittest.mock import patch

from pipeline import step_handlers
from pipeline.catalog import TASKS, _frames_extracted
from pipeline.context import PipelineContext
from pipeline.contracts import TaskResult, TaskStatus
from pipeline.steps.inspection import (
    classify_frames,
    detect_interviews,
    detect_subtitles,
    infer_video_type,
)


class StepHandlerContractTests(unittest.TestCase):
    def context(self, root: Path) -> PipelineContext:
        video = root / "abcdefghijk.mp4"
        video.touch()
        return PipelineContext(
            video_path=video,
            media={"duration_seconds": 60.0},
        )

    def test_every_catalog_handler_explicitly_returns_task_result(self) -> None:
        self.assertEqual(len(TASKS), 26)
        for task_id, spec in TASKS.items():
            with self.subTest(task_id=task_id):
                self.assertIs(
                    get_type_hints(spec.handler).get("return"),
                    TaskResult,
                )

        self.assertNotIn("SimpleNamespace", inspect.getsource(step_handlers))

    def test_frames_handler_distinguishes_success_cache_and_blockage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))
            target = context.outputs_dir / "images"

            def create_frame(*_args, **_kwargs):
                target.mkdir(parents=True, exist_ok=True)
                (target / "00_00.jpg").write_bytes(b"frame")
                return 1

            with patch(
                "pipeline.steps.inspection.extract_frames.extract_images",
                side_effect=create_frame,
            ):
                created = step_handlers.extract_frames(context)
            self.assertIs(created.status, TaskStatus.SUCCEEDED)
            self.assertTrue(created.reason)
            self.assertEqual(created.artifacts, (target,))

            with patch(
                "pipeline.steps.inspection.extract_frames.extract_images",
                return_value=1,
            ):
                cached = step_handlers.extract_frames(context)
            self.assertIs(cached.status, TaskStatus.CACHED)
            self.assertTrue(cached.reason)

            (target / "00_00.jpg").unlink()
            with patch(
                "pipeline.steps.inspection.extract_frames.extract_images",
                return_value=0,
            ):
                blocked = step_handlers.extract_frames(context)
            self.assertIs(blocked.status, TaskStatus.BLOCKED)
            self.assertTrue(blocked.reason)

    def test_frames_postcondition_requires_a_real_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))
            images = context.outputs_dir / "images"
            images.mkdir(parents=True)
            self.assertFalse(_frames_extracted(context))

            (images / "placeholder.txt").write_text("x", encoding="utf-8")
            self.assertFalse(_frames_extracted(context))

            (images / "frame.png").write_bytes(b"png")
            self.assertTrue(_frames_extracted(context))

    def test_named_inspection_apis_build_typed_options(self) -> None:
        video = Path("video.mp4")
        with patch.object(
            classify_frames,
            "existing_images_dir",
            return_value=Path("images"),
        ), patch.object(
            classify_frames,
            "classify_video_images",
            return_value=Path("manifest.json"),
        ) as classify:
            result = classify_frames.classify_video(
                video,
                model="model.joblib",
                batch_size=8,
                device="cpu",
                force=True,
            )

        self.assertEqual(result, Path("manifest.json"))
        classify_options = classify.call_args.args[1]
        self.assertIsInstance(
            classify_options,
            classify_frames.ClassificationOptions,
        )
        self.assertEqual(classify_options.batch_size, 8)
        self.assertEqual(classify_options.device, "cpu")
        self.assertTrue(classify_options.force)

        with patch.object(
            detect_interviews,
            "detect_for_video",
            return_value=Path("interview.json"),
        ) as detect:
            result = detect_interviews.detect_video(
                video,
                source_dirs=("footage", "mixture"),
                min_run_frames=4,
                force=True,
            )

        self.assertEqual(result, Path("interview.json"))
        interview_options = detect.call_args.args[1]
        self.assertIsInstance(
            interview_options,
            detect_interviews.InterviewDetectionOptions,
        )
        self.assertEqual(
            interview_options.source_dirs,
            ("footage", "mixture"),
        )
        self.assertEqual(interview_options.min_run_frames, 4)
        self.assertTrue(interview_options.force)

    def test_routing_handlers_update_context_without_analysis_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))

            with patch.object(
                infer_video_type,
                "infer_for_video",
                return_value="motion_design",
            ):
                type_result = step_handlers.infer_video_type(context)

            subtitle_details = {
                "has_subtitles": True,
                "reason": "test",
            }
            with patch.object(
                detect_subtitles,
                "detect_for_video",
                return_value=subtitle_details,
            ):
                subtitle_result = step_handlers.detect_subtitles(context)

            self.assertIs(type_result.status, TaskStatus.SUCCEEDED)
            self.assertIs(subtitle_result.status, TaskStatus.SUCCEEDED)
            self.assertEqual(context.video_type, "motion_design")
            self.assertTrue(context.has_subtitles)
            self.assertEqual(
                context.routing_facts.has_subtitles_details,
                subtitle_details,
            )
            self.assertFalse(
                (context.metadata_dir / "pipeline_analysis.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
