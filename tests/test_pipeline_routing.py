from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.catalog import TASKS
from pipeline.context import LONG_VIDEO_THRESHOLD_SECONDS, PipelineContext
from pipeline.contracts import RoutingFacts, VideoType
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
        (metadata_dir / "video_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "routing_facts": analysis,
                }
            ),
            encoding="utf-8",
        )
        return video

    def context(
        self,
        video: Path,
        duration_seconds: float,
    ) -> PipelineContext:
        (video.parent / "metadata" / "youtube_video_metadata.json").write_text(
            json.dumps({"duration_seconds": duration_seconds}),
            encoding="utf-8",
        )
        media = {
            "path": video.resolve().as_posix(),
            "filename": video.name,
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
            context.options = self.options()
            context.set_plan(processing_plan(context))
            manifest = build_manifest(context)

        self.assertEqual(
            manifest["routing_facts"]["video_type"],
            "motion_design",
        )
        self.assertEqual(
            manifest["route"]["pipeline_id"],
            "short.ocr.motion_design",
        )
        self.assertNotIn("questions", manifest)
        self.assertNotIn("features", manifest)
        self.assertNotIn("routing", manifest)

    def test_pending_characteristics_produce_an_inspection_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=None,
                video_type=None,
            )
            context = self.context(video, 120)
            context.options = self.options()
            context.set_plan(inspection_plan())
            manifest = build_manifest(context)

        self.assertEqual(
            manifest["route"]["status"],
            "needs_content_inspection",
        )
        self.assertIsNone(manifest["routing_facts"]["has_subtitles"])
        self.assertIn("has_subtitles", manifest["route"]["missing_facts"])

    def test_plan_records_python_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = self.make_video(
                Path(temporary_directory),
                has_subtitles=False,
                video_type="interview",
            )
            context = self.context(video, 180)
            context.options = self.options()
            context.set_plan(processing_plan(context))
            manifest = build_manifest(context)

        tasks = {task["id"]: task for task in manifest["plan"]["tasks"]}
        self.assertEqual(
            tasks["transcript.whisper"]["handler"],
            "pipeline.step_handlers.transcribe_whisper",
        )
        self.assertNotIn("command", tasks["transcript.whisper"])
        self.assertEqual(tasks["transcript.whisper"]["version"], "1")

    def test_ocr_and_whisper_branches_are_exclusive_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ocr_video = self.make_video(
                root / "ocr",
                has_subtitles=True,
                video_type="interview",
            )
            whisper_video = self.make_video(
                root / "whisper",
                has_subtitles=False,
                video_type="interview",
            )
            ocr_ids = [
                task.id
                for task in processing_plan(self.context(ocr_video, 180))
            ]
            whisper_ids = [
                task.id
                for task in processing_plan(self.context(whisper_video, 180))
            ]

        self.assertIn("transcript.extract_ocr", ocr_ids)
        self.assertNotIn("transcript.whisper", ocr_ids)
        self.assertLess(
            ocr_ids.index("transcript.extract_ocr"),
            ocr_ids.index("chunks.create"),
        )
        self.assertIn("transcript.whisper", whisper_ids)
        self.assertNotIn("transcript.extract_ocr", whisper_ids)
        self.assertLess(
            whisper_ids.index("transcript.whisper"),
            whisper_ids.index("chunks.create"),
        )

    def test_every_planned_task_is_registered(self) -> None:
        self.assertTrue(
            all(task.id in TASKS for task in inspection_plan())
        )

    def test_routing_facts_validate_their_domain(self) -> None:
        facts = RoutingFacts.from_dict(
            {
                "has_subtitles": False,
                "video_type": "interview",
                "has_subtitles_details": {"score": 0.12},
            }
        )
        self.assertIs(facts.video_type, VideoType.INTERVIEW)
        self.assertFalse(facts.has_subtitles)

        with self.assertRaisesRegex(ValueError, "has_subtitles"):
            RoutingFacts.from_dict({"has_subtitles": "false"})
        with self.assertRaisesRegex(ValueError, "video_type invalide"):
            RoutingFacts.from_dict({"video_type": "motion_desgin"})
        with self.assertRaisesRegex(ValueError, "has_subtitles_details"):
            RoutingFacts.from_dict({"has_subtitles_details": []})


if __name__ == "__main__":
    unittest.main()
