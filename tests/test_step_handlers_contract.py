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
from pipeline.contracts import RoutingFacts, TaskResult, TaskStatus, VideoType
from pipeline.options import PipelineOptions
from pipeline.steps.inspection import (
    classify_frames,
    detect_interviews,
    detect_subtitles,
    infer_video_type,
)
from pipeline.steps.ocr import build_processed_ocr
from pipeline.steps.transcripts import extract_ocr_subtitles


class StepHandlerContractTests(unittest.TestCase):
    def context(self, root: Path) -> PipelineContext:
        video = root / "abcdefghijk.mp4"
        video.touch()
        return PipelineContext(
            video_path=video,
            media={"duration_seconds": 60.0},
        )

    def test_every_catalog_handler_explicitly_returns_task_result(self) -> None:
        self.assertEqual(len(TASKS), 27)
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

    def test_long_video_whisper_skips_diarization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))
            context.media["duration_seconds"] = 601.0
            context.routing_facts = RoutingFacts(
                video_type=VideoType.LONG_VIDEO,
            )

            def create_transcript(
                _whisperx,
                _model,
                _video,
                transcript_directory,
                _audio_directory,
                _device,
                **_kwargs,
            ):
                target = transcript_directory / "transcript_1_brut.txt"
                target.write_text("[00:00-00:01] Bonjour.\n", encoding="utf-8")
                return target

            with (
                patch(
                    "pipeline.steps.transcripts.transcribe_with_whisper.load_whisperx_model",
                    return_value=(object(), object(), "cuda"),
                ),
                patch(
                    "pipeline.steps.transcripts.transcribe_with_whisper.load_diarization_pipeline",
                ) as load_diarization,
                patch(
                    "pipeline.steps.transcripts.transcribe_with_whisper.transcribe_video",
                    side_effect=create_transcript,
                ) as transcribe,
            ):
                result = step_handlers.transcribe_whisper(context)

        self.assertIs(result.status, TaskStatus.SUCCEEDED)
        load_diarization.assert_not_called()
        self.assertIsNone(
            transcribe.call_args.kwargs["diarization_pipeline"]
        )

    def test_section_summaries_use_configured_openai_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))
            context.options = PipelineOptions(openai_mode="batch")
            target = context.outputs_dir / "chunks" / "transcript_chunks.json"
            payload = {
                "chunks": [
                    {
                        "chunk_index": 1,
                        "chunk_level": "section",
                        "content": "Resume",
                    }
                ]
            }
            def write_summary(*_args, **_kwargs):
                target.parent.mkdir(parents=True)
                target.write_text("{}", encoding="utf-8")
                return target

            with (
                patch(
                    "pipeline.steps.chunks.hierarchical_chunks.load_chunks",
                    return_value=(payload, target),
                ),
                patch(
                    "pipeline.steps.chunks.hierarchical_chunks.summarize_sections",
                    side_effect=write_summary,
                ) as summarize,
            ):
                result = step_handlers.summarize_sections(context)

        self.assertIs(result.status, TaskStatus.SUCCEEDED)
        self.assertEqual(summarize.call_args.kwargs["mode"], "batch")
        self.assertFalse(summarize.call_args.kwargs["reset_batch"])

    def test_ocr_review_is_skipped_when_scope_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))
            context.options = PipelineOptions(review_scope="none")

            with patch(
                "pipeline.steps.ocr.review_other_text_candidates.review_video",
            ) as review:
                result = step_handlers.review_other_text(context)

        self.assertIs(result.status, TaskStatus.SKIPPED)
        review.assert_not_called()

    def test_remaining_openai_steps_use_configured_batch_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))
            context.options = PipelineOptions(openai_mode="batch")

            transcript_dir = context.outputs_dir / "transcripts_whisper"
            transcript_dir.mkdir(parents=True)
            source = transcript_dir / "transcript_1_brut.txt"
            target = transcript_dir / "transcript_2_corrected.txt"
            report = transcript_dir / "transcript_2_corrections.tsv"
            source.write_text("[00:00-00:01] Bonjour.\n", encoding="utf-8")

            def reconcile(*_args, **_kwargs):
                target.write_text("Bonjour.\n", encoding="utf-8")
                report.write_text("", encoding="utf-8")
                return target

            with patch(
                "pipeline.steps.transcripts.reconcile_whisper_with_ocr.reconcile_file",
                side_effect=reconcile,
            ) as reconcile_mock:
                result = step_handlers.reconcile_whisper_with_ocr(context)

            self.assertIs(result.status, TaskStatus.SUCCEEDED)
            self.assertEqual(reconcile_mock.call_args.kwargs["mode"], "batch")
            self.assertFalse(
                reconcile_mock.call_args.kwargs["reset_batch"]
            )

            chunks_target = (
                context.outputs_dir / "chunks" / "transcript_chunks.json"
            )
            chunks_target.parent.mkdir(parents=True)
            chunks_target.write_text("{}", encoding="utf-8")
            global_payload = {
                "chunks": [
                    {
                        "chunk_index": 1,
                        "chunk_level": "global",
                        "content": "Resume global",
                    }
                ]
            }
            with (
                patch(
                    "pipeline.steps.chunks.hierarchical_chunks.load_chunks",
                    return_value=(global_payload, chunks_target),
                ),
                patch(
                    "pipeline.steps.chunks.hierarchical_chunks.summarize_video",
                    return_value=chunks_target,
                ) as summarize_mock,
            ):
                result = step_handlers.summarize_video(context)

            self.assertIs(result.status, TaskStatus.CACHED)
            self.assertEqual(summarize_mock.call_args.kwargs["mode"], "batch")
            self.assertFalse(
                summarize_mock.call_args.kwargs["reset_batch"]
            )

            embedding_target = (
                chunks_target.parent / "chunk_01_embedding.json"
            )

            def embed(*_args, **_kwargs):
                embedding_target.write_text("{}", encoding="utf-8")
                return True

            with patch(
                "pipeline.steps.embeddings.create_chunk_embeddings.create_embeddings_batch",
                side_effect=embed,
            ) as embed_mock:
                result = step_handlers.create_embeddings(context)

            self.assertIs(result.status, TaskStatus.SUCCEEDED)
            self.assertTrue(embed_mock.call_args.kwargs["wait"])
            self.assertFalse(embed_mock.call_args.kwargs["reset_batch"])

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

    def test_processed_ocr_keeps_subtitles_for_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))
            with patch.object(
                build_processed_ocr,
                "process_video",
                return_value=None,
            ) as process:
                step_handlers.build_processed_ocr(context)

        self.assertFalse(process.call_args.kwargs["strip_subtitles"])

    def test_missing_ocr_correction_reference_is_informational(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.context(Path(temporary_directory))
            with patch.object(
                extract_ocr_subtitles,
                "extract_for_video",
                return_value=None,
            ):
                result = step_handlers.extract_ocr_transcript(context)

        self.assertIs(result.status, TaskStatus.SKIPPED)


if __name__ == "__main__":
    unittest.main()
