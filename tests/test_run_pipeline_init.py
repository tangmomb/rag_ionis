from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
INIT_DIR = ROOT_DIR / "scripts" / "init"
if str(INIT_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_DIR))

from common.pipeline_paths import consolidate_init_dir


def load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, INIT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


run_pipeline = load_script("run_pipeline_init_for_tests", "RUN_PIPELINE_INIT.py")
get_data = load_script("get_data_for_tests", "01_get_data.py")
download_videos = load_script("download_videos_for_tests", "02_download_videos.py")
upload_s3 = load_script("upload_s3_for_tests", "final_01_upload_outputs_to_s3.py")
update_sql = load_script("update_sql_for_tests", "final_02_update_sql_assets.py")


class RunPipelineInitTests(unittest.TestCase):
    def test_youtube_metadata_keeps_the_complete_api_video(self) -> None:
        video = {
            "kind": "youtube#video",
            "etag": "etag-test",
            "id": "abcdefghijk",
            "snippet": {
                "publishedAt": "2026-07-14T12:00:00Z",
                "title": "Video test",
                "description": "Description",
                "thumbnails": {
                    "default": {"url": "default.jpg", "width": 120, "height": 90},
                    "medium": {"url": "medium.jpg", "width": 320, "height": 180},
                    "high": {"url": "high.jpg", "width": 480, "height": 360},
                    "maxres": {"url": "maxres.jpg", "width": 1280, "height": 720},
                },
                "tags": ["energie", "formation"],
            },
            "contentDetails": {"duration": "PT1M31S", "definition": "hd"},
            "statistics": {"viewCount": "371", "likeCount": "1"},
            "extraApiField": {"preserved": True},
        }

        payload = get_data.video_info_payload(video)

        self.assertEqual(payload["snippet"]["thumbnails"], video["snippet"]["thumbnails"])
        self.assertEqual(payload["contentDetails"], video["contentDetails"])
        self.assertEqual(payload["extraApiField"], {"preserved": True})
        self.assertEqual(payload["thumbnail_medium_url"], "medium.jpg")
        self.assertEqual(payload["duration_seconds"], 91)

    def test_yes_no_answers_are_normalized(self) -> None:
        self.assertTrue(run_pipeline.normalize_yes_no("oui"))
        self.assertFalse(run_pipeline.normalize_yes_no("non"))
        self.assertFalse(run_pipeline.normalize_yes_no("", default=False))
        with self.assertRaises(ValueError):
            run_pipeline.normalize_yes_no("peut-etre")

    def test_pipeline_stages_can_be_selected_non_interactively(self) -> None:
        stages = run_pipeline.parse_pipeline_stages("process,s3,sql")

        self.assertFalse(stages.download)
        self.assertTrue(stages.process)
        self.assertTrue(stages.upload_s3)
        self.assertTrue(stages.update_sql)
        self.assertEqual(
            run_pipeline.parse_pipeline_stages("all"),
            run_pipeline.PipelineStages(True, True, True, True),
        )
        with self.assertRaises(run_pipeline.argparse.ArgumentTypeError):
            run_pipeline.parse_pipeline_stages("inconnue")

    def test_one_pipeline_stage_is_selected_from_numeric_menu(self) -> None:
        with patch("builtins.input", side_effect=["invalide", "3"]):
            stages = run_pipeline.ask_pipeline_stages()

        self.assertEqual(
            stages,
            run_pipeline.PipelineStages(upload_s3=True),
        )

    def test_existing_local_videos_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            video_dir = download_root / "init" / "abcdefghijk"
            video_dir.mkdir(parents=True)
            video = video_dir / "abcdefghijk.mp4"
            video.write_bytes(b"video")

            self.assertEqual(run_pipeline.existing_local_videos(download_root), [video])

    def test_redownload_cleanup_preserves_metadata_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            info_dir = download_root / "info_videos"
            info_dir.mkdir()
            metadata = info_dir / "abcdefghijk.youtube_api_infos.json"
            metadata.write_text("{}", encoding="utf-8")
            old_run = download_root / "init"
            old_run.mkdir()
            (old_run / "video.mp4").write_bytes(b"video")

            with patch.object(run_pipeline, "ROOT_DIR", download_root.parent):
                run_pipeline.clean_download_root(download_root)

            self.assertTrue(metadata.exists())
            self.assertFalse(old_run.exists())

    def test_redownload_cleanup_refuses_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            workspace_root = Path(temporary_dir)
            with patch.object(run_pipeline, "ROOT_DIR", workspace_root):
                with self.assertRaises(RuntimeError):
                    run_pipeline.clean_download_root(workspace_root)

    def test_existing_video_is_reused_without_youtube_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            video_id = "abcdefghijk"
            video_dir = download_root / "init" / video_id
            video_dir.mkdir(parents=True)
            (video_dir / f"{video_id}.mp4").write_bytes(b"video")
            outputs = video_dir / "outputs"
            outputs.mkdir()
            (outputs / "result.txt").write_text("done", encoding="utf-8")
            video = (
                video_id,
                "Test",
                f"https://www.youtube.com/watch?v={video_id}",
                {"youtube_video_id": video_id},
            )

            with patch.object(download_videos.yt_dlp, "YoutubeDL") as youtube_dl:
                reused = download_videos.download_video(
                    video,
                    download_root / "init",
                    download_root,
                    force=False,
                )

            youtube_dl.assert_not_called()
            self.assertEqual(reused.read_bytes(), b"video")
            self.assertEqual(
                (video_dir / "outputs" / "result.txt").read_text(encoding="utf-8"),
                "done",
            )

    def test_legacy_dated_init_directories_are_merged_into_single_init(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            old_run = download_root / "20260714_1200_init" / "abcdefghijk"
            recent_run = download_root / "20260714_1300_init" / "lmnopqrstuv"
            old_run.mkdir(parents=True)
            recent_run.mkdir(parents=True)
            (old_run / "abcdefghijk.mp4").write_bytes(b"old")
            (recent_run / "lmnopqrstuv.mp4").write_bytes(b"recent")

            pipeline_dir = consolidate_init_dir(download_root)

            self.assertEqual(pipeline_dir, download_root / "init")
            self.assertTrue((pipeline_dir / "abcdefghijk" / "abcdefghijk.mp4").exists())
            self.assertTrue((pipeline_dir / "lmnopqrstuv" / "lmnopqrstuv.mp4").exists())
            self.assertFalse((download_root / "20260714_1200_init").exists())
            self.assertFalse((download_root / "20260714_1300_init").exists())

    def test_publication_is_limited_to_selected_video_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            first = run_dir / "abcdefghijk"
            second = run_dir / "lmnopqrstuv"
            first.mkdir()
            second.mkdir()
            (first / "first.txt").write_text("first", encoding="utf-8")
            (second / "second.txt").write_text("second", encoding="utf-8")

            result = upload_s3.upload_directory(
                client=None,
                bucket="test",
                root_dir=run_dir,
                dry_run=True,
                video_ids=[first.name],
            )
            sql_dirs = update_sql.candidate_video_dirs(run_dir, video_ids=[first.name])

            self.assertEqual(result, {"uploaded": 1, "skipped": 0})
            self.assertEqual(sql_dirs, [first])

    def test_sql_reset_recreates_the_complete_data_schema(self) -> None:
        class RecordingCursor:
            def __init__(self) -> None:
                self.queries = []

            def execute(self, query, params=None) -> None:
                self.queries.append((query, params))

        cursor = RecordingCursor()

        update_sql.reset_data_schema(cursor)

        statements = [query for query, _params in cursor.queries]
        self.assertIn("public.videos", statements[0])
        self.assertEqual(statements[1], "DROP SCHEMA IF EXISTS data CASCADE")
        self.assertEqual(statements[2], "CREATE SCHEMA data")
        self.assertIn("SET search_path TO data, public", statements[-1])
        self.assertIn("CREATE TABLE IF NOT EXISTS videos", statements[-1])
        self.assertNotIn("video_summary", statements[-1])
        self.assertNotIn("DROP SCHEMA IF EXISTS chat", "\n".join(statements))

    def test_stage_options_are_forwarded_to_publication_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            video_id = "abcdefghijk"
            video_dir = download_root / "init" / video_id
            video_dir.mkdir(parents=True)
            (video_dir / f"{video_id}.mp4").write_bytes(b"video")
            commands = []

            def capture_step(_index, _total, label, command, _env):
                commands.append((label, command))

            argv = [
                "run_pipeline_init.py",
                "1",
                "--stages",
                "s3,sql",
                "--download-dir",
                str(download_root),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                run_pipeline,
                "run_step_numbered",
                side_effect=capture_step,
            ):
                run_pipeline.main()

            self.assertEqual(len(commands), 2)
            s3_command = commands[0][1]
            sql_command = commands[1][1]
            self.assertIn("--force", s3_command)
            self.assertIn("--clean-init-prefix", s3_command)
            self.assertIn("--video-id", s3_command)
            self.assertIn("--reset-database", sql_command)
            self.assertIn(str(download_root / "init"), s3_command)
            self.assertIn(str(download_root / "init"), sql_command)
            self.assertNotIn("--dry-run", s3_command)
            self.assertNotIn("--dry-run", sql_command)
            self.assertNotIn("--skip-transcripts", sql_command)

    def test_image_review_model_is_forwarded_to_step_13(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            video_id = "abcdefghijk"
            video_dir = download_root / "init" / video_id
            video_dir.mkdir(parents=True)
            (video_dir / f"{video_id}.mp4").write_bytes(b"video")
            calls = []

            def capture_video_script(*args, **kwargs):
                calls.append((args, kwargs))

            argv = [
                "run_pipeline_init.py",
                "1",
                "--stages",
                "process",
                "--download-dir",
                str(download_root),
                "--image-review-model",
                "gpt-image-test",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                run_pipeline,
                "run_video_script",
                side_effect=capture_video_script,
            ), patch.object(
                run_pipeline,
                "print_post_step09_routing",
            ), patch.object(
                run_pipeline,
                "selected_branch_after_step09",
                return_value="has_sub",
            ):
                run_pipeline.main()

            step13_calls = [call for call in calls if call[0][0] == 13]
            self.assertEqual(len(step13_calls), 1)
            self.assertEqual(
                step13_calls[0][1]["extra_args"],
                ["--model", "gpt-image-test", "--limit-images", "1"],
            )

    def test_whisper_pipeline_runs_speakers_before_combined_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            video_id = "abcdefghijk"
            video_dir = download_root / "init" / video_id
            video_dir.mkdir(parents=True)
            (video_dir / f"{video_id}.mp4").write_bytes(b"video")
            calls = []

            def capture_video_script(*args, **kwargs):
                calls.append((args, kwargs))

            argv = [
                "run_pipeline_init.py",
                "1",
                "--stages",
                "process",
                "--download-dir",
                str(download_root),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                run_pipeline,
                "run_video_script",
                side_effect=capture_video_script,
            ), patch.object(
                run_pipeline,
                "print_post_step09_routing",
            ), patch.object(
                run_pipeline,
                "selected_branch_after_step09",
                return_value="no_sub",
            ):
                run_pipeline.main()

            scripts = [Path(call[0][3]).name for call in calls]
            expected = [
                "16_WHISPER_transcribe_with_whisper.py",
                "17A_WHISPER_propose_speakers.py",
                "17B_WHISPER_validate_speakers.py",
                "18_WHISPER_correct_with_ocr_and_speakers.py",
                "19_WHISPER_enrich_transcripts.py",
                "20_WHISPER_create_plain_transcript.py",
                "21_CHUNK_create_transcript_chunks.py",
                "22_CHUNK_create_chunk_embeddings.py",
            ]
            positions = [scripts.index(script) for script in expected]
            self.assertEqual(positions, sorted(positions))

    def test_ocr_pipeline_runs_speakers_between_steps_17_and_18(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            video_id = "abcdefghijk"
            video_dir = download_root / "init" / video_id
            video_dir.mkdir(parents=True)
            (video_dir / f"{video_id}.mp4").write_bytes(b"video")
            calls = []

            def capture_video_script(*args, **kwargs):
                calls.append((args, kwargs))

            argv = [
                "run_pipeline_init.py",
                "1",
                "--stages",
                "process",
                "--download-dir",
                str(download_root),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                run_pipeline,
                "run_video_script",
                side_effect=capture_video_script,
            ), patch.object(
                run_pipeline,
                "print_post_step09_routing",
            ), patch.object(
                run_pipeline,
                "selected_branch_after_step09",
                return_value="has_sub",
            ):
                run_pipeline.main()

            scripts = [Path(call[0][3]).name for call in calls]
            expected = [
                "17_OCR_normalize_ionis_stm.py",
                "17A_OCR_propose_speakers.py",
                "17B_OCR_validate_speakers.py",
                "17C_OCR_correct_speakers.py",
                "18_OCR_create_plain_transcript.py",
                "19_OCR_enrich_transcripts.py",
            ]
            positions = [scripts.index(script) for script in expected]
            self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
