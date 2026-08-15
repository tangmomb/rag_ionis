from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.support.runpod_jobs import (
    RunpodClient,
    RunpodConfig,
    RunpodJobError,
    download_s3_prefix,
    upload_s3_directory,
)
from pipeline.support import runpod_execution
from pipeline.workers import runpod_ingestion


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class RunpodClientTests(unittest.TestCase):
    def test_async_job_is_submitted_then_polled_until_completed(self) -> None:
        session = Mock()
        session.post.return_value = FakeResponse({"id": "job-123"})
        session.get.side_effect = [
            FakeResponse({"status": "IN_QUEUE"}),
            FakeResponse({"status": "IN_PROGRESS"}),
            FakeResponse({"status": "COMPLETED", "output": {"ok": True}}),
        ]
        client = RunpodClient(
            RunpodConfig("secret", "endpoint", poll_seconds=0.1, timeout_seconds=10),
            session=session,
            sleep=lambda _seconds: None,
        )

        job_id, output = client.run({"video": {"id": "video-1"}})

        self.assertEqual(job_id, "job-123")
        self.assertEqual(output, {"ok": True})
        self.assertEqual(session.get.call_count, 3)
        self.assertEqual(
            session.post.call_args.kwargs["json"],
            {"input": {"video": {"id": "video-1"}}},
        )

    def test_failed_job_raises_with_remote_status(self) -> None:
        session = Mock()
        session.get.return_value = FakeResponse(
            {"status": "FAILED", "error": "CUDA out of memory"}
        )
        client = RunpodClient(
            RunpodConfig("secret", "endpoint"),
            session=session,
            sleep=lambda _seconds: None,
        )

        with self.assertRaisesRegex(RunpodJobError, "CUDA out of memory"):
            client.wait("job-123")

    def test_s3_prefix_is_downloaded_below_selected_video_directory(self) -> None:
        paginator = Mock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "youtube/20260815_0300/video-1/metadata/info.json"},
                    {"Key": "youtube/20260815_0300/video-1/outputs/result.txt"},
                ]
            }
        ]
        client = Mock()
        client.get_paginator.return_value = paginator
        with tempfile.TemporaryDirectory() as temporary_dir:
            count = download_s3_prefix(
                client,
                "bucket",
                "youtube/20260815_0300/video-1",
                Path(temporary_dir) / "video-1",
            )

        self.assertEqual(count, 2)
        targets = [Path(call.args[2]) for call in client.download_file.call_args_list]
        self.assertTrue(str(targets[0]).endswith("video-1\\metadata\\info.json"))
        self.assertTrue(str(targets[1]).endswith("video-1\\outputs\\result.txt"))

    def test_s3_directory_upload_preserves_relative_paths(self) -> None:
        client = Mock()
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir)
            (source / "metadata").mkdir()
            (source / "metadata" / "info.json").write_text("{}", encoding="utf-8")
            (source / "video.mp4").write_bytes(b"video")
            count = upload_s3_directory(
                client,
                "bucket",
                source,
                "runpod/job/result",
                excluded_suffixes={".mp4"},
            )

        self.assertEqual(count, 1)
        self.assertEqual(
            client.upload_file.call_args.args[2],
            "runpod/job/result/metadata/info.json",
        )

    def test_s3_directory_upload_can_restrict_top_level_content(self) -> None:
        client = Mock()
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir)
            (source / "video.mp4").write_bytes(b"video")
            (source / ".env").write_text("SECRET=value", encoding="utf-8")
            (source / "metadata").mkdir()
            (source / "metadata" / "info.json").write_text("{}", encoding="utf-8")
            count = upload_s3_directory(
                client,
                "bucket",
                source,
                "runpod/job/input",
                included_roots={"video.mp4", "metadata", "outputs"},
            )

        self.assertEqual(count, 2)
        keys = [call.args[2] for call in client.upload_file.call_args_list]
        self.assertNotIn("runpod/job/input/.env", keys)


class RunpodWorkerTests(unittest.TestCase):
    def test_worker_runs_pipeline_and_uploads_one_video_prefix(self) -> None:
        video = {
            "id": "abcdefghijk",
            "snippet": {"title": "Video test"},
            "contentDetails": {"duration": "PT1M"},
            "statistics": {},
        }
        object_store = Mock()
        with (
            patch.dict(
                "os.environ",
                {"S3_BUCKET_NAME": "bucket", "S3_REGION": "eu-west-3"},
            ),
            patch.object(runpod_ingestion, "download_video") as download,
            patch.object(runpod_ingestion, "run_pipeline") as pipeline,
            patch.object(runpod_ingestion, "delete_prefix") as delete,
            patch.object(runpod_ingestion, "upload_s3_directory") as upload,
        ):
            output = runpod_ingestion.process_job(
                {"video": video, "archive_name": "20260815_0300"},
                object_store_client=object_store,
            )

        self.assertEqual(output["prefix"], "youtube/20260815_0300/abcdefghijk")
        download.assert_called_once()
        pipeline.assert_called_once()
        delete.assert_called_once_with(
            object_store,
            "bucket",
            "youtube/20260815_0300/abcdefghijk",
        )
        self.assertEqual(
            upload.call_args.args[3],
            "youtube/20260815_0300/abcdefghijk",
        )
        self.assertIn(".mp4", upload.call_args.kwargs["excluded_suffixes"])

    def test_worker_downloads_cli_source_from_s3_without_youtube(self) -> None:
        object_store = Mock()
        with (
            patch.dict(
                "os.environ",
                {"S3_BUCKET_NAME": "bucket", "S3_REGION": "eu-west-3"},
            ),
            patch.object(runpod_ingestion, "download_s3_prefix") as download_s3,
            patch.object(runpod_ingestion, "download_video") as download_youtube,
            patch.object(runpod_ingestion, "run_pipeline") as pipeline,
            patch.object(runpod_ingestion, "delete_prefix"),
            patch.object(runpod_ingestion, "upload_s3_directory"),
        ):
            output = runpod_ingestion.process_job(
                {
                    "schema_version": 2,
                    "operation": "pipeline_command",
                    "command": "task",
                    "task_id": "transcript.whisper",
                    "video_id": "video-1",
                    "archive_name": "cli-job",
                    "source": {
                        "type": "s3",
                        "bucket": "bucket",
                        "prefix": "runpod/jobs/job/input",
                    },
                    "result": {
                        "bucket": "bucket",
                        "prefix": "runpod/jobs/job/result",
                    },
                },
                object_store_client=object_store,
            )

        self.assertEqual(output["prefix"], "runpod/jobs/job/result")
        download_s3.assert_called_once()
        download_youtube.assert_not_called()
        self.assertEqual(pipeline.call_args.kwargs["command"], "task")
        self.assertEqual(
            pipeline.call_args.kwargs["task_id"],
            "transcript.whisper",
        )


class RunpodExecutionTests(unittest.TestCase):
    def test_gpu_commands_are_delegated_but_dry_runs_stay_local(self) -> None:
        with patch.dict(
            "os.environ",
            {"PIPELINE_EXECUTION_BACKEND": "runpod"},
        ):
            self.assertTrue(runpod_execution.should_delegate_to_runpod("run"))
            self.assertTrue(
                runpod_execution.should_delegate_to_runpod(
                    "task",
                    task_id="transcript.whisper",
                )
            )
            self.assertFalse(
                runpod_execution.should_delegate_to_runpod(
                    "task",
                    task_id="chunks.create",
                )
            )
            self.assertFalse(
                runpod_execution.should_delegate_to_runpod("run", dry_run=True)
            )

    def test_cli_job_uploads_source_and_restores_remote_artifacts(self) -> None:
        from pipeline.options import PipelineOptions

        object_store = Mock()
        remote_client = Mock()
        with tempfile.TemporaryDirectory() as temporary_dir:
            video_dir = Path(temporary_dir) / "video-1"
            video_dir.mkdir()
            video_path = video_dir / "video.mp4"
            video_path.write_bytes(b"video")

            def download_result(_client, _bucket, _prefix, destination):
                (destination / "outputs").mkdir()
                (destination / "outputs" / "result.txt").write_text(
                    "remote",
                    encoding="utf-8",
                )
                (destination / "metadata").mkdir()
                (destination / "metadata" / "video_manifest.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return 2

            remote_client.run.side_effect = lambda payload: (
                "job-1",
                {
                    "video_id": payload["video_id"],
                    "bucket": "bucket",
                    "prefix": payload["result"]["prefix"],
                },
            )
            with (
                patch.dict(
                    "os.environ",
                    {
                        "S3_BUCKET_NAME": "bucket",
                        "S3_REGION": "eu-west-3",
                        "RUNPOD_API_KEY": "secret",
                        "RUNPOD_ENDPOINT_ID": "endpoint",
                    },
                ),
                patch.object(runpod_execution, "s3_client", return_value=object_store),
                patch.object(runpod_execution, "upload_s3_directory", return_value=1),
                patch.object(runpod_execution, "RunpodClient", return_value=remote_client),
                patch.object(
                    runpod_execution,
                    "download_s3_prefix",
                    side_effect=download_result,
                ),
                patch.object(runpod_execution, "delete_prefix") as delete,
            ):
                result = runpod_execution.run_video_on_runpod(
                    video_path,
                    PipelineOptions(),
                    runpod_execution.RemoteCommand(command="run"),
                )

            self.assertEqual(result["job_id"], "job-1")
            self.assertEqual(
                (video_dir / "outputs" / "result.txt").read_text(encoding="utf-8"),
                "remote",
            )
            payload = remote_client.run.call_args.args[0]
            self.assertEqual(payload["command"], "run")
            self.assertEqual(payload["source"]["type"], "s3")
            self.assertEqual(delete.call_count, 2)


if __name__ == "__main__":
    unittest.main()
