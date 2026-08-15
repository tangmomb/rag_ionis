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
)
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
            patch.object(runpod_ingestion, "upload_directory") as upload,
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
        self.assertEqual(upload.call_args.kwargs["video_ids"], ["abcdefghijk"])
        self.assertTrue(upload.call_args.kwargs["force"])


if __name__ == "__main__":
    unittest.main()
