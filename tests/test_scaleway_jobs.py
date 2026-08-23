from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.support import scaleway_execution
from pipeline.support.scaleway_jobs import (
    ScalewayClient,
    ScalewayConfig,
    download_s3_prefix,
    upload_s3_directory,
)
from pipeline.workers import scaleway_ingestion


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200, headers=None):
        self.payload = payload or {}
        self.text = json.dumps(self.payload)
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class ScalewayClientTests(unittest.TestCase):
    def test_api_error_never_echoes_response_body(self) -> None:
        secret = "must-never-appear"
        response = FakeResponse(status_code=400, headers={"x-request-id": "request-1"})
        response.text = json.dumps({"message": f"invalid request {secret}"})
        response.raise_for_status = Mock(
            side_effect=__import__("requests").HTTPError("bad request")
        )
        session = Mock()
        session.request.return_value = response
        config = ScalewayConfig(
            "secret",
            "project",
            "rg.fr-par.scw.cloud/rag-ionis/worker:latest",
        )
        client = ScalewayClient(config, session=session)

        with self.assertRaisesRegex(
            Exception, r"API Scaleway: HTTP 400, requete request-1\."
        ) as raised:
            client.server_status("server-123")

        self.assertNotIn(secret, str(raised.exception))

    def test_completed_job_starts_and_stops_dedicated_instance(self) -> None:
        config = ScalewayConfig(
            "secret",
            "project",
            "rg.fr-par.scw.cloud/rag-ionis/worker:latest",
            server_id="server-123",
        )
        client = ScalewayClient(config)
        client.ensure_started = Mock()
        client.stop_server = Mock()
        object_store = Mock()
        object_store.get_object.return_value = {
            "Body": io.BytesIO(
                json.dumps(
                    {"status": "completed", "output": {"video_id": "video-1"}}
                ).encode("utf-8")
            )
        }
        payload = {
            "control": {"bucket": "bucket", "status_key": "scaleway/jobs/1/status.json"}
        }

        server_id, output = client.run(payload, object_store=object_store, bucket="bucket")

        self.assertEqual(server_id, "1")
        self.assertEqual(output["video_id"], "video-1")
        client.ensure_started.assert_called_once_with("server-123")
        client.stop_server.assert_called_once_with("server-123")
        self.assertTrue(object_store.put_object.called)

    def test_cleanup_error_does_not_mask_job_error(self) -> None:
        config = ScalewayConfig(
            "secret",
            "project",
            "rg.fr-par.scw.cloud/rag-ionis/worker:latest",
            server_id="server-123",
        )
        client = ScalewayClient(config)
        client.ensure_started = Mock(side_effect=RuntimeError("original failure"))
        client.stop_server = Mock(side_effect=RuntimeError("cleanup failure"))
        object_store = Mock()

        with self.assertRaisesRegex(RuntimeError, "original failure"):
            client.run(
                {"control": {"status_key": "scaleway/jobs/1/status.json"}},
                object_store=object_store,
                bucket="bucket",
            )

    def test_s3_prefix_is_downloaded_below_selected_video_directory(self) -> None:
        paginator = Mock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "youtube/archive/video-1/metadata/info.json"},
                    {"Key": "youtube/archive/video-1/outputs/result.txt"},
                ]
            }
        ]
        client = Mock()
        client.get_paginator.return_value = paginator
        with tempfile.TemporaryDirectory() as temporary_dir:
            count = download_s3_prefix(
                client,
                "bucket",
                "youtube/archive/video-1",
                Path(temporary_dir) / "video-1",
            )

        self.assertEqual(count, 2)
        targets = [Path(call.args[2]) for call in client.download_file.call_args_list]
        self.assertTrue(str(targets[0]).endswith("video-1\\metadata\\info.json"))
        self.assertTrue(str(targets[1]).endswith("video-1\\outputs\\result.txt"))

    def test_s3_upload_excludes_secrets_and_video_when_requested(self) -> None:
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
                "scaleway/job/input",
                included_roots={"video.mp4", "metadata", "outputs"},
            )

        self.assertEqual(count, 2)
        keys = [call.args[2] for call in client.upload_file.call_args_list]
        self.assertNotIn("scaleway/job/input/.env", keys)


class ScalewayWorkerTests(unittest.TestCase):
    def test_healthcheck_operation_skips_video_processing(self) -> None:
        expected = {"operation": "healthcheck", "status": "completed"}
        with (
            patch.object(scaleway_ingestion, "validate_worker_gpu") as validate,
            patch.object(
                scaleway_ingestion,
                "run_gpu_healthcheck",
                return_value=expected,
            ) as healthcheck,
            patch.object(scaleway_ingestion, "download_video") as download_video,
        ):
            output = scaleway_ingestion.process_job({"operation": "healthcheck"})

        self.assertEqual(output, expected)
        validate.assert_called_once_with()
        healthcheck.assert_called_once_with()
        download_video.assert_not_called()

    def test_worker_runs_youtube_pipeline_and_uploads_artifacts(self) -> None:
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
            patch.object(scaleway_ingestion, "download_video") as download,
            patch.object(scaleway_ingestion, "run_pipeline") as pipeline,
            patch.object(scaleway_ingestion, "delete_prefix") as delete,
            patch.object(scaleway_ingestion, "upload_s3_directory") as upload,
        ):
            output = scaleway_ingestion.process_job(
                {"video": video, "archive_name": "20260815_0300"},
                object_store_client=object_store,
            )

        self.assertEqual(output["prefix"], "youtube/20260815_0300/abcdefghijk")
        download.assert_called_once()
        pipeline.assert_called_once()
        delete.assert_called_once()
        self.assertIn(".mp4", upload.call_args.kwargs["excluded_suffixes"])

    def test_worker_downloads_cli_source_without_youtube(self) -> None:
        object_store = Mock()
        with (
            patch.dict("os.environ", {"S3_BUCKET_NAME": "bucket"}),
            patch.object(scaleway_ingestion, "download_s3_prefix") as download_s3,
            patch.object(scaleway_ingestion, "download_video") as download_youtube,
            patch.object(scaleway_ingestion, "run_pipeline") as pipeline,
            patch.object(scaleway_ingestion, "delete_prefix"),
            patch.object(scaleway_ingestion, "upload_s3_directory"),
        ):
            output = scaleway_ingestion.process_job(
                {
                    "command": "task",
                    "task_id": "transcript.whisper",
                    "video_id": "video-1",
                    "archive_name": "cli-job",
                    "source": {
                        "type": "s3",
                        "bucket": "bucket",
                        "prefix": "scaleway/jobs/job/input",
                    },
                    "result": {
                        "bucket": "bucket",
                        "prefix": "scaleway/jobs/job/result",
                    },
                },
                object_store_client=object_store,
            )

        self.assertEqual(output["prefix"], "scaleway/jobs/job/result")
        download_s3.assert_called_once()
        download_youtube.assert_not_called()
        self.assertEqual(pipeline.call_args.kwargs["task_id"], "transcript.whisper")


class ScalewayExecutionTests(unittest.TestCase):
    def test_gpu_commands_are_delegated_but_cpu_and_dry_runs_stay_local(self) -> None:
        with patch.dict("os.environ", {"PIPELINE_EXECUTION_BACKEND": "scaleway"}):
            self.assertTrue(scaleway_execution.should_delegate_to_scaleway("run"))
            self.assertTrue(
                scaleway_execution.should_delegate_to_scaleway(
                    "task", task_id="transcript.whisper"
                )
            )
            self.assertFalse(
                scaleway_execution.should_delegate_to_scaleway(
                    "task", task_id="chunks.create"
                )
            )
            self.assertFalse(
                scaleway_execution.should_delegate_to_scaleway("run", dry_run=True)
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
                    "remote", encoding="utf-8"
                )
                (destination / "metadata").mkdir()
                (destination / "metadata" / "video_manifest.json").write_text(
                    "{}", encoding="utf-8"
                )
                return 2

            remote_client.run.side_effect = lambda payload, **_kwargs: (
                "server-1",
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
                        "SCW_SECRET_KEY": "secret",
                        "SCW_DEFAULT_PROJECT_ID": "project",
                        "SCALEWAY_SERVER_ID": "server-1",
                        "SCALEWAY_CONTAINER_IMAGE": (
                            "rg.fr-par.scw.cloud/rag-ionis/worker:latest"
                        ),
                    },
                ),
                patch.object(scaleway_execution, "s3_client", return_value=object_store),
                patch.object(scaleway_execution, "upload_s3_directory", return_value=1),
                patch.object(scaleway_execution, "ScalewayClient", return_value=remote_client),
                patch.object(
                    scaleway_execution, "download_s3_prefix", side_effect=download_result
                ),
                patch.object(scaleway_execution, "delete_prefix") as delete,
            ):
                result = scaleway_execution.run_video_on_scaleway(
                    video_path,
                    PipelineOptions(),
                    scaleway_execution.RemoteCommand(command="run"),
                )

            self.assertEqual(result["job_id"], "server-1")
            self.assertEqual(
                (video_dir / "outputs" / "result.txt").read_text(encoding="utf-8"),
                "remote",
            )
            payload = remote_client.run.call_args.args[0]
            self.assertEqual(payload["source"]["type"], "s3")
            self.assertIn("status_key", payload["control"])
            delete.assert_called_once()

    def test_single_worker_processes_every_selected_video_sequentially(self) -> None:
        from pipeline.options import PipelineOptions

        videos = [Path("video-1.mp4"), Path("video-2.mp4")]
        remote_command = scaleway_execution.RemoteCommand(command="run")
        with (
            patch.dict("os.environ", {"SCALEWAY_MAX_CONCURRENT_JOBS": "1"}),
            patch.object(
                scaleway_execution,
                "run_video_on_scaleway",
                side_effect=[{"video": "video-1"}, {"video": "video-2"}],
            ) as run_video,
        ):
            results = scaleway_execution.run_videos_on_scaleway(
                videos,
                PipelineOptions(),
                remote_command,
            )

        self.assertEqual(
            results,
            [{"video": "video-1"}, {"video": "video-2"}],
        )
        self.assertEqual(
            [call.args[0] for call in run_video.call_args_list],
            videos,
        )


if __name__ == "__main__":
    unittest.main()
