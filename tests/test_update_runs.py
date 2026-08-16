from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline import update_runs
from pipeline.ingest import fetch_youtube_metadata
from pipeline.publish import sync_database


class YoutubeDailySyncTests(unittest.TestCase):
    def test_archive_directory_uses_start_minute_and_never_overwrites(self) -> None:
        started_at = datetime.now().astimezone().replace(
            year=2026,
            month=8,
            day=2,
            hour=14,
            minute=37,
            second=49,
            microsecond=0,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_dir = Path(temporary_dir)

            first = update_runs.create_archive_directory(
                download_dir,
                started_at,
            )
            second = update_runs.create_archive_directory(
                download_dir,
                started_at,
            )

            self.assertEqual(first.name, "20260802_1437")
            self.assertEqual(second.name, "20260802_1437_02")

    def test_daily_json_is_only_saved_in_the_dated_video_directory(self) -> None:
        video = {
            "id": "abcdefghijk",
            "snippet": {
                "title": "Video test",
                "description": "Description",
                "publishedAt": "2026-08-01T10:00:00Z",
                "thumbnails": {},
            },
            "contentDetails": {"duration": "PT42S"},
            "statistics": {"viewCount": "12", "commentCount": "1"},
        }
        comments = {
            "youtube_video_id": "abcdefghijk",
            "comments": [{"youtube_comment_id": "comment-1", "text": "Bonjour"}],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_dir = Path(temporary_dir)
            archive_dir = download_dir / "20260802_1437"
            archive_dir.mkdir()

            update_runs.write_archive(
                video,
                comments,
                archive_dir,
            )

            video_dir = archive_dir / "abcdefghijk" / "metadata"
            archived_video = json.loads(
                (video_dir / "youtube_video_metadata.json").read_text(encoding="utf-8")
            )
            archived_comments = json.loads(
                (video_dir / "youtube_comments.json").read_text(encoding="utf-8")
            )
            self.assertEqual(archived_video["statistics"]["viewCount"], "12")
            self.assertEqual(archived_comments, comments)
            self.assertFalse(
                (download_dir / "_00_info_videos").exists(),
            )
            self.assertFalse(
                (download_dir / "_00_info_comments").exists(),
            )
            self.assertFalse((download_dir / "init").exists())

    def test_archive_log_records_detection_pipeline_and_video_statistics(self) -> None:
        started_at = datetime(2026, 8, 13, 3, 0, tzinfo=timezone.utc)
        videos = [
            {
                "id": "new-video-1",
                "snippet": {"title": "Nouvelle vidéo"},
                "statistics": {
                    "viewCount": "123",
                    "likeCount": "17",
                    "commentCount": "4",
                },
            },
            {
                "id": "existing-video",
                "snippet": {"title": "Vidéo existante"},
                "statistics": {
                    "viewCount": "456",
                    "likeCount": "31",
                    "commentCount": "8",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive_dir = Path(temporary_dir) / "20260813_0300"
            archive_dir.mkdir()

            log_path = update_runs.write_archive_log(
                archive_dir,
                started_at=started_at,
                finished_at=started_at,
                status="completed",
                videos=videos,
                new_video_ids=["new-video-1"],
                pipeline_results={
                    "new-video-1": {
                        "status": "completed",
                        "succeeded": True,
                        "error": None,
                    }
                },
                comparison_directory=Path("downloads/youtube/20260812_0300"),
                errors=[],
            )

            payload = json.loads(log_path.read_text(encoding="utf-8"))

        self.assertEqual(log_path.name, "daily_sync_log.json")
        self.assertEqual(payload["snapshot_date"], "2026-08-13")
        self.assertEqual(payload["new_videos_detected"], 1)
        self.assertEqual(payload["new_video_ids"], ["new-video-1"])
        self.assertTrue(payload["pipelines"][0]["succeeded"])
        self.assertEqual(payload["pipelines"][0]["status"], "completed")
        self.assertEqual(
            payload["videos"][0],
            {
                "youtube_video_id": "new-video-1",
                "title": "Nouvelle vidéo",
                "view_count": 123,
                "like_count": 17,
                "comment_count": 4,
            },
        )

    def test_new_videos_are_compared_with_init_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_dir = Path(temporary_dir)
            archive_dir = download_dir / "20260802_1437"
            for video_id in ("existing-1", "new-1", "new-2"):
                (archive_dir / video_id).mkdir(parents=True)
            (download_dir / "init" / "existing-1").mkdir(parents=True)
            (download_dir / "init" / "old-not-returned").mkdir(parents=True)

            new_ids = update_runs.new_video_ids_from_archive(
                archive_dir,
                download_dir,
            )

            self.assertEqual(new_ids, ["new-1", "new-2"])

    def test_current_archive_is_compared_with_previous_dated_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_dir = Path(temporary_dir)
            older = download_dir / "20260801_0300"
            previous = download_dir / "20260802_0300"
            current = download_dir / "20260803_0300"
            future = download_dir / "20260804_0300"
            for root in (older, previous, current, future):
                root.mkdir()
            (download_dir / "_00_info_videos").mkdir()
            for video_id in ("existing-1", "existing-2"):
                (previous / video_id).mkdir()
            for video_id in ("existing-1", "existing-2", "new-1"):
                (current / video_id).mkdir()

            selected = update_runs.previous_archive_directory(
                download_dir,
                current,
            )
            new_ids = update_runs.new_video_ids_since_previous(
                current,
                selected,
            )

            self.assertEqual(selected, previous)
            self.assertEqual(new_ids, ["new-1"])

    def test_pipeline_selection_only_uses_archive_comparison(self) -> None:
        first_run_ids = update_runs.pipeline_video_ids_from_archives(
            ["new-versus-init"],
            [],
            has_previous_archive=False,
        )
        later_run_ids = update_runs.pipeline_video_ids_from_archives(
            ["new-versus-init", "missing-from-sql"],
            ["new-versus-previous"],
            has_previous_archive=True,
        )

        self.assertEqual(first_run_ids, ["new-versus-init"])
        self.assertEqual(later_run_ids, ["new-versus-previous"])

    def test_new_comment_ids_are_compared_with_previous_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_dir = Path(temporary_dir)
            previous = download_dir / "20260802_0300"
            current = download_dir / "20260803_0300"
            previous_comments = (
                previous / "video-1" / "metadata" / "youtube_comments.json"
            )
            current_comments = (
                current / "video-1" / "metadata" / "youtube_comments.json"
            )
            update_runs.write_json(
                previous_comments,
                {
                    "comments": [
                        {"youtube_comment_id": "comment-1"},
                        {"youtube_comment_id": "comment-2"},
                    ]
                },
            )
            update_runs.write_json(
                current_comments,
                {
                    "comments": [
                        {"youtube_comment_id": "comment-1"},
                        {"youtube_comment_id": "comment-2"},
                        {"youtube_comment_id": "comment-3"},
                    ]
                },
            )

            new_comments = update_runs.new_comment_ids_by_video(
                current,
                previous,
            )

            self.assertEqual(new_comments, {"video-1": ["comment-3"]})

    def test_new_video_is_downloaded_and_pipeline_runs_in_archive(self) -> None:
        video = {
            "id": "new-video-1",
            "snippet": {
                "title": "Nouvelle video",
                "publishedAt": "2026-08-03T10:00:00Z",
                "thumbnails": {},
            },
            "contentDetails": {"duration": "PT1M"},
            "statistics": {},
        }
        archive_dir = Path("downloads/youtube/20260803_0300")
        download_dir = Path("downloads/youtube")

        with (
            patch.dict("os.environ", {"PIPELINE_EXECUTION_BACKEND": "local"}),
            patch.object(update_runs, "download_video") as download,
            patch.object(update_runs.subprocess, "run") as pipeline_run,
        ):
            update_runs.download_and_run_pipeline(
                video,
                archive_dir,
                download_dir,
                datetime(2026, 8, 3).date(),
            )

        self.assertEqual(download.call_args.args[1], archive_dir)
        self.assertEqual(download.call_args.args[2], download_dir)
        self.assertFalse(download.call_args.kwargs["reuse_previous"])
        self.assertFalse(download.call_args.kwargs["sync_cached_metadata"])
        self.assertEqual(pipeline_run.call_count, 3)
        run_command = pipeline_run.call_args_list[0].args[0]
        embedding_command = pipeline_run.call_args_list[1].args[0]
        sql_command = pipeline_run.call_args_list[2].args[0]
        self.assertEqual(
            run_command[:4],
            [update_runs.sys.executable, "-m", "pipeline", "run"],
        )
        self.assertEqual(run_command[4], "new-video-1")
        self.assertEqual(run_command[5], "--root")
        self.assertEqual(Path(run_command[6]), archive_dir.resolve())
        self.assertIn("embeddings.create", embedding_command)
        self.assertEqual(
            sql_command[1:3],
            ["-m", "pipeline.publish.sync_database"],
        )
        self.assertIn("--video-id", sql_command)
        snapshot_flag = sql_command.index("--snapshot-date")
        self.assertEqual(sql_command[snapshot_flag + 1], "2026-08-03")
        self.assertTrue(
            all(call.kwargs["check"] for call in pipeline_run.call_args_list)
        )

    def test_scaleway_backend_delegates_without_local_download(self) -> None:
        video = {"id": "new-video-1"}
        expected = {
            "backend": "scaleway",
            "job_id": "job-123",
            "s3_uri": "s3://bucket/youtube/archive/new-video-1",
        }
        with (
            patch.dict("os.environ", {"PIPELINE_EXECUTION_BACKEND": "scaleway"}),
            patch.object(
                update_runs,
                "run_pipeline_on_scaleway",
                return_value=expected,
            ) as remote,
            patch.object(update_runs, "download_video") as local_download,
        ):
            result = update_runs.download_and_run_pipeline(
                video,
                Path("downloads/youtube/20260815_0300"),
                Path("downloads/youtube"),
                datetime(2026, 8, 15).date(),
            )

        self.assertEqual(result, expected)
        remote.assert_called_once()
        local_download.assert_not_called()

    def test_stats_snapshot_is_written_even_when_counts_are_unchanged(self) -> None:
        class RecordingCursor:
            def __init__(self) -> None:
                self.calls = []

            def execute(self, sql, params=None) -> None:
                self.calls.append((" ".join(sql.split()), params))

        cursor = RecordingCursor()
        payload = {
            "statistics": {
                "viewCount": "100",
                "likeCount": "10",
                "commentCount": "2",
            }
        }
        first_date = datetime(2026, 8, 2).date()
        second_date = datetime(2026, 8, 3).date()

        sync_database.upsert_video_stats(cursor, 42, payload, first_date)
        sync_database.upsert_video_stats(cursor, 42, payload, second_date)

        self.assertEqual(len(cursor.calls), 2)
        self.assertIn("ON CONFLICT (video_id, snapshot_date)", cursor.calls[0][0])
        self.assertEqual(cursor.calls[0][1], (42, first_date, 100, 10, 2))
        self.assertEqual(cursor.calls[1][1], (42, second_date, 100, 10, 2))

    def test_youtube_api_retries_a_transient_server_error(self) -> None:
        failed = Mock(status_code=500)
        succeeded = Mock(status_code=200)
        succeeded.json.return_value = {"items": []}

        with (
            patch.dict("os.environ", {"YOUTUBE_API_KEY": "test-key"}),
            patch.object(
                fetch_youtube_metadata.requests,
                "get",
                side_effect=[failed, succeeded],
            ) as request,
            patch.object(fetch_youtube_metadata.time, "sleep"),
        ):
            payload = fetch_youtube_metadata.youtube("videos", id="video-1")

        self.assertEqual(payload, {"items": []})
        self.assertEqual(request.call_count, 2)
        succeeded.raise_for_status.assert_called_once_with()

    def test_daily_sync_rejects_selection_options(self) -> None:
        rejected_options = (
            ["--limit", "12"],
            ["--video-url", "https://youtu.be/abcdefghijk"],
            ["--download-dir", "other"],
            ["--skip-comments"],
        )
        for argv in rejected_options:
            with self.subTest(argv=argv), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    update_runs.parse_args(argv)

    def test_collection_uses_the_same_channel_request_as_initial_ingestion(self) -> None:
        expected = [{"id": "video-1"}]
        with patch.object(
            update_runs,
            "fetch_videos",
            return_value=expected,
        ) as fetch_videos:
            videos = update_runs.collect_videos()

        self.assertEqual(videos, expected)
        fetch_videos.assert_called_once_with(update_runs.CHANNEL)

    def test_incremental_comments_keep_existing_rows_and_soft_delete_missing(self) -> None:
        class RecordingCursor:
            def __init__(self) -> None:
                self.calls = []
                self.rowcount = 0
                self.last_sql = ""
                self.last_params = None

            def execute(self, sql, params=None) -> None:
                self.last_sql = " ".join(sql.split())
                self.last_params = params
                self.calls.append((self.last_sql, params))
                self.rowcount = 1 if "SET is_deleted = TRUE" in self.last_sql else 0

            def fetchall(self):
                return [("top-1",), ("missing-1",)]

            def fetchone(self):
                return (10 if self.last_params[2] == "top-1" else 11,)

        cursor = RecordingCursor()
        collected_at = datetime(2026, 8, 2, 3, 0, tzinfo=timezone.utc)
        result = sync_database.sync_video_comments_incremental(
            cursor,
            42,
            {
                "comments": [
                    {
                        "youtube_comment_id": "top-1",
                        "text": "Texte actualise",
                        "like_count": 4,
                        "updated_at": "2026-08-02T02:00:00Z",
                    },
                    {
                        "youtube_comment_id": "reply-1",
                        "parent_youtube_comment_id": "top-1",
                        "text": "Nouveau",
                    },
                ]
            },
            collected_at=collected_at,
        )

        self.assertEqual(result.seen, 2)
        self.assertEqual(result.created, 1)
        self.assertEqual(result.refreshed, 1)
        self.assertEqual(result.deleted, 1)
        self.assertEqual(cursor.calls[1][1][1], None)
        self.assertEqual(cursor.calls[2][1][1], 10)
        conflict_update = cursor.calls[1][0].split(
            "ON CONFLICT (youtube_comment_id) DO UPDATE SET",
            1,
        )[1]
        self.assertNotIn("first_seen_at =", conflict_update)
        self.assertEqual(cursor.calls[-1][1], (42, ["top-1", "reply-1"]))

    def test_run_journal_receives_all_metrics(self) -> None:
        class RecordingCursor:
            def __init__(self) -> None:
                self.params = None

            def execute(self, _sql, params=None) -> None:
                self.params = params

        cursor = RecordingCursor()
        metrics = update_runs.empty_metrics()
        metrics.update({"videos_updated": 2, "comments_new": 3})

        update_runs.finish_run(cursor, 7, "completed", metrics, [])

        self.assertEqual(cursor.params[0], "completed")
        self.assertEqual(cursor.params[2], 2)
        self.assertEqual(cursor.params[6], 3)
        self.assertEqual(cursor.params[-1], 7)

    def test_run_journal_records_the_archive_path(self) -> None:
        class RecordingCursor:
            def __init__(self) -> None:
                self.params = None

            def execute(self, _sql, params=None) -> None:
                self.params = params

            def fetchone(self):
                return (12,)

        cursor = RecordingCursor()
        run_id = update_runs.create_run(
            cursor,
            Path("downloads/youtube/20260802_1437"),
        )

        self.assertEqual(run_id, 12)
        self.assertEqual(
            cursor.params,
            (str(Path("downloads/youtube/20260802_1437")),),
        )


if __name__ == "__main__":
    unittest.main()
