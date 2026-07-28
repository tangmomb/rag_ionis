from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.ingest import download_videos
from pipeline.ingest import fetch_youtube_metadata as get_data
from pipeline.publish import sync_database as update_sql
from pipeline.publish import upload_outputs_to_s3 as upload_s3
from pipeline.support.paths import consolidate_init_dir


class IngestionPublicationTests(unittest.TestCase):
    def test_sql_publication_uses_only_the_whisperx_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            video_dir = Path(temporary_dir)
            whisper_dir = video_dir / "outputs" / "transcripts_whisper"
            ocr_dir = video_dir / "outputs" / "transcripts_ocr"
            whisper_dir.mkdir(parents=True)
            ocr_dir.mkdir(parents=True)
            (whisper_dir / "plain_transcript.txt").write_text(
                "Canonique WhisperX",
                encoding="utf-8",
            )
            (whisper_dir / "transcript_3_enriched.txt").write_text(
                "[00:00] INTERCALAIRE: Canonique WhisperX",
                encoding="utf-8",
            )
            (ocr_dir / "plain_transcript.txt").write_text(
                "Comparaison OCR",
                encoding="utf-8",
            )

            found = update_sql.transcript_paths(video_dir)
            for path in whisper_dir.iterdir():
                path.unlink()
            whisper_dir.rmdir()
            ocr_only = update_sql.transcript_paths(video_dir)

        self.assertEqual(
            [(kind, path.parent.name) for kind, path in found],
            [
                ("plain", "transcripts_whisper"),
                ("enriched", "transcripts_whisper"),
            ],
        )
        self.assertEqual(ocr_only, [])

    def test_hierarchical_embedding_loader_uses_level_specific_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            chunks_dir = Path(temporary_dir)
            (chunks_dir / "chunk_01_embedding.json").write_text(
                json.dumps({"content": "detail"}),
                encoding="utf-8",
            )
            (chunks_dir / "chunk_section_01_embedding.json").write_text(
                json.dumps({"content": "section"}),
                encoding="utf-8",
            )

            with patch.object(update_sql, "existing_chunks_dir", return_value=chunks_dir):
                payload = update_sql.load_chunk_embedding_payload(
                    Path("video"),
                    chunk_index=1,
                    chunk_level="section",
                )

        self.assertEqual(payload["content"], "section")

    def test_chunk_upsert_defaults_to_detail_level(self) -> None:
        class RecordingCursor:
            def __init__(self):
                self.calls = []

            def execute(self, sql, params=None):
                self.calls.append((sql, params))

        cursor = RecordingCursor()

        inserted = update_sql.upsert_chunk(
            cursor,
            video_id=12,
            chunk_payload={"chunk_index": 3, "content": "Contenu"},
        )

        self.assertTrue(inserted)
        sql, params = cursor.calls[0]
        self.assertIn("ON CONFLICT (video_id, chunk_level, chunk_index)", sql)
        self.assertNotIn("speakers", sql)
        self.assertEqual(params[:4], (12, 3, "detail", None))

    def test_video_speakers_are_loaded_from_validation_not_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            video_dir = Path(temporary_dir) / "video123"
            speakers_dir = video_dir / "outputs" / "speakers"
            speakers_dir.mkdir(parents=True)
            (speakers_dir / "speakers_validated.json").write_text(
                json.dumps(
                    {
                        "speakers": [
                            "Alice Martin",
                            " Alice   Martin ",
                            "Bob Durand",
                        ],
                        "speaker_details": [
                            {
                                "speaker": "Alice Martin",
                                "title": "Directrice générale",
                            },
                            {
                                "speaker": " Alice   Martin ",
                                "title": "Doublon",
                            },
                            {
                                "speaker": "Bob Durand",
                                "title": "",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            speakers = update_sql.load_video_speakers(video_dir)
            details = update_sql.load_video_speaker_details(video_dir)

        self.assertEqual(speakers, ["Alice Martin", "Bob Durand"])
        self.assertEqual(
            details,
            [
                {"name": "Alice Martin", "title": "Directrice générale"},
                {"name": "Bob Durand", "title": None},
            ],
        )

    def test_video_speakers_are_replaced_in_the_relational_table(self) -> None:
        class RecordingCursor:
            def __init__(self) -> None:
                self.calls = []

            def execute(self, sql, params=None) -> None:
                self.calls.append((sql, params))

        cursor = RecordingCursor()

        count = update_sql.replace_video_speakers(
            cursor,
            42,
            [
                {"name": "Alice Martin", "title": "Directrice générale"},
                {"name": "Bob Durand", "title": None},
            ],
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            cursor.calls[0],
            ("DELETE FROM speakers WHERE video_id = %s", (42,)),
        )
        self.assertIn("INSERT INTO speakers", cursor.calls[1][0])
        self.assertEqual(
            cursor.calls[1][1],
            (42, "Alice Martin", "Directrice générale"),
        )

    def test_chunk_upsert_rejects_unknown_level(self) -> None:
        with self.assertRaises(ValueError):
            update_sql.upsert_chunk(
                cursor=None,
                video_id=12,
                chunk_payload={
                    "chunk_index": 3,
                    "chunk_level": "chapter",
                    "content": "Contenu",
                },
            )

    def test_hierarchical_chunk_parents_are_resolved_to_database_ids(self) -> None:
        class ReturningCursor:
            def __init__(self) -> None:
                self.calls = []
                self.next_id = 100

            def execute(self, sql, params=None):
                self.calls.append((sql, params))

            def fetchone(self):
                self.next_id += 1
                return (self.next_id,)

        cursor = ReturningCursor()
        chunks = [
            {
                "chunk_index": 1,
                "chunk_level": "detail",
                "chunk_parent": {"chunk_level": "section", "chunk_index": 1},
                "content": "Detail",
            },
            {
                "chunk_index": 1,
                "chunk_level": "global",
                "content": "Global",
            },
            {
                "chunk_index": 1,
                "chunk_level": "section",
                "chunk_parent": {"chunk_level": "global", "chunk_index": 1},
                "content": "Section",
            },
        ]

        with patch.object(update_sql, "load_chunk_embedding_payload", return_value=None):
            inserted = update_sql.upsert_video_chunks(
                cursor,
                video_id=12,
                video_path=Path("video"),
                chunks=chunks,
            )

        self.assertEqual(inserted, 3)
        params = [call[1] for call in cursor.calls]
        self.assertEqual(params[0][2:4], ("global", None))
        self.assertEqual(params[1][2:4], ("section", 101))
        self.assertEqual(params[2][2:4], ("detail", 102))

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

    def test_youtube_metadata_overwrites_existing_local_video_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_root = Path(temporary_dir)
            video_id = "abcdefghijk"
            video_dir = download_root / "init" / video_id
            video_dir.mkdir(parents=True)
            video_path = video_dir / f"{video_id}.mp4"
            video_path.write_bytes(b"video")
            target = video_dir / "metadata" / "youtube_video_metadata.json"
            target.parent.mkdir()
            target.write_text('{"title": "ancien"}\n', encoding="utf-8")
            video = {
                "id": video_id,
                "snippet": {
                    "publishedAt": "2026-07-14T12:00:00Z",
                    "title": "Nouveau titre",
                    "description": "Description",
                },
                "contentDetails": {"duration": "PT1M31S"},
                "statistics": {"viewCount": "371"},
            }
            cache_path = get_data.write_video_info(
                video,
                download_root / "info_videos",
            )

            synced = get_data.sync_existing_video_info(
                video,
                cache_path,
                download_root,
            )

            self.assertEqual(synced, target)
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8"))["title"],
                "Nouveau titre",
            )
            self.assertEqual(video_path.read_bytes(), b"video")

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

    def test_database_publication_requires_an_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            ready = run_dir / "abcdefghijk"
            pending = run_dir / "lmnopqrstuv"
            ready.mkdir()
            pending.mkdir()
            (ready / "outputs" / "chunks").mkdir(parents=True)
            (ready / "outputs" / "chunks" / "chunk_01_embedding.json").write_text(
                "{}", encoding="utf-8"
            )

            candidates = update_sql.candidate_video_dirs(run_dir)

            self.assertEqual(update_sql.ready_video_dirs(candidates), [ready])

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
        self.assertIn("CREATE TABLE IF NOT EXISTS speakers", statements[-1])
        self.assertIn("video_id BIGINT NOT NULL REFERENCES videos(id)", statements[-1])
        self.assertIn("name TEXT NOT NULL", statements[-1])
        self.assertIn("title TEXT", statements[-1])
        self.assertIn("is_long_video BOOLEAN GENERATED ALWAYS", statements[-1])
        self.assertIn("transcript TEXT", statements[-1])
        self.assertIn("transcript_enriched TEXT", statements[-1])
        self.assertNotIn("transcript_timecodes TEXT", statements[-1])
        self.assertNotIn("transcript_timecodes_enrichi TEXT", statements[-1])
        self.assertNotIn("video_summary", statements[-1])
        self.assertNotIn("DROP SCHEMA IF EXISTS chat", "\n".join(statements))

if __name__ == "__main__":
    unittest.main()
