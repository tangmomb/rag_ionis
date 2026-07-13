from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dagster as dg

from dagster_pipeline import definitions as pipeline


class DagsterPipelineTests(unittest.TestCase):
    def test_catalog_job_materializes_graph_for_one_video_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_dir = root / "20260713_1200_init" / "video123"
            (video_dir / "metadata").mkdir(parents=True)
            (video_dir / "outputs" / "images" / "graphic").mkdir(parents=True)
            (video_dir / "outputs" / "ocr").mkdir(parents=True)
            (video_dir / "outputs" / "transcripts_ocr").mkdir(parents=True)
            (video_dir / "outputs" / "chunks").mkdir(parents=True)
            (video_dir / "video123.mp4").write_bytes(b"video")
            (video_dir / "outputs" / "images" / "graphic" / "00_01.jpg").write_bytes(b"image")
            (video_dir / "metadata" / "youtube_video_metadata.json").write_text(
                json.dumps({"youtube_video_id": "video123", "title": "Vidéo test", "duration_seconds": 42}),
                encoding="utf-8",
            )
            (video_dir / "metadata" / "pipeline_analysis.json").write_text(
                json.dumps({"video_type": "interview", "has_subtitles": True}), encoding="utf-8"
            )
            (video_dir / "outputs" / "ocr" / "03_reviewed_ocr_overlays.json").write_text(
                json.dumps({"kinds": {"subtitle": {"00:01": "Bonjour"}}}), encoding="utf-8"
            )
            (video_dir / "outputs" / "transcripts_ocr" / "plain_transcript.txt").write_text(
                "Bonjour depuis le transcript. " * 10, encoding="utf-8"
            )
            (video_dir / "outputs" / "transcripts_ocr" / "video_summary.md").write_text(
                "# Résumé", encoding="utf-8"
            )
            (video_dir / "outputs" / "chunks" / "transcript_chunks.json").write_text(
                json.dumps({"chunks": [{"chunk_index": 1, "content": "Premier chunk"}]}), encoding="utf-8"
            )
            (video_dir / "outputs" / "chunks" / "chunk_01_embedding.json").write_text("{}", encoding="utf-8")

            with patch.object(pipeline, "DOWNLOAD_ROOT", root), dg.instance_for_test() as instance:
                instance.add_dynamic_partitions(pipeline.VIDEO_PARTITIONS.name, ["video123"])
                job = pipeline.defs.resolve_job_def("cataloguer_video")
                result = job.execute_in_process(instance=instance, partition_key="video123")

            self.assertTrue(result.success)
            self.assertEqual(len(result.get_asset_materialization_events()), 6)
            check_events = [event for event in result.all_events if event.event_type_value == "ASSET_CHECK_EVALUATION"]
            self.assertEqual(len(check_events), 6)
            self.assertTrue(all(event.event_specific_data.passed for event in check_events))


if __name__ == "__main__":
    unittest.main()
