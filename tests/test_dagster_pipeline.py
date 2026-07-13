import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dagster as dg

from dagster_pipeline.assets import video_source
from dagster_pipeline.definitions import defs
from dagster_pipeline.ingestion import _selection_args
from dagster_pipeline import runtime


class DagsterPipelineTests(unittest.TestCase):
    def _create_video(self, root: Path, video_id: str = "video123") -> Path:
        video_dir = root / "20260713_1200_init" / video_id
        (video_dir / "metadata").mkdir(parents=True)
        (video_dir / f"{video_id}.mp4").write_bytes(b"video")
        (video_dir / "metadata" / "youtube_video_metadata.json").write_text(
            json.dumps({"youtube_video_id": video_id, "title": "Vidéo test"}),
            encoding="utf-8",
        )
        return video_dir

    def test_definitions_are_loadable(self) -> None:
        dg.Definitions.validate_loadable(defs)

    def test_video_source_materializes_one_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_video(root)
            with patch.object(runtime, "DOWNLOAD_ROOT", root), dg.instance_for_test() as instance:
                instance.add_dynamic_partitions(runtime.VIDEO_PARTITIONS.name, ["video123"])
                result = dg.materialize(
                    [video_source],
                    instance=instance,
                    partition_key="video123",
                )
        self.assertTrue(result.success)
        self.assertEqual(len(result.get_asset_materialization_events()), 1)

    def test_import_selection_supports_count_all_and_youtube_url(self) -> None:
        self.assertEqual(_selection_args("3"), ["--limit", "3"])
        self.assertEqual(_selection_args("all"), [])
        url = "https://www.youtube.com/watch?v=degQjvAoxvE"
        self.assertEqual(_selection_args(url), ["--video-url", url])

    def test_pipeline_settings_expose_launchpad_choices(self) -> None:
        fields = runtime.PipelineSettings.to_config_schema().as_field().config_type.fields

        def choices(name: str) -> list[str]:
            return [value.config_value for value in fields[name].config_type.enum_values]

        self.assertEqual(choices("openai_mode"), ["normal", "batch"])
        self.assertEqual(choices("review_scope"), ["duo", "all"])
        self.assertEqual(choices("correction_mode"), ["conservative", "balanced", "aggressive"])
        custom_model = runtime.PipelineSettings(chunk_speaker_validation_model="gpt-custom")
        self.assertEqual(custom_model.chunk_speaker_validation_model, "gpt-custom")
        with self.assertRaises(ValueError):
            runtime.PipelineSettings(openai_mode="live")


if __name__ == "__main__":
    unittest.main()
