import json
import tempfile
import unittest
from pathlib import Path

from dagster_pipeline.asset_metadata import (
    changed_result_files,
    snapshot_result_files,
    step_output_metadata,
)
from dagster_pipeline.runtime import explorer_metadata


class DagsterAssetMetadataTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_interview_metadata_explains_negative_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            manifest = video_dir / "outputs" / "interview" / "interview_detection_manifest.json"
            self._write_json(
                manifest,
                {
                    "is_interview": False,
                    "frame_count": 107,
                    "selected_frame_count": 68,
                    "sequence_count": 6,
                    "blocked_by_graphic_majority": False,
                    "thresholds": {"max_interview_sequences": 5},
                    "sequences": [
                        {
                            "sequence_id": 1,
                            "start_second": 2,
                            "end_second": 9,
                            "duration_seconds": 7,
                            "frame_count": 15,
                        }
                    ],
                },
            )

            metadata = step_output_metadata("step_05_detect_interviews", "video123", video_dir)

        self.assertEqual(metadata["resultat"], "Pas une interview")
        self.assertIs(metadata["is_interview"], False)
        self.assertEqual(metadata["sequences_detectees"], 6)
        self.assertIn("moins de 5", metadata["raison"])
        self.assertEqual(metadata["apercu_sequences"].value[0]["duree_s"], 7)
        self.assertEqual(metadata["fichier_principal"].value, str(manifest))

    def test_subtitle_metadata_makes_selected_route_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            self._write_json(
                video_dir / "metadata" / "pipeline_analysis.json",
                {
                    "has_subtitles": True,
                    "has_subtitles_details": {
                        "reason": "stable_anchor_across_continuous_seconds",
                        "anchor_count": 12,
                        "longest_continuous_seconds_duration": 16.0,
                        "required_continuous_seconds": 10.0,
                    },
                },
            )

            metadata = step_output_metadata("step_09_detect_ocr_subtitles", "video123", video_dir)

        self.assertEqual(metadata["resultat"], "Sous-titres détectés")
        self.assertIs(metadata["has_subtitles"], True)
        self.assertEqual(metadata["route_choisie"], "has_sub")
        self.assertEqual(metadata["duree_continue_secondes"], 16.0)

    def test_result_file_snapshot_reports_changes_and_ignores_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "video123"
            output = video_dir / "outputs" / "result.json"
            cache = video_dir / "outputs" / "images" / ".embedding_cache" / "cache.joblib"
            output.parent.mkdir(parents=True)
            cache.parent.mkdir(parents=True)
            output.write_text("before", encoding="utf-8")
            cache.write_text("before", encoding="utf-8")
            before = snapshot_result_files(video_dir)

            output.write_text("after with a different size", encoding="utf-8")
            cache.write_text("after", encoding="utf-8")
            changed = changed_result_files(before, snapshot_result_files(video_dir))

        self.assertEqual(changed, ["outputs/result.json"])

    def test_video_identity_uses_the_first_extracted_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "20260713_1200_init" / "video123"
            self._write_json(
                video_dir / "metadata" / "youtube_video_metadata.json",
                {
                    "youtube_video_id": "video123",
                    "title": "Vidéo facile à reconnaître",
                    "url": "https://www.youtube.com/watch?v=video123",
                    "thumbnail_medium_url": "https://example.test/youtube.jpg",
                },
            )
            later = video_dir / "outputs" / "images" / "footage" / "00_02.jpg"
            first = video_dir / "outputs" / "images" / "graphic" / "00_00.jpg"
            later.parent.mkdir(parents=True)
            first.parent.mkdir(parents=True)
            later.touch()
            first.touch()

            metadata = explorer_metadata("video123", video_dir)

        self.assertEqual(metadata["titre_video"], "Vidéo facile à reconnaître")
        self.assertIn("outputs/images/graphic/00_00.jpg", metadata["image_apercu"].value)
        self.assertIn("![Première frame extraite]", metadata["apercu_video"].value)


if __name__ == "__main__":
    unittest.main()
