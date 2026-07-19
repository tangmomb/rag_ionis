import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.steps.ocr import review_other_text_candidates as review
from pipeline.steps.speakers import validate_speakers as speakers
from pipeline.support.json_io import write_json
from pipeline.support.openai_batch import (
    batch_request_fingerprint,
    batch_state_matches,
    save_batch_state,
)


class BatchFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_tracks_every_input_dimension(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.txt"
            source.write_text("alpha", encoding="utf-8")
            first = batch_request_fingerprint(
                "model-a",
                sources=(source,),
                custom_ids=("one", "two"),
                options={"temperature": 0, "profile": "strict"},
            )
            reordered_mapping = batch_request_fingerprint(
                "model-a",
                sources=(source,),
                custom_ids=("one", "two"),
                options={"profile": "strict", "temperature": 0},
            )
            self.assertEqual(first, reordered_mapping)
            self.assertTrue(
                batch_state_matches(
                    {"request_fingerprint": first},
                    first,
                )
            )
            self.assertFalse(batch_state_matches({}, first))
            self.assertNotEqual(
                first,
                batch_request_fingerprint(
                    "model-b",
                    sources=(source,),
                    custom_ids=("one", "two"),
                    options={"temperature": 0, "profile": "strict"},
                ),
            )
            source.write_text("beta", encoding="utf-8")
            self.assertNotEqual(
                first,
                batch_request_fingerprint(
                    "model-a",
                    sources=(source,),
                    custom_ids=("one", "two"),
                    options={"temperature": 0, "profile": "strict"},
                ),
            )

    def test_speaker_validation_reinitializes_an_incompatible_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            video_path = Path(temporary_dir) / "video.mp4"
            speaker_dir = Path(temporary_dir) / "outputs" / "speakers"
            speaker_dir.mkdir(parents=True)
            write_json(
                speaker_dir / speakers.SPEAKER_CANDIDATES_NAME,
                {"video_title": "Test", "speakers": ["Alice"]},
            )
            state_path = speakers.batch_state_path(video_path)
            save_batch_state(
                state_path,
                {
                    "batch_id": "old-speakers",
                    "status": "in_progress",
                    "request_fingerprint": "stale",
                },
            )

            with patch.object(
                speakers,
                "submit_batch_validation",
                return_value=state_path,
            ) as submit:
                result = speakers.validate_file_batch(
                    "model-new",
                    video_path,
                    wait=False,
                )

            self.assertEqual(result, state_path)
            submit.assert_called_once()
            self.assertNotEqual(
                submit.call_args.kwargs["request_fingerprint"],
                "stale",
            )

    def test_review_reinitializes_an_incompatible_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            video_path = Path(temporary_dir) / "video.mp4"
            source_dir = (
                Path(temporary_dir)
                / "outputs"
                / "ocr"
                / review.SOURCE_DIRNAME
            )
            source_dir.mkdir(parents=True)
            crop = source_dir / "crop.png"
            crop.write_bytes(b"image-v1")
            write_json(
                source_dir / review.SOURCE_MANIFEST_NAME,
                {
                    "items": [
                        {
                            "crop": crop.name,
                            "text": "IONIS",
                            "timecode": "00:01",
                        }
                    ]
                },
            )
            state_path = review.batch_state_path(video_path)
            save_batch_state(
                state_path,
                {
                    "batch_id": "old-review",
                    "status": "in_progress",
                    "request_fingerprint": "stale",
                },
            )

            with patch.object(
                review,
                "submit_batch_review",
                return_value=state_path,
            ) as submit:
                result = review.review_video_batch(
                    video_path,
                    "model-new",
                    wait=False,
                )

            self.assertEqual(result, state_path)
            submit.assert_called_once()
            self.assertNotEqual(
                submit.call_args.kwargs["request_fingerprint"],
                "stale",
            )

    def test_review_rejects_a_partial_batch_result(self):
        jobs = [{"custom_id": "review-001"}]
        with (
            patch.object(review, "openai_client"),
            patch.object(review, "download_batch_files"),
            patch.object(review, "parse_jsonl", return_value=[]),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "custom_id manquant",
            ):
                review.finalize_batch_review(
                    Path("video.mp4"),
                    "model",
                    jobs,
                    {"output_file_id": "output"},
                )


if __name__ == "__main__":
    unittest.main()
