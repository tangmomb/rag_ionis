from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.steps.inspection import extract_raw_ocr
from pipeline.support.json_io import read_json


class RecordingRawOcr:
    def __init__(self) -> None:
        self.images: list[Path] = []

    def recognize_raw(self, image_path: Path) -> object:
        self.images.append(image_path)
        return {"recognized": image_path.stem}


class ExtractRawOcrTests(unittest.TestCase):
    def test_extract_for_video_uses_typed_options_and_injected_recognizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "video"
            images_dir = root / "images"
            output_dir = root / "ocr" / "raw"
            image_paths = []
            for group_name in extract_raw_ocr.IMAGE_GROUPS:
                image_path = images_dir / group_name / f"{group_name}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"not-decoded-by-this-step")
                image_paths.append(image_path)

            recognizer = RecordingRawOcr()
            with (
                patch.object(
                    extract_raw_ocr,
                    "existing_images_dir",
                    return_value=images_dir,
                ),
                patch.object(
                    extract_raw_ocr,
                    "output_ocr_raw_dir",
                    return_value=output_dir,
                ),
            ):
                output_paths = extract_raw_ocr.extract_for_video(
                    video_path,
                    device="cpu",
                    lang="en",
                    min_confidence=0.75,
                    force=True,
                    ocr=recognizer,
                )

            self.assertCountEqual(recognizer.images, image_paths)
            self.assertEqual(len(output_paths), len(extract_raw_ocr.IMAGE_GROUPS))
            for group_name, output_path in zip(
                extract_raw_ocr.IMAGE_GROUPS,
                output_paths,
                strict=True,
            ):
                payload = read_json(output_path)
                self.assertEqual(payload["device"], "cpu")
                self.assertEqual(payload["lang"], "en")
                self.assertEqual(payload["min_confidence"], 0.75)
                self.assertEqual(
                    payload["items"],
                    [
                        {
                            "image": f"{group_name}/{group_name}.png",
                            "raw": {"recognized": group_name},
                        }
                    ],
                )


if __name__ == "__main__":
    unittest.main()
