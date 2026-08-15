from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.steps.inspection import extract_raw_ocr
from pipeline.support.json_io import read_json
from pipeline.support.paddle_ocr import LocalPaddleOCR


class RecordingRawOcr:
    def __init__(self) -> None:
        self.images: list[Path] = []

    def recognize_raw(self, image_path: Path) -> object:
        self.images.append(image_path)
        return {"recognized": image_path.stem}


class RecordingBatchRawOcr(RecordingRawOcr):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[Path]] = []

    def recognize_raw_batch(self, image_paths: list[Path]) -> list[object]:
        paths = list(image_paths)
        self.batches.append(paths)
        return [{"recognized": path.stem} for path in paths]


class ExtractRawOcrTests(unittest.TestCase):
    def test_isolated_extraction_uses_a_clean_python_process(self) -> None:
        video_path = Path("videos") / "abcdefghijk.mp4"

        with patch.object(extract_raw_ocr.subprocess, "run") as run:
            output_paths = extract_raw_ocr.extract_for_video_isolated(
                video_path,
                device="gpu:1",
                lang="en",
                min_confidence=0.75,
                force=True,
            )

        command = run.call_args.args[0]
        self.assertEqual(command[0], extract_raw_ocr.sys.executable)
        self.assertEqual(command[1:3], ["-m", "pipeline.workers.raw_ocr"])
        self.assertIn(str(video_path.resolve()), command)
        self.assertIn("gpu:1", command)
        self.assertIn("en", command)
        self.assertIn("0.75", command)
        self.assertIn("--force", command)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(len(output_paths), len(extract_raw_ocr.IMAGE_GROUPS))

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

    def test_extract_for_video_batches_images_when_recognizer_supports_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_dir = root / "images"
            output_dir = root / "ocr" / "raw"
            for index in range(5):
                image_path = images_dir / "footage" / f"{index}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"raw")

            recognizer = RecordingBatchRawOcr()
            with (
                patch.object(extract_raw_ocr, "existing_images_dir", return_value=images_dir),
                patch.object(extract_raw_ocr, "output_ocr_raw_dir", return_value=output_dir),
            ):
                extract_raw_ocr.extract_for_video(
                    root / "video",
                    batch_size=2,
                    force=True,
                    ocr=recognizer,
                )

            self.assertEqual([len(batch) for batch in recognizer.batches], [2, 2, 1])
            self.assertEqual(recognizer.images, [])

    def test_local_paddle_ocr_passes_device_and_recognition_batch_size(self) -> None:
        captured = {}

        class FakePaddleOCR:
            def __init__(self, lang=None, text_recognition_batch_size=None, **kwargs):
                captured.update(
                    lang=lang,
                    text_recognition_batch_size=text_recognition_batch_size,
                    **kwargs,
                )

        fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
        with (
            patch("pipeline.support.paddle_ocr.install_torch_import_stub"),
            patch.dict("sys.modules", {"paddleocr": fake_module}),
        ):
            recognizer = LocalPaddleOCR(device="gpu:1", lang="en", batch_size=12)

        self.assertEqual(recognizer.backend, "paddleocr")
        self.assertEqual(captured["device"], "gpu:1")
        self.assertEqual(captured["text_recognition_batch_size"], 12)

    def test_local_paddle_ocr_maps_batch_results_to_each_image(self) -> None:
        payloads = [{"rec_texts": ["one"]}, {"rec_texts": ["two"]}]
        engine = Mock()
        engine.predict.return_value = [{"res": payload} for payload in payloads]
        recognizer = LocalPaddleOCR.__new__(LocalPaddleOCR)
        recognizer.engine = engine
        recognizer.backend = "paddleocr"
        recognizer.min_confidence = 0.9

        results = recognizer.recognize_raw_batch([Path("one.png"), Path("two.png")])

        self.assertEqual(results, [[payloads[0]], [payloads[1]]])
        self.assertEqual(
            engine.predict.call_args.kwargs["input"],
            ["one.png", "two.png"],
        )


if __name__ == "__main__":
    unittest.main()
