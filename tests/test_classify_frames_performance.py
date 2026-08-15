from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipeline.steps.inspection import classify_frames


class RecordingPrefetchEmbedder:
    def __init__(self) -> None:
        self.config = SimpleNamespace(crop_bottom=0.0)
        self.prepared: list[list[int]] = []
        self.embedded: list[list[int]] = []

    def prepare_images(self, images):
        batch = list(images)
        self.prepared.append(batch)
        return batch

    def embed_prepared(self, prepared):
        batch = list(prepared)
        self.embedded.append(batch)
        return np.asarray([[value] for value in batch], dtype=np.float32)


class ClassifyFramesPerformanceTests(unittest.TestCase):
    def test_embedding_prefetch_preserves_batch_and_output_order(self) -> None:
        paths = [Path(f"{index}.png") for index in range(5)]
        embedder = RecordingPrefetchEmbedder()

        with patch.object(
            classify_frames,
            "load_image_rgb",
            side_effect=lambda path, crop_bottom=0.0: int(path.stem),
        ):
            result = classify_frames.embed_image_paths(
                paths,
                embedder,
                batch_size=2,
            )

        self.assertEqual(result[:, 0].tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(embedder.prepared, [[0, 1], [2, 3], [4]])
        self.assertEqual(embedder.embedded, [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()
