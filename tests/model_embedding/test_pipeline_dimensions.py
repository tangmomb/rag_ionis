from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
INIT_DIR = PROJECT_DIR / "scripts" / "init"
if str(INIT_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_DIR))

if "openai" not in sys.modules:
    openai_stub = ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from common import used_by_hs23_ns24_create_chunk_embeddings as chunk_embeddings
from interface.backend.config import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL


class PipelineEmbeddingDimensionsTests(unittest.TestCase):
    def test_production_defaults_use_large_2000(self):
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "text-embedding-3-large")
        self.assertEqual(DEFAULT_EMBEDDING_DIMENSIONS, 2000)
        self.assertEqual(chunk_embeddings.DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(chunk_embeddings.DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_DIMENSIONS)

    def test_existing_embedding_must_match_model_and_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "embedding.json"
            path.write_text(
                json.dumps({"model": DEFAULT_EMBEDDING_MODEL, "embedding": [0.0] * 2000}),
                encoding="utf-8",
            )
            self.assertTrue(
                chunk_embeddings.existing_embedding_matches(
                    path, DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_DIMENSIONS
                )
            )
            self.assertFalse(
                chunk_embeddings.existing_embedding_matches(path, DEFAULT_EMBEDDING_MODEL, 3072)
            )

    def test_chunk_generation_requests_2000_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "chunks.json"
            source.write_text("{}", encoding="utf-8")
            target = root / "chunk_01_embedding.json"
            video = root / "video.mp4"
            video.write_bytes(b"")
            response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 2000)])
            client = SimpleNamespace(embeddings=SimpleNamespace(create=Mock(return_value=response)))
            chunks = [{"chunk_index": 1, "content": "Texte du chunk", "meta_data": {}}]

            with (
                patch.object(chunk_embeddings, "chunks_path", return_value=source),
                patch.object(chunk_embeddings, "load_chunks", return_value=({}, chunks)),
                patch.object(chunk_embeddings, "chunks_dir", return_value=root),
                patch.object(chunk_embeddings, "embedding_path", return_value=target),
                patch.object(chunk_embeddings, "relative_to_video_dir", return_value="chunks.json"),
            ):
                chunk_embeddings.create_embeddings(
                    client,
                    DEFAULT_EMBEDDING_MODEL,
                    DEFAULT_EMBEDDING_DIMENSIONS,
                    video,
                )

            client.embeddings.create.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
                input="Texte du chunk",
            )
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["dimensions"], 2000)
            self.assertEqual(len(payload["embedding"]), 2000)


if __name__ == "__main__":
    unittest.main()
