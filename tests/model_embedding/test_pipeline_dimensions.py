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
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if "openai" not in sys.modules:
    openai_stub = ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from pipeline.steps.embeddings import create_chunk_embeddings as chunk_embeddings
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
                json.dumps(
                    {
                        "model": DEFAULT_EMBEDDING_MODEL,
                        "content": "Texte actuel",
                        "embedding": [0.0] * 2000,
                    }
                ),
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
            self.assertFalse(
                chunk_embeddings.existing_embedding_matches(
                    path,
                    DEFAULT_EMBEDDING_MODEL,
                    DEFAULT_EMBEDDING_DIMENSIONS,
                    expected_text="Nouveau texte",
                )
            )

    def test_hierarchical_embedding_paths_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "video.mp4"

            with patch.object(chunk_embeddings, "chunks_dir", return_value=root):
                detail = chunk_embeddings.embedding_path(video, 1, "detail")
                section = chunk_embeddings.embedding_path(video, 1, "section")
                global_chunk = chunk_embeddings.embedding_path(video, 1, "global")

            self.assertEqual(detail.name, "chunk_01_embedding.json")
            self.assertEqual(section.name, "chunk_section_01_embedding.json")
            self.assertEqual(global_chunk.name, "chunk_global_01_embedding.json")

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
            chunks = [
                {
                    "chunk_index": 1,
                    "content": "Texte du chunk",
                    "meta_data": {"speakers": ["Alice Martin"]},
                }
            ]

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
            self.assertEqual(payload["chunk_level"], "detail")
            self.assertIsNone(payload["chunk_parent_id"])
            self.assertNotIn("speakers", payload)
            self.assertNotIn("speakers", payload.get("meta_data", {}))

    def test_cached_embedding_speaker_metadata_is_removed_without_api_call(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "chunks.json"
            source.write_text("{}", encoding="utf-8")
            target = root / "chunk_01_embedding.json"
            target.write_text(
                json.dumps(
                    {
                        "model": DEFAULT_EMBEDDING_MODEL,
                        "content": "Texte du chunk",
                        "meta_data": {
                            "speakers": ["Alice Martin"],
                            "summary_strategy": "luna",
                        },
                        "embedding": [0.0] * DEFAULT_EMBEDDING_DIMENSIONS,
                    }
                ),
                encoding="utf-8",
            )
            video = root / "video.mp4"
            video.touch()
            client = SimpleNamespace(embeddings=SimpleNamespace(create=Mock()))
            chunks = [{"chunk_index": 1, "content": "Texte du chunk"}]

            with (
                patch.object(chunk_embeddings, "chunks_path", return_value=source),
                patch.object(chunk_embeddings, "load_chunks", return_value=({}, chunks)),
                patch.object(chunk_embeddings, "chunks_dir", return_value=root),
                patch.object(chunk_embeddings, "embedding_path", return_value=target),
            ):
                result = chunk_embeddings.create_embeddings(
                    client,
                    DEFAULT_EMBEDDING_MODEL,
                    DEFAULT_EMBEDDING_DIMENSIONS,
                    video,
                )

            client.embeddings.create.assert_not_called()
            self.assertIsNone(result)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["meta_data"], {"summary_strategy": "luna"})


if __name__ == "__main__":
    unittest.main()
