from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.steps.chunks.hierarchical_chunks import summarize_sections, summarize_video


class HierarchicalChunkingTests(unittest.TestCase):
    def test_long_profile_creates_section_and_global_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "abcdefghijk"
            chunks_dir = video_dir / "outputs" / "chunks"
            chunks_dir.mkdir(parents=True)
            video = video_dir / "abcdefghijk.mp4"
            video.touch()
            details = [
                {
                    "chunk_index": index,
                    "chunk_level": "detail",
                    "chunk_parent_id": None,
                    "content": (
                        f"La section {index} presente un sujet important. "
                        f"Elle donne un exemple concret numero {index}. "
                        "Cette explication permet de comprendre le raisonnement."
                    ),
                }
                for index in range(1, 8)
            ]
            target = chunks_dir / "transcript_chunks.json"
            target.write_text(
                json.dumps(
                    {
                        "chunking": {"profile": "long"},
                        "chunks": details,
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "pipeline.steps.chunks.hierarchical_chunks.luna_summary",
                side_effect=lambda text, **_kwargs: text.splitlines()[0],
            ):
                summarize_sections(video, force=True, details_per_section=3)
                summarize_video(video, force=True)
            payload = json.loads(target.read_text(encoding="utf-8"))

        chunks = payload["chunks"]
        global_chunks = [chunk for chunk in chunks if chunk["chunk_level"] == "global"]
        sections = [chunk for chunk in chunks if chunk["chunk_level"] == "section"]
        updated_details = [chunk for chunk in chunks if chunk["chunk_level"] == "detail"]
        self.assertEqual(len(global_chunks), 1)
        self.assertEqual(len(sections), 3)
        self.assertEqual(len(updated_details), 7)
        self.assertEqual(
            sections[0]["chunk_parent"],
            {"chunk_level": "global", "chunk_index": 1},
        )
        self.assertEqual(
            updated_details[0]["chunk_parent"],
            {"chunk_level": "section", "chunk_index": 1},
        )
        self.assertTrue(
            all(
                "speakers" not in chunk
                and "speakers" not in chunk.get("meta_data", {})
                for chunk in chunks
            )
        )
        self.assertEqual(payload["hierarchy"]["strategy"], "luna")


if __name__ == "__main__":
    unittest.main()
