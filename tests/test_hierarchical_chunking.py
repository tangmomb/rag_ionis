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

from pipeline.steps.chunks import hierarchical_chunks
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

    def test_batch_results_are_rebuilt_in_section_order(self) -> None:
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
                    "content": f"Contenu detail {index}.",
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
            jobs = hierarchical_chunks.section_summary_jobs(details, 3)
            output_path = (
                chunks_dir
                / hierarchical_chunks.SECTION_BATCH_OUTPUT_NAME
            )
            output_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "custom_id": job["custom_id"],
                            "response": {
                                "status_code": 200,
                                "body": {
                                    "output_text": json.dumps(
                                        {
                                            "summary": (
                                                f"Resume section "
                                                f"{job['section_index']}"
                                            )
                                        }
                                    )
                                },
                            },
                        }
                    )
                    for job in reversed(jobs)
                ),
                encoding="utf-8",
            )
            state = {
                "status": "completed",
                "output_file_id": "output-file",
            }
            with (
                patch("openai.OpenAI", return_value=object()),
                patch.object(
                    hierarchical_chunks,
                    "download_batch_files",
                ),
            ):
                hierarchical_chunks.finalize_section_summary_batch(
                    video,
                    jobs,
                    state,
                    details_per_section=3,
                )
            payload = json.loads(target.read_text(encoding="utf-8"))

        sections = [
            chunk
            for chunk in payload["chunks"]
            if chunk["chunk_level"] == "section"
        ]
        self.assertEqual(
            [section["content"] for section in sections],
            [
                "Resume section 1",
                "Resume section 2",
                "Resume section 3",
            ],
        )
        self.assertEqual(
            sections[0]["meta_data"]["detail_chunk_indexes"],
            [1, 2, 3],
        )

    def test_global_summary_can_be_finalized_from_batch_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "abcdefghijk"
            chunks_dir = video_dir / "outputs" / "chunks"
            chunks_dir.mkdir(parents=True)
            video = video_dir / "abcdefghijk.mp4"
            video.touch()
            target = chunks_dir / "transcript_chunks.json"
            target.write_text(
                json.dumps(
                    {
                        "chunking": {"profile": "long"},
                        "chunks": [
                            {
                                "chunk_index": 1,
                                "chunk_level": "section",
                                "content": "Resume section.",
                            },
                            {
                                "chunk_index": 1,
                                "chunk_level": "detail",
                                "content": "Detail.",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_path = (
                chunks_dir / hierarchical_chunks.GLOBAL_BATCH_OUTPUT_NAME
            )
            output_path.write_text(
                json.dumps(
                    {
                        "custom_id": "global-summary",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "output_text": json.dumps(
                                    {"summary": "Resume global batch."}
                                )
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            state = {
                "status": "completed",
                "output_file_id": "output-file",
            }
            with (
                patch("openai.OpenAI", return_value=object()),
                patch.object(
                    hierarchical_chunks,
                    "load_batch_state",
                    return_value=state,
                ),
                patch.object(
                    hierarchical_chunks,
                    "batch_state_matches",
                    return_value=True,
                ),
                patch.object(
                    hierarchical_chunks,
                    "poll_batch_state",
                    return_value=state,
                ),
                patch.object(
                    hierarchical_chunks,
                    "download_batch_files",
                ),
            ):
                hierarchical_chunks.summarize_video(
                    video,
                    mode="batch",
                )
            payload = json.loads(target.read_text(encoding="utf-8"))

        global_chunks = [
            chunk
            for chunk in payload["chunks"]
            if chunk["chunk_level"] == "global"
        ]
        self.assertEqual(global_chunks[0]["content"], "Resume global batch.")


if __name__ == "__main__":
    unittest.main()
