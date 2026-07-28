import tempfile
import unittest
from pathlib import Path

from utils.clean_init_generated import clean_generated_directories


class CleanInitGeneratedTests(unittest.TestCase):
    def create_video_directory(self, root: Path, video_id: str) -> Path:
        video_dir = root / video_id
        (video_dir / "metadata").mkdir(parents=True)
        (video_dir / "outputs" / "chunks").mkdir(parents=True)
        (video_dir / "metadata" / "video_manifest.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (video_dir / "outputs" / "chunks" / "chunk.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (video_dir / f"{video_id}.mp4").write_bytes(b"video")
        return video_dir

    def test_removes_only_generated_directories_from_each_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "init"
            first = self.create_video_directory(root, "video-1")
            second = self.create_video_directory(root, "video-2")
            unrelated = first / "notes"
            unrelated.mkdir()
            (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

            removed = clean_generated_directories(root)

            self.assertEqual(len(removed), 4)
            for video_dir in (first, second):
                self.assertFalse((video_dir / "metadata").exists())
                self.assertFalse((video_dir / "outputs").exists())
                self.assertTrue((video_dir / f"{video_dir.name}.mp4").is_file())
            self.assertTrue((unrelated / "keep.txt").is_file())

    def test_dry_run_preserves_generated_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "init"
            video_dir = self.create_video_directory(root, "video-1")

            targets = clean_generated_directories(root, dry_run=True)

            self.assertEqual(len(targets), 2)
            self.assertTrue((video_dir / "metadata").is_dir())
            self.assertTrue((video_dir / "outputs").is_dir())

    def test_missing_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"

            with self.assertRaises(FileNotFoundError):
                clean_generated_directories(missing)


if __name__ == "__main__":
    unittest.main()
