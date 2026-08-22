import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.support.environment import load_project_env


class ProjectEnvironmentTests(unittest.TestCase):
    def test_loads_the_selected_production_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / ".env.production").write_text(
                "PIPELINE_EXECUTION_BACKEND=scaleway\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"RAG_IONIS_ENV": "production"},
                clear=True,
            ):
                selected = load_project_env(root)

                self.assertEqual(selected, root / ".env.production")
                self.assertEqual(
                    os.environ["PIPELINE_EXECUTION_BACKEND"],
                    "scaleway",
                )

    def test_local_environment_falls_back_to_legacy_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / ".env").write_text(
                "PIPELINE_EXECUTION_BACKEND=local\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                selected = load_project_env(root)

                self.assertEqual(selected, root / ".env")
                self.assertEqual(
                    os.environ["PIPELINE_EXECUTION_BACKEND"],
                    "local",
                )

    def test_uses_ipv4_for_local_postgres_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / ".env.local").write_text(
                "DATABASE_URL=postgresql://user:password@localhost:5432/app\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with patch("pipeline.support.environment.os.name", "nt"):
                    load_project_env(root)

                self.assertEqual(
                    os.environ["DATABASE_URL"],
                    "postgresql://user:password@127.0.0.1:5432/app",
                )

    def test_rejects_unknown_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.dict(
                os.environ,
                {"RAG_IONIS_ENV": "staging"},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "local, production"):
                    load_project_env(Path(temporary_dir))


if __name__ == "__main__":
    unittest.main()
