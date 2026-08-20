import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interface.backend.environment import load_project_env


class ApiEnvironmentTests(unittest.TestCase):
    def test_allows_injected_production_environment_without_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.dict(
                os.environ,
                {
                    "RAG_IONIS_ENV": "production",
                    "RAG_IONIS_ENV_FILE_OPTIONAL": "1",
                    "PIPELINE_EXECUTION_BACKEND": "scaleway",
                },
                clear=True,
            ):
                selected = load_project_env(Path(temporary_dir))

                self.assertEqual(
                    selected,
                    Path(temporary_dir) / ".env.production",
                )


if __name__ == "__main__":
    unittest.main()
