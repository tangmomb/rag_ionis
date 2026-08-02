from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT_DIR / "docker-compose.yml"


class DockerConfigurationTests(unittest.TestCase):
    def test_postgres_uses_paris_timezone(self) -> None:
        compose = COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertIn("TZ: Europe/Paris", compose)
        self.assertIn('command: ["postgres", "-c", "timezone=Europe/Paris"]', compose)


if __name__ == "__main__":
    unittest.main()
