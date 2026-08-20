from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT_DIR / "docker-compose.yml"
PRODUCTION_COMPOSE_PATH = ROOT_DIR / "docker-compose.prod.yml"
API_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.api"
ANALYTICS_SQL_PATH = ROOT_DIR / "docker" / "postgres" / "init" / "002_analytics_readonly.sql"


class DockerConfigurationTests(unittest.TestCase):
    def test_postgres_uses_paris_timezone(self) -> None:
        compose = COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertIn("TZ: Europe/Paris", compose)
        self.assertIn('command: ["postgres", "-c", "timezone=Europe/Paris"]', compose)

    def test_production_services_are_not_directly_public(self) -> None:
        common_compose = COMPOSE_PATH.read_text(encoding="utf-8")
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertIn('"127.0.0.1:${POSTGRES_PORT:-5432}:5432"', common_compose)
        self.assertIn('"127.0.0.1:${PHOENIX_PORT:-6006}:6006"', common_compose)
        self.assertIn('expose:\n      - "8006"', production_compose)
        self.assertNotIn('"8006:8006"', production_compose)

    def test_production_api_runs_without_reload_as_non_root(self) -> None:
        dockerfile = API_DOCKERFILE_PATH.read_text(encoding="utf-8")

        self.assertIn("USER app", dockerfile)
        self.assertIn('"--workers", "2"', dockerfile)
        self.assertNotIn('"--reload"', dockerfile)

    def test_analytics_password_is_not_hardcoded_in_sql(self) -> None:
        sql = ANALYTICS_SQL_PATH.read_text(encoding="utf-8")

        self.assertNotIn("rag_ionis_analytics_dev_password", sql)


if __name__ == "__main__":
    unittest.main()
