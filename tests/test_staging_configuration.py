from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class StagingConfigurationTests(unittest.TestCase):
    def test_staging_is_a_separate_api_and_database(self) -> None:
        compose = (ROOT_DIR / "docker-compose.prod.yml").read_text(encoding="utf-8")

        self.assertIn("staging-postgres:", compose)
        self.assertIn("staging-api:", compose)
        self.assertIn("staging_postgres_data", compose)
        self.assertIn("@staging-postgres", compose)
        # The host Caddy strips /staging and proxies here.
        self.assertIn('"127.0.0.1:8007:8006"', compose)

    def test_staging_is_stopped_after_the_recall_gate(self) -> None:
        # Staging only serves the deploy gate; it must not keep RAM on the shared VPS.
        workflow = (ROOT_DIR / ".github" / "workflows" / "deploy-vps.yml").read_text(encoding="utf-8")

        self.assertIn("--profile staging stop staging-api staging-postgres", workflow)
        self.assertIn("trap stop_staging EXIT", workflow)

    def test_staging_page_uses_staging_api_prefix(self) -> None:
        page = (ROOT_DIR / "interface" / "index.html").read_text(encoding="utf-8")

        self.assertIn('window.location.pathname.startsWith("/staging")', page)
        self.assertIn('"/staging/api"', page)


if __name__ == "__main__":
    unittest.main()
