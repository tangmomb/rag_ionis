from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class StagingConfigurationTests(unittest.TestCase):
    def test_staging_is_a_separate_api_and_database(self) -> None:
        compose = (ROOT_DIR / "docker-compose.prod.yml").read_text(encoding="utf-8")
        caddyfile = (ROOT_DIR / "Caddyfile").read_text(encoding="utf-8")

        self.assertIn("staging-postgres:", compose)
        self.assertIn("staging-api:", compose)
        self.assertIn("staging_postgres_data", compose)
        self.assertIn("@staging-postgres", compose)
        self.assertIn("handle_path /staging/*", caddyfile)

    def test_staging_page_uses_staging_api_prefix(self) -> None:
        page = (ROOT_DIR / "interface" / "index.html").read_text(encoding="utf-8")

        self.assertIn('window.location.pathname.startsWith("/staging")', page)
        self.assertIn('"/staging/api"', page)


if __name__ == "__main__":
    unittest.main()
