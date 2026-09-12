from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT_DIR / "docker-compose.yml"
PRODUCTION_COMPOSE_PATH = ROOT_DIR / "docker-compose.prod.yml"
API_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.api"
VPS_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.vps"
VPS_REQUIREMENTS_PATH = ROOT_DIR / "requirements-vps.txt"
VPS_DEPLOY_SCRIPT_PATH = ROOT_DIR / "deploy" / "deploy-vps.sh"
VPS_WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "deploy-vps.yml"
ANALYTICS_SQL_PATH = ROOT_DIR / "docker" / "postgres" / "init" / "002_analytics_readonly.sql"


class VpsConfigurationTests(unittest.TestCase):
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

    def test_production_api_receives_only_explicit_environment(self) -> None:
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("env_file:", production_compose)
        self.assertIn("GOOGLE_API_KEY:", production_compose)
        self.assertIn("MISTRAL_API_KEY:", production_compose)
        self.assertIn("OPENAI_API_KEY:", production_compose)
        self.assertIn("COHERE_API_KEY:", production_compose)
        self.assertNotIn("HUGGINGFACE_TOKEN:", production_compose)

    def test_vps_updater_is_cpu_only_and_opt_in(self) -> None:
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")
        dockerfile = VPS_DOCKERFILE_PATH.read_text(encoding="utf-8")
        requirements = VPS_REQUIREMENTS_PATH.read_text(encoding="utf-8")

        self.assertIn('profiles: ["jobs"]', production_compose)
        self.assertIn("rag-ionis-updater:${UPDATER_IMAGE_TAG", production_compose)
        self.assertIn("USER app", dockerfile)
        self.assertNotIn("torch", requirements)
        self.assertNotIn("paddle", requirements)
        self.assertNotIn("fastapi", requirements)

    def test_production_compose_uses_immutable_published_images(self) -> None:
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("build:", production_compose)
        self.assertIn("rag-ionis-api:${API_IMAGE_TAG", production_compose)
        self.assertIn("rag-ionis-updater:${UPDATER_IMAGE_TAG", production_compose)

    def test_vps_deployment_script_supports_healthcheck_and_rollback(self) -> None:
        vps_deploy = VPS_DEPLOY_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("docker pull", vps_deploy)
        self.assertIn("rolling back", vps_deploy)
        self.assertIn("State.Health", vps_deploy)

    def test_vps_workflow_uses_registry_cache(self) -> None:
        vps_workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("Select affected images", vps_workflow)
        self.assertIn("pull_request:", vps_workflow)
        self.assertIn("deploy-vps:", vps_workflow)
        self.assertIn("actionlint@sha256:", vps_workflow)
        self.assertIn("docker/build-push-action@v7", vps_workflow)
        self.assertIn("cache-from: type=registry", vps_workflow)

    def test_analytics_password_is_not_hardcoded_in_sql(self) -> None:
        sql = ANALYTICS_SQL_PATH.read_text(encoding="utf-8")

        self.assertNotIn("rag_ionis_analytics_dev_password", sql)


if __name__ == "__main__":
    unittest.main()
