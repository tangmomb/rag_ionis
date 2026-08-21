from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT_DIR / "docker-compose.yml"
PRODUCTION_COMPOSE_PATH = ROOT_DIR / "docker-compose.prod.yml"
API_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.api"
VPS_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.vps"
GPU_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.scaleway"
GPU_REQUIREMENTS_PATH = ROOT_DIR / "requirements-gpu.txt"
VPS_REQUIREMENTS_PATH = ROOT_DIR / "requirements-vps.txt"
WORKER_SCRIPT_PATH = ROOT_DIR / "deploy" / "rag-ionis-scaleway-worker.sh"
PUBLISH_SCRIPT_PATH = ROOT_DIR / "deploy" / "publish-scaleway-image.sh"
WORKER_ENV_EXAMPLE_PATH = ROOT_DIR / "deploy" / "scaleway-worker.env.example"
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

    def test_production_api_receives_only_explicit_environment(self) -> None:
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("env_file:", production_compose)
        self.assertIn("MISTRAL_API_KEY:", production_compose)
        self.assertNotIn("GOOGLE_API_KEY:", production_compose)
        self.assertNotIn("HUGGINGFACE_TOKEN:", production_compose)

    def test_vps_updater_is_cpu_only_and_opt_in(self) -> None:
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")
        dockerfile = VPS_DOCKERFILE_PATH.read_text(encoding="utf-8")
        requirements = VPS_REQUIREMENTS_PATH.read_text(encoding="utf-8")

        self.assertIn('profiles: ["jobs"]', production_compose)
        self.assertIn("dockerfile: Dockerfile.vps", production_compose)
        self.assertIn("USER app", dockerfile)
        self.assertNotIn("torch", requirements)
        self.assertNotIn("paddle", requirements)
        self.assertNotIn("fastapi", requirements)

    def test_gpu_worker_has_supported_node_and_persistent_model_cache(self) -> None:
        dockerfile = GPU_DOCKERFILE_PATH.read_text(encoding="utf-8")
        worker_script = WORKER_SCRIPT_PATH.read_text(encoding="utf-8")
        worker_env = WORKER_ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

        self.assertIn("node:22.18.0-bullseye-slim", dockerfile)
        self.assertNotIn("ffmpeg nodejs ca-certificates", dockerfile)
        self.assertIn('dst=/models', worker_script)
        self.assertIn("SCALEWAY_MODEL_CACHE_DIR=/var/lib/rag-ionis/models", worker_env)

    def test_scaleway_image_publisher_pushes_commit_and_latest_tags(self) -> None:
        publisher = PUBLISH_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('rev-parse --short=7 HEAD', publisher)
        self.assertIn('-t "${commit_image}"', publisher)
        self.assertIn('-t "${latest_image}"', publisher)
        self.assertIn('docker push "${commit_image}"', publisher)
        self.assertIn('docker push "${latest_image}"', publisher)

    def test_gpu_requirements_do_not_include_api_or_database_stack(self) -> None:
        requirements = GPU_REQUIREMENTS_PATH.read_text(encoding="utf-8")

        self.assertNotIn("fastapi", requirements)
        self.assertNotIn("uvicorn", requirements)
        self.assertNotIn("psycopg", requirements)
        self.assertNotIn("pgvector", requirements)
        self.assertNotIn("cohere", requirements)

    def test_analytics_password_is_not_hardcoded_in_sql(self) -> None:
        sql = ANALYTICS_SQL_PATH.read_text(encoding="utf-8")

        self.assertNotIn("rag_ionis_analytics_dev_password", sql)


if __name__ == "__main__":
    unittest.main()
