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
YTDLP_REQUIREMENTS_PATH = ROOT_DIR / "requirements-ytdlp.txt"
VPS_REQUIREMENTS_PATH = ROOT_DIR / "requirements-vps.txt"
WORKER_SCRIPT_PATH = ROOT_DIR / "deploy" / "rag-ionis-scaleway-worker.sh"
PUBLISH_SCRIPT_PATH = ROOT_DIR / "deploy" / "publish-scaleway-image.sh"
SHARED_PUBLISH_SCRIPT_PATH = ROOT_DIR / "deploy" / "publish-image.sh"
VPS_DEPLOY_SCRIPT_PATH = ROOT_DIR / "deploy" / "deploy-vps.sh"
SCALEWAY_DEPLOY_SCRIPT_PATH = ROOT_DIR / "deploy" / "deploy-scaleway-worker.sh"
VPS_WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "deploy-vps.yml"
SCALEWAY_WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "deploy-scaleway.yml"
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
        self.assertIn("GOOGLE_API_KEY:", production_compose)
        self.assertIn("OPENAI_API_KEY:", production_compose)
        self.assertIn("COHERE_API_KEY:", production_compose)
        self.assertNotIn("MISTRAL_API_KEY:", production_compose)
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

    def test_gpu_worker_has_supported_node_and_persistent_model_cache(self) -> None:
        dockerfile = GPU_DOCKERFILE_PATH.read_text(encoding="utf-8")
        worker_script = WORKER_SCRIPT_PATH.read_text(encoding="utf-8")
        worker_env = WORKER_ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

        self.assertIn("node:22.18.0-bullseye-slim", dockerfile)
        self.assertNotIn("ffmpeg nodejs ca-certificates", dockerfile)
        self.assertIn('dst=/models', worker_script)
        self.assertIn("SCALEWAY_MODEL_CACHE_DIR=/var/lib/rag-ionis/models", worker_env)

    def test_scaleway_image_publisher_uses_remote_build_cache(self) -> None:
        publisher = PUBLISH_SCRIPT_PATH.read_text(encoding="utf-8")
        shared_publisher = SHARED_PUBLISH_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('rev-parse --short=7 HEAD', publisher)
        self.assertIn('publish-image.sh" scaleway', publisher)
        self.assertIn("docker buildx build", shared_publisher)
        self.assertIn('--tag "${commit_image}"', shared_publisher)
        self.assertIn('--tag "${latest_image}"', shared_publisher)
        self.assertIn('--cache-from "type=registry,ref=${cache_image}"', shared_publisher)
        self.assertIn('--cache-to "type=registry,ref=${cache_image},mode=max"', shared_publisher)
        self.assertIn("--push", shared_publisher)

    def test_production_compose_uses_immutable_published_images(self) -> None:
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("build:", production_compose)
        self.assertIn("rag-ionis-api:${API_IMAGE_TAG", production_compose)
        self.assertIn("rag-ionis-updater:${UPDATER_IMAGE_TAG", production_compose)

    def test_deployment_scripts_support_healthcheck_and_rollback(self) -> None:
        vps_deploy = VPS_DEPLOY_SCRIPT_PATH.read_text(encoding="utf-8")
        scaleway_deploy = SCALEWAY_DEPLOY_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("docker pull", vps_deploy)
        self.assertIn("rolling back", vps_deploy)
        self.assertIn("State.Health", vps_deploy)
        self.assertIn("docker pull", scaleway_deploy)
        self.assertIn("--gpus all", scaleway_deploy)
        self.assertIn("rolling back", scaleway_deploy)
        self.assertIn("Waiting for the current GPU worker", scaleway_deploy)

    def test_deployment_workflows_are_separated_with_registry_cache(self) -> None:
        vps_workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")
        scaleway_workflow = SCALEWAY_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("Select affected images", vps_workflow)
        self.assertIn("pull_request:", vps_workflow)
        self.assertIn("deploy-vps:", vps_workflow)
        self.assertIn("actionlint@sha256:", vps_workflow)
        self.assertIn("docker/build-push-action@v7", vps_workflow)
        self.assertIn("cache-from: type=registry", vps_workflow)
        self.assertIn("name: Deploy Scaleway GPU worker", scaleway_workflow)
        self.assertIn("Dockerfile.scaleway", scaleway_workflow)
        self.assertIn("deploy-scaleway-worker.sh", scaleway_workflow)
        self.assertIn("cache-to: type=registry", scaleway_workflow)

    def test_gpu_requirements_do_not_include_api_or_database_stack(self) -> None:
        requirements = GPU_REQUIREMENTS_PATH.read_text(encoding="utf-8")

        self.assertNotIn("fastapi", requirements)
        self.assertNotIn("uvicorn", requirements)
        self.assertNotIn("psycopg", requirements)
        self.assertNotIn("pgvector", requirements)
        self.assertNotIn("cohere", requirements)

    def test_ytdlp_is_installed_after_expensive_gpu_dependencies(self) -> None:
        dockerfile = GPU_DOCKERFILE_PATH.read_text(encoding="utf-8")
        gpu_requirements = GPU_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        ytdlp_requirements = YTDLP_REQUIREMENTS_PATH.read_text(encoding="utf-8")

        self.assertNotIn("yt-dlp", gpu_requirements)
        self.assertIn("yt-dlp==", ytdlp_requirements)
        self.assertLess(
            dockerfile.index("COPY --from=paddle-builder /paddle-parts/libphi_gpu.so"),
            dockerfile.index("COPY requirements-ytdlp.txt ./"),
        )
        self.assertLess(
            dockerfile.index("COPY requirements-ytdlp.txt ./"),
            dockerfile.index("COPY pipeline ./pipeline"),
        )

    def test_analytics_password_is_not_hardcoded_in_sql(self) -> None:
        sql = ANALYTICS_SQL_PATH.read_text(encoding="utf-8")

        self.assertNotIn("rag_ionis_analytics_dev_password", sql)


if __name__ == "__main__":
    unittest.main()
