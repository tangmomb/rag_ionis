from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 local environment.
    import tomli as tomllib


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT_DIR / "docker-compose.yml"
PRODUCTION_COMPOSE_PATH = ROOT_DIR / "docker-compose.prod.yml"
API_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.api"
VPS_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.vps"
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
VPS_DEPLOY_SCRIPT_PATH = ROOT_DIR / "deploy" / "deploy-vps.sh"
SCHEDULED_UPDATE_SCRIPT_PATH = ROOT_DIR / "deploy" / "run-scheduled-update.sh"
SYSTEMD_INSTALL_SCRIPT_PATH = ROOT_DIR / "deploy" / "install-systemd-timers.sh"
SYSTEMD_DIR = ROOT_DIR / "deploy" / "systemd"
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
        self.assertIn('"127.0.0.1:8006:8006"', production_compose)
        self.assertIn('"127.0.0.1:8007:8006"', production_compose)
        self.assertNotIn('"8006:8006"', production_compose)
        self.assertNotIn('"80:80"', production_compose)

    def test_production_services_have_memory_limits(self) -> None:
        # The VPS is shared with the agent_mots_cles project: every service, the updater
        # job included, is capped, with no extra swap and a lower OOM priority.
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertEqual(production_compose.count("    mem_limit:"), 8)
        self.assertEqual(production_compose.count("    memswap_limit:"), 8)
        self.assertEqual(production_compose.count("    oom_score_adj:"), 8)
        # Production API and PostgreSQL win over the agent project under memory pressure.
        self.assertEqual(production_compose.count("    mem_reservation:"), 2)
        self.assertEqual(production_compose.count("    oom_score_adj: -500"), 2)

    def test_youtube_proxy_relay_never_listens_publicly(self) -> None:
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")
        relay = production_compose.split("\n  yt-proxy:\n", 1)[1].split("\n\n", 1)[0]

        self.assertIn("bind=172.17.0.1", relay)
        self.assertIn("TCP:127.0.0.1:1080", relay)
        self.assertIn("alpine/socat:1.8.0.1@sha256:", relay)

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
        pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        requirements = "\n".join(pyproject["dependency-groups"]["vps"])

        self.assertIn('profiles: ["jobs"]', production_compose)
        self.assertIn("rag-ionis-updater:${UPDATER_IMAGE_TAG", production_compose)
        self.assertIn("USER app", dockerfile)
        self.assertIn("--no-default-groups --group vps", dockerfile)
        self.assertNotIn("torch", requirements)
        self.assertNotIn("paddle", requirements)
        self.assertNotIn("fastapi", requirements)

    def test_vps_updater_starts_without_ytdlp(self) -> None:
        # yt-dlp is only imported to download a video: the rest of the updater
        # must start without it. A fresh interpreter keeps the import chain
        # honest even where it is installed.
        script = "import sys; sys.modules['yt_dlp'] = None; import pipeline.update_runs"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

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
        self.assertNotIn("caddy", vps_deploy)

    def test_vps_workflow_uses_registry_cache(self) -> None:
        vps_workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("Select affected images", vps_workflow)
        self.assertIn("pull_request:", vps_workflow)
        self.assertIn("deploy-vps:", vps_workflow)
        self.assertIn("actionlint@sha256:", vps_workflow)
        self.assertIn("docker/build-push-action@v7", vps_workflow)
        self.assertIn("cache-from: type=registry", vps_workflow)

    def test_scheduled_updates_use_ephemeral_locked_containers(self) -> None:
        update_script = SCHEDULED_UPDATE_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("flock 9", update_script)
        self.assertIn("--profile jobs run --rm updater", update_script)
        self.assertIn("pipeline.update_stats", update_script)
        self.assertIn("pipeline.update_videos", update_script)
        self.assertIn("3 * 24 * 60 * 60", update_script)

    def test_systemd_timers_are_persistent_and_installed_by_deployment(self) -> None:
        stats_timer = (SYSTEMD_DIR / "rag-ionis-stats.timer").read_text(encoding="utf-8")
        videos_timer = (SYSTEMD_DIR / "rag-ionis-videos.timer").read_text(encoding="utf-8")
        service = (SYSTEMD_DIR / "rag-ionis-update@.service").read_text(encoding="utf-8")
        installer = SYSTEMD_INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")
        workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("Persistent=true", stats_timer)
        self.assertIn("Unit=rag-ionis-update@stats.service", stats_timer)
        self.assertIn("OnCalendar=Sun", videos_timer)
        self.assertIn("Persistent=true", videos_timer)
        self.assertIn("Unit=rag-ionis-update@videos.service", videos_timer)
        self.assertIn("run-scheduled-update.sh %i", service)
        self.assertIn("systemctl enable --now", installer)
        self.assertIn("install-systemd-timers.sh", workflow)

    def test_scheduled_jobs_send_telegram_alert_on_failure(self) -> None:
        service = (SYSTEMD_DIR / "rag-ionis-update@.service").read_text(encoding="utf-8")
        notify_unit = (SYSTEMD_DIR / "rag-ionis-notify-failure@.service").read_text(encoding="utf-8")
        notify_script = (SYSTEMD_DIR / "rag-ionis-notify-failure.sh").read_text(encoding="utf-8")
        telegram_script = (SYSTEMD_DIR / "rag-ionis-telegram.sh").read_text(encoding="utf-8")
        installer = SYSTEMD_INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("OnFailure=rag-ionis-notify-failure@%n.service", service)
        self.assertIn("rag-ionis-notify-failure.sh %i", notify_unit)
        self.assertIn("rag-ionis-telegram.sh", notify_script)
        # Le token ne doit jamais apparaître sur la ligne de commande (ps).
        self.assertIn("--config -", telegram_script)
        self.assertIn("/etc/rag-ionis/telegram.env", telegram_script)
        self.assertIn("rag-ionis-notify-failure@.service", installer)
        self.assertIn("rag-ionis-telegram.sh", installer)
        self.assertIn("rag-ionis-backup.service.d/notify-failure.conf", installer)

    def test_quality_monitoring_is_scheduled_and_deployed(self) -> None:
        monitor_unit = (SYSTEMD_DIR / "rag-ionis-monitor@.service").read_text(encoding="utf-8")
        quality_timer = (SYSTEMD_DIR / "rag-ionis-quality.timer").read_text(encoding="utf-8")
        recall_timer = (SYSTEMD_DIR / "rag-ionis-recall.timer").read_text(encoding="utf-8")
        runner = (ROOT_DIR / "deploy" / "run-monitoring.sh").read_text(encoding="utf-8")
        installer = SYSTEMD_INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")
        workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("OnFailure=rag-ionis-notify-failure@%n.service", monitor_unit)
        self.assertIn("run-monitoring.sh %i", monitor_unit)
        self.assertIn("Unit=rag-ionis-monitor@quality.service", quality_timer)
        self.assertIn("Persistent=true", quality_timer)
        self.assertIn("OnCalendar=*-*-01,15", recall_timer)
        self.assertIn("Unit=rag-ionis-monitor@recall.service", recall_timer)
        # The canary must never trace into the production Phoenix project, and
        # the recall measurement must not trace at all.
        self.assertIn("PHOENIX_PROJECT_NAME=rag-ionis-canary", runner)
        self.assertIn("PHOENIX_ENABLED=false", runner)
        self.assertIn("run --rm --no-deps", runner)
        restore_timer = (SYSTEMD_DIR / "rag-ionis-restore.timer").read_text(encoding="utf-8")
        self.assertIn("Unit=rag-ionis-monitor@restore.service", restore_timer)
        # The restore test never touches the production database.
        self.assertIn("rag-ionis-restore-test", runner)
        self.assertIn("pg_restore --no-owner --no-privileges", runner)
        for timer in ("rag-ionis-quality.timer", "rag-ionis-recall.timer", "rag-ionis-restore.timer"):
            self.assertIn(timer, installer)
        self.assertIn("rag-ionis-monitor@.service", installer)
        self.assertGreaterEqual(workflow.count("deploy/run-monitoring.sh"), 4)

    def test_logs_are_shipped_to_grafana_cloud_by_alloy(self) -> None:
        compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")
        config = (ROOT_DIR / "deploy" / "alloy" / "config.alloy").read_text(encoding="utf-8")
        workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        alloy = compose.split("\n  alloy:\n", 1)[1].split("\nvolumes:\n", 1)[0]
        # Pinned image, opt-in profile, read-only host access, capped memory.
        self.assertRegex(alloy, r"grafana/alloy:v[\d.]+@sha256:[0-9a-f]{64}")
        self.assertIn('profiles: ["observability"]', alloy)
        self.assertIn("/var/run/docker.sock:/var/run/docker.sock:ro", alloy)
        self.assertIn("/var/log/journal:/var/log/journal:ro", alloy)
        self.assertIn("mem_limit: 192m", alloy)
        # Credentials come from the environment, never from the config file.
        self.assertIn('sys.env("GCLOUD_RW_API_KEY")', config)
        self.assertNotIn("glc_", config)
        self.assertIn("com.docker.compose.project=rag_ionis", config)
        self.assertIn('regex         = "rag-ionis-.*"', config)
        # Started only once credentials exist, and recreated when the config changes.
        self.assertIn("grep -q '^GCLOUD_HOSTED_LOGS_URL=.' .env.production", workflow)
        self.assertIn("ALLOY_CONFIG_SHA", workflow)
        self.assertIn("scp deploy/alloy/config.alloy", workflow)

    def test_every_monitoring_report_is_recorded(self) -> None:
        recorder = (ROOT_DIR / "deploy" / "record-report.sh").read_text(encoding="utf-8")
        runner = (ROOT_DIR / "deploy" / "run-monitoring.sh").read_text(encoding="utf-8")
        workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS monitoring.reports", recorder)
        # Values reach SQL only as psql variables, never concatenated.
        self.assertIn(":'metrics'::jsonb", recorder)
        for kind in ("quality", "recall", "restore"):
            self.assertIn(f"publish {kind} ", runner)
        self.assertIn("record_failure quality", runner)
        self.assertIn("record-report.sh recall_gate", workflow)
        self.assertGreaterEqual(workflow.count("deploy/record-report.sh"), 4)

    def test_vps_workflow_tests_candidate_before_deployment(self) -> None:
        vps_workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")
        production_compose = PRODUCTION_COMPOSE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "Refresh staging corpus and verify candidate API recall",
            vps_workflow,
        )
        self.assertIn("needs.changes.outputs.recall_gate == 'true'", vps_workflow)
        self.assertIn("staging-postgres", vps_workflow)
        self.assertIn("STAGING_API_IMAGE_TAG", vps_workflow)
        self.assertIn("Staging corpus copied", vps_workflow)
        self.assertIn("Golden Dataset loaded", vps_workflow)
        self.assertIn("exec -T staging-postgres sh -c", vps_workflow)
        self.assertIn("TRUNCATE data.videos, data.speakers", vps_workflow)
        self.assertIn("ServerAliveInterval=30", vps_workflow)
        self.assertIn("ServerAliveCountMax=60", vps_workflow)
        self.assertIn("< /dev/null", vps_workflow)
        self.assertIn("python -m interface.backend.golden_recall", vps_workflow)
        self.assertIn("GOLDEN_DATASET_MIN_RECALL", production_compose)
        self.assertIn('--baseline-recall "${baseline}"', vps_workflow)
        self.assertIn('mv ".golden-recall-candidate-$3" .golden-recall-baseline', vps_workflow)

    def test_recall_gate_skips_infra_only_changes(self) -> None:
        vps_workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        for pattern in (
            "                docker-compose.prod.yml)",
            "                .github/workflows/deploy-vps.yml)",
            "                .github/workflows/release-images.yml|",
        ):
            start = vps_workflow.index(pattern)
            branch = vps_workflow[start:vps_workflow.index(";;", start)]
            self.assertNotIn("recall_gate=true", branch, pattern)

    def test_recall_gate_skips_front_end_only_changes(self) -> None:
        vps_workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        backend_case = vps_workflow.index("interface/backend/*|interface/connaissances_*)")
        static_case = vps_workflow.index("                interface/*)")
        # Bash case uses the first matching pattern: backend must come first.
        self.assertLess(backend_case, static_case)
        static_branch = vps_workflow[static_case:vps_workflow.index(";;", static_case)]
        self.assertIn("api=true", static_branch)
        self.assertNotIn("recall_gate=true", static_branch)

    def test_production_compose_changes_rebuild_the_api_candidate(self) -> None:
        vps_workflow = VPS_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("docker-compose.prod.yml)", vps_workflow)
        self.assertIn("api=true", vps_workflow)

    def test_analytics_password_is_not_hardcoded_in_sql(self) -> None:
        sql = ANALYTICS_SQL_PATH.read_text(encoding="utf-8")

        self.assertNotIn("rag_ionis_analytics_dev_password", sql)


if __name__ == "__main__":
    unittest.main()
