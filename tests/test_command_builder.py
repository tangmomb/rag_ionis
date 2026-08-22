from __future__ import annotations

import re
import unittest
from pathlib import Path

from pipeline.catalog import TASKS


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "utils" / "app_command_builder" / "app.js"
INDEX_PATH = ROOT_DIR / "utils" / "app_command_builder" / "index.html"


class CommandBuilderTests(unittest.TestCase):
    def test_task_selector_matches_pipeline_catalog(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        task_source = source.split("const selectorField", 1)[0]
        command_builder_tasks = re.findall(
            r'^\s*\["([^"]+)",\s*"[^"]+"\],?$',
            task_source,
            flags=re.MULTILINE,
        )

        self.assertEqual(command_builder_tasks, list(TASKS))

    def test_task_count_is_derived_from_task_definitions(self) -> None:
        index = INDEX_PATH.read_text(encoding="utf-8")
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('id="task-count"', index)
        self.assertIn(
            'document.querySelector("#task-count").textContent = String(TASKS.length);',
            source,
        )

    def test_test_action_uses_standard_library_unittest(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'fixedArgs: ["-m", "unittest", "discover"]',
            source,
        )
        self.assertNotIn('module: "pytest"', source)

    def test_run_action_exposes_global_batch_shortcut(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('const runBatchField = {', source)
        self.assertIn('flag: "--batch"', source)
        self.assertIn('if (action.id === "run")', source)
        self.assertIn(
            'executionFields.splice(openaiModeIndex, 1, runBatchField);',
            source,
        )

    def test_speakers_merge_action_exposes_database_review_options(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('id: "speakers-merge"', source)
        self.assertIn(
            'fixedArgs: ["-m", "pipeline", "speakers-merge"]',
            source,
        )
        self.assertIn('flag: "--max-distance"', source)
        self.assertIn('flag: "--dry-run"', source)

    def test_daily_youtube_sync_actions_are_available(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        index = INDEX_PATH.read_text(encoding="utf-8")

        self.assertIn('id: "youtube-update-stats"', source)
        self.assertIn('id: "youtube-update-videos"', source)
        self.assertIn(
            'fixedArgs: ["-m", "pipeline.update_stats"]',
            source,
        )
        self.assertIn(
            'fixedArgs: ["-m", "pipeline.update_videos"]',
            source,
        )
        daily_sync = source.split('id: "youtube-update-stats"', 1)[1].split(
            'id: "download"', 1
        )[0]
        self.assertIn("sections: []", daily_sync)
        self.assertNotIn('flag: "--', daily_sync)
        self.assertIn("API YouTube", daily_sync)
        self.assertIn("snapshots de stats", daily_sync)
        self.assertIn('app.js?v=20260822-vps-update-stats', index)

    def test_sql_backup_action_uses_a_binary_dump_without_powershell_redirection(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('id: "backup-sql"', source)
        self.assertIn('commandBuilder: buildSqlBackupCommand', source)
        self.assertIn('pg_dump -U rag_ionis -d rag_ionis -Fc -f /tmp/$backupFile', source)
        self.assertIn('docker compose cp "postgres:/tmp/$backupFile" "utils/backup_sql/$backupFile"', source)
        self.assertIn('pg_restore -l "/tmp/$backupFile"', source)
        self.assertNotIn('pg_dump -U rag_ionis -d rag_ionis -Fc >', source)

    def test_vps_actions_keep_only_the_requested_operations(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('category: "VPS"', source)
        for action_id in (
            "vps-connect",
            "vps-phoenix",
            "vps-browse-postgres",
            "vps-restore-postgres",
            "vps-pull-code",
            "vps-rebuild-stack",
            "vps-run-update-stats",
            "vps-run-update-videos",
            "vps-send-production-env",
        ):
            self.assertIn(f'id: "{action_id}"', source)
        self.assertIn('git pull --ff-only', source)
        pull_code = source.split('id: "vps-pull-code"', 1)[1].split(
            'id: "vps-rebuild-stack"', 1
        )[0]
        self.assertIn('shellCommand: "cd rag_ionis/ && git pull --ff-only"', pull_code)
        self.assertIn('shellLabel: "Commande Linux VPS"', pull_code)
        self.assertIn('terminal: { label: "Linux · VPS", prompt: "ubuntu@vps:~$" }', pull_code)
        self.assertNotIn('vpsRemoteCommand', pull_code)
        rebuild_stack = source.split('id: "vps-rebuild-stack"', 1)[1].split(
            'id: "vps-send-production-env"', 1
        )[0]
        self.assertIn(
            'shellCommand: "cd rag_ionis/ && docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production up -d --no-build"',
            rebuild_stack,
        )
        self.assertIn('shellLabel: "Commande Linux VPS"', rebuild_stack)
        self.assertNotIn('vpsRemoteCommand', rebuild_stack)
        update_stats = source.split('id: "vps-run-update-stats"', 1)[1].split(
            'id: "vps-send-production-env"', 1
        )[0]
        self.assertIn('--profile jobs run --rm --no-build updater', update_stats)
        self.assertIn('python -m pipeline.update_stats', update_stats)
        self.assertIn('shellLabel: "Commande Linux VPS"', update_stats)
        update_videos = source.split('id: "vps-run-update-videos"', 1)[1].split(
            'id: "vps-send-production-env"', 1
        )[0]
        self.assertIn('--profile jobs run --rm --no-build updater', update_videos)
        self.assertIn('python -m pipeline.update_videos', update_videos)
        self.assertIn('shellLabel: "Commande Linux VPS"', update_videos)
        self.assertIn('docker-compose.prod.yml', source)
        self.assertIn('scp -i', source)
        self.assertIn('ubuntu@179.237.98.117:rag_ionis/.env.production', source)
        self.assertNotIn('ubuntu@179.237.98.117:/srv/rag_ionis/.env.production', source)
        self.assertIn('.env.production', source)

        self.assertIn('docker exec -it rag_ionis_postgres psql -U rag_ionis -d rag_ionis', source)
        self.assertIn('command: "\\\\dt data.*"', source)
        self.assertIn("FROM data.video_speakers", source)
        self.assertIn('command: "\\\\q"', source)
        self.assertIn('utils\\\\backup_sql\\\\rag_ionis_20260822_001644.dump', source)
        self.assertIn('pg_restore -U rag_ionis -d rag_ionis --clean --if-exists /tmp/local.dump', source)

        for removed_action_id in (
            "vps-tunnels",
            "vps-list-hidden",
            "vps-start-stack",
            "vps-logs",
            "vps-update-stats",
            "vps-update-videos",
            "vps-stop-stack",
            "vps-sync-db",
            "vps-clean-session",
        ):
            self.assertNotIn(f'id: "{removed_action_id}"', source)

        phoenix = source.split('id: "vps-phoenix"', 1)[1].split(
            'id: "vps-browse-postgres"', 1
        )[0]
        self.assertIn('-L 6007:127.0.0.1:6006', phoenix)
        self.assertIn('http://127.0.0.1:6007/projects', phoenix)

    def test_scaleway_actions_include_image_lifecycle_and_worker_info(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        index = INDEX_PATH.read_text(encoding="utf-8")

        self.assertIn('"Scaleway"', source)
        self.assertIn('id: "scaleway-build-image"', source)
        self.assertIn('rg.fr-par.scw.cloud/rag-ionis/rag-ionis-scaleway', source)
        self.assertNotIn('rg.fr-par.scw.cloud/NAMESPACE/rag-ionis-scaleway', source)
        self.assertIn('id: "scaleway-publish-image"', source)
        self.assertIn('$env:SCALEWAY_IMAGE_REPOSITORY = $repository', source)
        self.assertIn('bash deploy/publish-scaleway-image.sh $commitTag', source)
        self.assertIn("cache distant", source)
        self.assertIn('id: "scaleway-inspect-image"', source)
        self.assertIn('docker buildx imagetools inspect', source)
        self.assertIn('id: "scaleway-pull-latest"', source)
        self.assertIn('id: "scaleway-connect"', source)
        self.assertIn('title: "Se connecter à la VM Scaleway"', source)
        self.assertIn('command: `ssh${identityOption}', source)
        self.assertIn('id: "scaleway-worker-info"', source)
        worker_info = source.split('id: "scaleway-worker-info"', 1)[1].split(
            '];', 1
        )[0]
        self.assertIn('fields: scalewayWorkerConnectionFields', worker_info)
        self.assertIn('id: "scalewayWorkerHost"', source)
        self.assertIn(r'C:\Users\rgb\.ssh\rag_ionis_scaleway_admin', source)
        self.assertIn('value: "51.159.135.156"', source)
        self.assertIn('value: "root"', source)
        self.assertNotIn('ubuntu@179.237.98.117', worker_info)
        self.assertIn('sudo systemctl cat rag-ionis-scaleway-worker', source)
        self.assertIn('/etc/rag-ionis/scaleway-worker.env', source)
        self.assertIn('sudo cat /etc/rag-ionis/scaleway-worker.env', source)
        self.assertIn('clés S3 et API en clair', source)
        self.assertIn('app.js?v=20260822-vps-update-stats', index)

if __name__ == "__main__":
    unittest.main()
