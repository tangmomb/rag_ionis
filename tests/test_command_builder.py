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

    def test_daily_youtube_sync_action_is_available(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        index = INDEX_PATH.read_text(encoding="utf-8")

        self.assertIn('id: "youtube-daily-sync"', source)
        self.assertIn('title: "Daily sync YouTube"', source)
        self.assertIn('module: "pipeline.update_runs"', source)
        daily_sync = source.split('id: "youtube-daily-sync"', 1)[1].split(
            'id: "download"', 1
        )[0]
        self.assertIn("sections: []", daily_sync)
        self.assertNotIn('flag: "--', daily_sync)
        self.assertIn("archive horodatée précédente", daily_sync)
        self.assertIn("dossier était absent", daily_sync)
        self.assertIn("daily_sync_log.json", daily_sync)
        self.assertIn('app.js?v=20260815-vps-tools', index)

    def test_vps_actions_include_combined_tunnels_and_safe_sql_sync(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('category: "VPS"', source)
        self.assertIn('id: "vps-tunnels"', source)
        self.assertIn(
            '-L 127.0.0.1:15432:127.0.0.1:5432 -L 127.0.0.1:16006:127.0.0.1:6006',
            source,
        )
        self.assertIn('id: "vps-phoenix"', source)
        self.assertIn('http://127.0.0.1:16006', source)
        self.assertIn('id: "vps-sync-db"', source)
        self.assertIn('PYTHON_DOTENV_DISABLED', source)
        self.assertIn('pipeline.publish.sync_database', source)
        self.assertIn('if (mode === "dry-run") syncArgs.push("--dry-run");', source)

if __name__ == "__main__":
    unittest.main()
