from __future__ import annotations

import re
import unittest
from pathlib import Path

from pipeline.catalog import TASKS


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "utils" / "command_builder" / "app.js"
INDEX_PATH = ROOT_DIR / "utils" / "command_builder" / "index.html"


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


if __name__ == "__main__":
    unittest.main()
