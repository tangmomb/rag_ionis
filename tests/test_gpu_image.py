from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 local environment.
    import tomli as tomllib


ROOT_DIR = Path(__file__).resolve().parents[1]
GPU_REQUIREMENTS_PATH = ROOT_DIR / "requirements-gpu.txt"
MODAL_APP_PATH = ROOT_DIR / "deploy" / "modal_app.py"
UV_LOCK_PATH = ROOT_DIR / "uv.lock"
PINNED_PACKAGE_PATTERN = re.compile(r"(?<![\w.-])([A-Za-z][\w.-]*)(?:\[[\w,-]+\])?==([\w.+!-]+)")


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions() -> dict[str, set[str]]:
    lock = tomllib.loads(UV_LOCK_PATH.read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        if "version" in package:
            versions.setdefault(normalize_package_name(package["name"]), set()).add(package["version"])
    return versions


class GpuImageTests(unittest.TestCase):
    def test_gpu_image_pins_match_uv_lock(self) -> None:
        # Modal builds the GPU image in hand-split layers: every version it pins
        # must be the one resolved in uv.lock, the single source of truth.
        locked = locked_versions()
        for source in (GPU_REQUIREMENTS_PATH, MODAL_APP_PATH):
            pins = PINNED_PACKAGE_PATTERN.findall(source.read_text(encoding="utf-8"))
            self.assertTrue(pins, source.name)
            for name, version in pins:
                with self.subTest(source=source.name, package=name):
                    self.assertIn(version, locked.get(normalize_package_name(name), set()))

    def test_gpu_requirements_do_not_include_api_or_database_stack(self) -> None:
        requirements = GPU_REQUIREMENTS_PATH.read_text(encoding="utf-8").lower()
        for package in ("fastapi", "uvicorn", "psycopg", "langgraph"):
            self.assertNotIn(package, requirements)


if __name__ == "__main__":
    unittest.main()
