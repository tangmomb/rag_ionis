from __future__ import annotations

import ast
import unittest
from pathlib import Path


STEPS_DIR = Path(__file__).resolve().parents[1] / "pipeline" / "steps"


def is_dunder_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    values = [node.test.left, *node.test.comparators]
    has_name = any(isinstance(value, ast.Name) and value.id == "__name__" for value in values)
    has_main = any(
        isinstance(value, ast.Constant) and value.value == "__main__"
        for value in values
    )
    return has_name and has_main


class StepModulesNoCliTests(unittest.TestCase):
    def test_step_modules_do_not_embed_command_line_interfaces(self) -> None:
        step_files = sorted(
            path for path in STEPS_DIR.rglob("*.py") if path.name != "__init__.py"
        )
        self.assertEqual(len(step_files), 29)

        violations: list[str] = []
        for path in step_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative_path = path.relative_to(STEPS_DIR)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(
                    alias.name == "argparse" for alias in node.names
                ):
                    violations.append(f"{relative_path}:{node.lineno}: import argparse")
                elif isinstance(node, ast.ImportFrom) and node.module == "argparse":
                    violations.append(f"{relative_path}:{node.lineno}: from argparse")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    node.name in {"parse_args", "main"}
                ):
                    violations.append(f"{relative_path}:{node.lineno}: def {node.name}")
                elif isinstance(node, ast.Call):
                    callable_name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                        node.func.id if isinstance(node.func, ast.Name) else ""
                    )
                    if callable_name == "ArgumentParser":
                        violations.append(
                            f"{relative_path}:{node.lineno}: ArgumentParser"
                        )
                elif is_dunder_main_guard(node):
                    violations.append(f"{relative_path}:{node.lineno}: __main__ guard")

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
