from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_DIR / "utils" / "cases_phoenix.csv"
DEFAULT_PHOENIX_BASE_URL = "http://127.0.0.1:6006"
VALID_ACTIONS = {"answer", "clarify", "abstain"}
VALID_SHADOW_STATUSES = {
    "acceptable",
    "bad_retrieval",
    "insufficient_sources",
    "unsupported_answer",
    "ambiguous_question",
}


def load_examples(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    examples: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        question = str(row.get("question") or "").strip()
        action = str(row.get("expected_action") or "").strip()
        shadow_status = str(row.get("shadow_status") or "").strip()
        if not question:
            raise ValueError(f"Ligne {row_number}: question vide.")
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Ligne {row_number}: expected_action invalide: {action!r}."
            )
        if shadow_status not in VALID_SHADOW_STATUSES:
            raise ValueError(
                f"Ligne {row_number}: shadow_status invalide: {shadow_status!r}."
            )
        examples.append(
            {
                "input": {"question": question},
                "output": {
                    "action": action,
                    "shadow_status": shadow_status,
                },
                "metadata": {
                    "difficulty": str(row.get("difficulty") or "").strip(),
                },
            }
        )
    if not examples:
        raise ValueError("Le fichier CSV ne contient aucun exemple.")
    return examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Importe le dataset RAG labellise dans Phoenix."
    )
    parser.add_argument("--dataset", default="cases_phoenix")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--phoenix-base-url",
        default=DEFAULT_PHOENIX_BASE_URL,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from phoenix.client import Client

    examples = load_examples(args.csv)
    client = Client(base_url=args.phoenix_base_url.rstrip("/"))
    dataset = client.datasets.create_dataset(
        name=args.dataset,
        examples=examples,
        dataset_description=(
            "Cas RAG locaux avec action et statut shadow attendus."
        ),
    )
    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "dataset_version_id": dataset.version_id,
        "example_count": len(dataset.examples),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
