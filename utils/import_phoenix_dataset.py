from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1JtLiCc_nJ7mhCmA_AOgGuRQ0Eu4OFvT07KFsCWhu2DM/edit?gid=1961057140"
)
DEFAULT_PHOENIX_BASE_URL = "http://127.0.0.1:6006"
VALID_EXECUTION_ROUTES = {"sql_search", "vector_search"}
VALID_ACTIONS = {"answer", "clarify", "abstain"}


def _parse_json_list(value: Any, *, row_number: int, column: str) -> list[Any]:
    raw_value = str(value or "").strip()
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Ligne {row_number}: {column} doit etre une liste JSON."
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Ligne {row_number}: {column} doit etre une liste JSON.")
    return parsed


def _load_golden_dataset_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        question = str(row.get("question") or "").strip()
        route = str(row.get("expected_execution_route") or "").strip()
        youtube_video_ids = _parse_json_list(
            row.get("youtube_video_ids"),
            row_number=row_number,
            column="youtube_video_ids",
        )
        relevant_chunk_ids = _parse_json_list(
            row.get("relevant_chunk_ids"),
            row_number=row_number,
            column="relevant_chunk_ids",
        )
        if not question:
            raise ValueError(f"Ligne {row_number}: question vide.")
        if route not in VALID_EXECUTION_ROUTES:
            raise ValueError(
                f"Ligne {row_number}: expected_execution_route invalide: {route!r}."
            )
        if not all(isinstance(value, str) and value.strip() for value in youtube_video_ids):
            raise ValueError(
                f"Ligne {row_number}: youtube_video_ids doit contenir des textes."
            )
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in relevant_chunk_ids):
            raise ValueError(
                f"Ligne {row_number}: relevant_chunk_ids doit contenir des entiers."
            )
        examples.append(
            {
                "input": {"question": question},
                "output": {
                    "expected_execution_route": route,
                    "youtube_video_ids": youtube_video_ids,
                    "relevant_chunk_ids": relevant_chunk_ids,
                },
            }
        )
    return examples


def _load_legacy_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        question = str(row.get("question") or "").strip()
        action = str(row.get("expected_action") or "").strip()
        if not question:
            raise ValueError(f"Ligne {row_number}: question vide.")
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Ligne {row_number}: expected_action invalide: {action!r}."
            )
        examples.append(
            {
                "input": {"question": question},
                "output": {
                    "action": action,
                },
                "metadata": {
                    "difficulty": str(row.get("difficulty") or "").strip(),
                },
            }
        )
    return examples


def examples_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Le fichier CSV ne contient aucun exemple.")
    if "expected_execution_route" in rows[0]:
        examples = _load_golden_dataset_examples(rows)
    else:
        examples = _load_legacy_examples(rows)
    if not examples:
        raise ValueError("Le fichier CSV ne contient aucun exemple.")
    return examples


def load_examples(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return examples_from_rows(rows)


def google_sheet_csv_url(google_sheet_url: str) -> str:
    parsed = urlparse(google_sheet_url)
    parts = parsed.path.split("/")
    try:
        spreadsheet_id = parts[parts.index("d") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("URL Google Sheets invalide.") from exc
    gid = parse_qs(parsed.query).get("gid", [""])[0]
    suffix = f"&gid={gid}" if gid else ""
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
        f"{suffix}"
    )


def load_examples_from_google_sheet(google_sheet_url: str) -> list[dict[str, Any]]:
    with urlopen(google_sheet_csv_url(google_sheet_url), timeout=30) as response:
        csv_text = response.read().decode("utf-8-sig")
    return examples_from_rows(list(csv.DictReader(io.StringIO(csv_text))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Importe un golden dataset dans Phoenix.")
    parser.add_argument("--dataset", default="golden_dataset")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path)
    source.add_argument("--google-sheet-url", default=DEFAULT_GOOGLE_SHEET_URL)
    parser.add_argument(
        "--phoenix-base-url",
        default=DEFAULT_PHOENIX_BASE_URL,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from phoenix.client import Client

    examples = (
        load_examples(args.csv)
        if args.csv is not None
        else load_examples_from_google_sheet(args.google_sheet_url)
    )
    client = Client(base_url=args.phoenix_base_url.rstrip("/"))
    dataset = client.datasets.create_dataset(
        name=args.dataset,
        examples=examples,
        dataset_description=(
            "Golden dataset RAG : route, videos YouTube et chunks attendus."
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
