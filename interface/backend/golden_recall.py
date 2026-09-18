"""Quality gate Golden Dataset executed in the isolated staging API."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from interface.backend.api import run_rag
from interface.backend.schemas import RagRequest


DEFAULT_GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1JtLiCc_nJ7mhCmA_AOgGuRQ0Eu4OFvT07KFsCWhu2DM/edit?gid=1961057140"
)
DEFAULT_MIN_RECALL = 0.85


@dataclass(frozen=True)
class GoldenCase:
    question: str
    youtube_video_ids: frozenset[str]
    relevant_chunk_ids: frozenset[int]


def google_sheet_csv_url(google_sheet_url: str) -> str:
    parsed = urlparse(google_sheet_url)
    parts = parsed.path.split("/")
    try:
        spreadsheet_id = parts[parts.index("d") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("URL Google Sheets invalide.") from exc
    gid = parse_qs(parsed.query).get("gid", [""])[0]
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
    return f"{url}&gid={gid}" if gid else url


def _json_set(value: str, *, field: str, row_number: int, item_type: type[str] | type[int]) -> frozenset[Any]:
    try:
        values = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ligne {row_number}: {field} doit etre une liste JSON.") from exc
    if not isinstance(values, list) or not all(type(item) is item_type for item in values):
        raise ValueError(f"Ligne {row_number}: {field} est invalide.")
    return frozenset(values)


def parse_golden_cases(csv_text: str) -> list[GoldenCase]:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows:
        raise ValueError("Le Golden Dataset ne contient aucun exemple.")
    required_columns = {"question", "youtube_video_ids", "relevant_chunk_ids"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValueError(f"Colonnes Golden Dataset absentes: {', '.join(sorted(missing_columns))}.")
    cases: list[GoldenCase] = []
    for row_number, row in enumerate(rows, start=2):
        question = (row.get("question") or "").strip()
        if not question:
            raise ValueError(f"Ligne {row_number}: question vide.")
        cases.append(
            GoldenCase(
                question=question,
                youtube_video_ids=_json_set(row.get("youtube_video_ids", ""), field="youtube_video_ids", row_number=row_number, item_type=str),
                relevant_chunk_ids=_json_set(row.get("relevant_chunk_ids", ""), field="relevant_chunk_ids", row_number=row_number, item_type=int),
            )
        )
    return cases


def load_golden_cases(google_sheet_url: str) -> list[GoldenCase]:
    with urlopen(google_sheet_csv_url(google_sheet_url), timeout=30) as response:
        return parse_golden_cases(response.read().decode("utf-8-sig"))


def youtube_video_id(video_url: str | None) -> str | None:
    if not video_url:
        return None
    parsed = urlparse(video_url)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/") or None
    return parse_qs(parsed.query).get("v", [None])[0]


def recall(expected: frozenset[Any], actual: Iterable[Any]) -> tuple[int, int]:
    if not expected:
        return 0, 0
    return len(expected & frozenset(actual)), len(expected)


def source_value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def evaluate(cases: Iterable[GoldenCase]) -> dict[str, Any]:
    cases = list(cases)
    video_matches = video_expected = chunk_matches = chunk_expected = 0
    video_case_recalls: list[float] = []
    chunk_case_recalls: list[float] = []
    failures: list[str] = []
    for case in cases:
        try:
            response = run_rag(RagRequest(question=case.question))
        except Exception as exc:
            failures.append(f"{case.question!r}: {exc}")
            continue
        # This is the exact source set evaluated by Phoenix.  ``sources`` only
        # contains citations selected after answer generation and must not be
        # used to assess retrieval quality.
        sources = response.retrieval.get("retrieved_sources") or []
        actual_videos = (
            youtube_video_id(source_value(source, "video_url")) for source in sources
        )
        matched, expected = recall(case.youtube_video_ids, filter(None, actual_videos))
        video_matches += matched
        video_expected += expected
        if expected:
            video_case_recalls.append(matched / expected)
        matched, expected = recall(
            case.relevant_chunk_ids,
            (source_value(source, "chunk_id") for source in sources),
        )
        chunk_matches += matched
        chunk_expected += expected
        if expected:
            chunk_case_recalls.append(matched / expected)
    return {
        "case_count": len(cases),
        "failed_cases": failures,
        "video_recall": (
            sum(video_case_recalls) / len(video_case_recalls)
            if video_case_recalls
            else None
        ),
        "video_recall_micro": video_matches / video_expected if video_expected else None,
        "video_matches": video_matches,
        "video_expected": video_expected,
        "chunk_recall": (
            sum(chunk_case_recalls) / len(chunk_case_recalls)
            if chunk_case_recalls
            else None
        ),
        "chunk_recall_micro": chunk_matches / chunk_expected if chunk_expected else None,
        "chunk_matches": chunk_matches,
        "chunk_expected": chunk_expected,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--google-sheet-url", default=os.getenv("GOLDEN_DATASET_URL", DEFAULT_GOOGLE_SHEET_URL))
    parser.add_argument(
        "--min-recall",
        type=float,
        default=float(os.getenv("GOLDEN_DATASET_MIN_RECALL", DEFAULT_MIN_RECALL)),
        help="Seuil bloquant du recall global des vidéos attendues.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.min_recall <= 1:
        raise SystemExit("Le seuil de recall doit etre compris entre 0 et 1.")
    result = evaluate(load_golden_cases(args.google_sheet_url))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["failed_cases"]:
        print("Le test Golden Dataset a rencontre des erreurs de requete.", file=sys.stderr)
        return 1
    if result["video_recall"] is not None and result["video_recall"] < args.min_recall:
        print("Seuil de recall global non atteint.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
