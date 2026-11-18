"""Historique public du recall et de la fidélité, lu dans monitoring.reports.

Sert les courbes de la FAQ. Ne renvoie que des agrégats (dates, scores, statut,
commit testé) : ni message, ni question, ni donnée d'utilisateur. Le résultat
est gardé en mémoire quelques minutes, la table ne changeant qu'une fois par jour.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

from fastapi import APIRouter, Response

router = APIRouter()

CACHE_SECONDS = 300
HISTORY_DAYS = 365
_cache: dict[str, Any] = {"at": 0.0, "value": None}
_lock = threading.Lock()


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _threshold(name: str, default: float) -> float:
    value = _number(os.getenv(name))
    return value if value is not None and 0 < value <= 1 else default


def history_from_rows(rows: list[tuple[Any, str, str, dict[str, Any]]]) -> dict[str, Any]:
    """Shape (created_at, kind, status, metrics) rows into the two chart series."""
    recall: list[dict[str, Any]] = []
    faithfulness: list[dict[str, Any]] = []
    for created_at, kind, status, metrics in rows:
        metrics = metrics or {}
        if kind in {"recall", "recall_gate"} and status != "failed":
            value = _number(metrics.get("video_recall"))
            if value is not None:
                tag = str(metrics.get("image_tag") or "")
                recall.append(
                    {
                        "at": created_at.isoformat(),
                        "kind": kind,
                        "status": status,
                        "video_recall": value,
                        "chunk_recall": _number(metrics.get("chunk_recall")),
                        "commit": tag[:7] or None,
                    }
                )
        elif kind == "quality" and status != "failed":
            canary = _number(metrics.get("canary_faithfulness"))
            if canary is not None:
                faithfulness.append(
                    {
                        "at": created_at.isoformat(),
                        "day": metrics.get("day"),
                        "status": status,
                        "canary_faithfulness": canary,
                        "canary_relevance": _number(metrics.get("canary_relevance")),
                        "traffic_faithfulness": _number(metrics.get("traffic_faithfulness")),
                        "traffic_judged": metrics.get("traffic_judged"),
                    }
                )
    return {
        "recall": recall,
        "faithfulness": faithfulness,
        "thresholds": {
            "recall": _threshold("GOLDEN_DATASET_MIN_RECALL", 0.80),
            "faithfulness": _threshold("QUALITY_MIN_FAITHFULNESS", 0.80),
        },
    }


def load_history() -> dict[str, Any]:
    from interface.backend.database import connect_database

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('monitoring.reports')")
            if cursor.fetchone()[0] is None:
                return history_from_rows([])
            cursor.execute(
                """
                SELECT created_at, kind, status, metrics
                FROM monitoring.reports
                WHERE kind IN ('recall', 'recall_gate', 'quality')
                  AND created_at >= now() - make_interval(days => %s)
                ORDER BY created_at
                """,
                (HISTORY_DAYS,),
            )
            return history_from_rows(cursor.fetchall())


@router.get("/api/monitoring/history")
def monitoring_history(response: Response) -> dict[str, Any]:
    # Public aggregates only: any site (the portfolio) may read them, without
    # credentials. The rest of the API keeps its strict CORS origins.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = f"public, max-age={CACHE_SECONDS}"
    with _lock:
        if _cache["value"] is not None and time.monotonic() - _cache["at"] < CACHE_SECONDS:
            return _cache["value"]
    try:
        value = load_history()
    except Exception as exc:
        print(f"[monitoring] history unavailable: {exc}", file=sys.stderr, flush=True)
        return {**history_from_rows([]), "error": "historique indisponible"}
    with _lock:
        _cache.update(at=time.monotonic(), value=value)
    return value
