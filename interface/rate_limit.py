"""Limitation du débit et budget quotidien des routes payantes du RAG.

Chaque question coûte des appels LLM (Mistral, Cohere) : sans limite, un script
pourrait vider le budget en quelques minutes. Trois compteurs, partagés par les
workers uvicorn via PostgreSQL :

- par IP et par minute (rafales) ;
- par IP et par jour (usage abusif) ;
- global par jour : le budget quotidien converti en nombre de questions à partir
  du coût moyen mesuré par les expériences Phoenix.

Les IP sont stockées hachées et les compteurs purgés après deux jours. La
protection n'est active qu'en production (RAG_IONIS_ENV=production) ; si la base
est indisponible, elle laisse passer la requête plutôt que de bloquer le site.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, time as day_time, timedelta, timezone
from ipaddress import ip_address
from typing import Any
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
# Questions, réponses en streaming et recherche élargie : les routes qui appellent un LLM.
PAID_PATH_PATTERN = re.compile(r"^/api/rag(?:/stream|/messages/\d+/expand)?/?$")

DEFAULT_PER_MINUTE = 10
DEFAULT_PER_DAY = 30
DEFAULT_VPS_PER_DAY = 100
DEFAULT_DAILY_BUDGET_USD = 2.0
# Coût moyen d'une question mesuré sur le golden dataset (0,0063 $), arrondi au-dessus.
DEFAULT_COST_PER_QUESTION_USD = 0.0065
PURGE_PROBABILITY = 0.01

SCHEMA_SQL = (
    "CREATE SCHEMA IF NOT EXISTS guard",
    """
    CREATE TABLE IF NOT EXISTS guard.request_counts (
        bucket TEXT NOT NULL,
        window_start TIMESTAMPTZ NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY (bucket, window_start)
    )
    """,
)


@dataclass(frozen=True)
class Limits:
    per_minute: int
    per_day: int
    global_per_day: int


@dataclass(frozen=True)
class Rejection:
    status_code: int
    detail: str
    retry_after_seconds: int


@dataclass(frozen=True)
class Windows:
    minute: datetime
    day: datetime
    next_minute_seconds: int
    next_day_seconds: int


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def enabled() -> bool:
    override = os.getenv("RAG_RATE_LIMIT_ENABLED", "").strip().lower()
    if override:
        return override in {"1", "true", "yes", "on"}
    return os.getenv("RAG_IONIS_ENV", "").strip().lower() == "production"


def limits_from_env(client_ip: str | None = None) -> Limits:
    budget = _env_number("RAG_DAILY_BUDGET_USD", DEFAULT_DAILY_BUDGET_USD)
    cost = _env_number("RAG_COST_PER_QUESTION_USD", DEFAULT_COST_PER_QUESTION_USD)
    per_day = int(_env_number("RAG_RATE_LIMIT_PER_DAY", DEFAULT_PER_DAY))
    vps_ip = os.getenv("RAG_RATE_LIMIT_VPS_IP", "").strip()
    if vps_ip and client_ip:
        try:
            is_vps = ip_address(client_ip) == ip_address(vps_ip)
        except ValueError:
            is_vps = False
        if is_vps:
            per_day = int(_env_number("RAG_RATE_LIMIT_VPS_PER_DAY", DEFAULT_VPS_PER_DAY))
    return Limits(
        per_minute=int(_env_number("RAG_RATE_LIMIT_PER_MINUTE", DEFAULT_PER_MINUTE)),
        per_day=per_day,
        global_per_day=max(1, math.floor(budget / cost)),
    )


def hashed_client(ip: str | None) -> str:
    salt = os.getenv("RAG_RATE_LIMIT_SALT", "rag-ionis")
    return hashlib.sha256(f"{salt}:{ip or 'unknown'}".encode()).hexdigest()[:32]


def windows_at(now: datetime) -> Windows:
    minute = now.replace(second=0, microsecond=0)
    local_day = now.astimezone(PARIS).date()
    day = datetime.combine(local_day, day_time.min, tzinfo=PARIS)
    next_day = datetime.combine(local_day + timedelta(days=1), day_time.min, tzinfo=PARIS)
    return Windows(
        minute=minute,
        day=day.astimezone(timezone.utc),
        next_minute_seconds=max(1, math.ceil((minute + timedelta(minutes=1) - now).total_seconds())),
        next_day_seconds=max(1, math.ceil((next_day - now).total_seconds())),
    )


def buckets_for(client: str, windows: Windows) -> dict[str, tuple[str, datetime]]:
    return {
        "minute": (f"ip-minute:{client}", windows.minute),
        "day": (f"ip-day:{client}", windows.day),
        "global": ("global-day", windows.day),
    }


def evaluate(counts: dict[str, int], limits: Limits, windows: Windows) -> Rejection | None:
    """Decide from the counters, already incremented for this request."""
    if counts.get("global", 0) > limits.global_per_day:
        return Rejection(
            503,
            "Le RAG a atteint son budget de questions pour aujourd'hui. Réessaie demain.",
            windows.next_day_seconds,
        )
    if counts.get("day", 0) > limits.per_day:
        return Rejection(
            429,
            f"Limite de {limits.per_day} questions par jour atteinte. Réessaie demain.",
            windows.next_day_seconds,
        )
    if counts.get("minute", 0) > limits.per_minute:
        return Rejection(
            429,
            f"Trop de questions d'affilée ({limits.per_minute} par minute). "
            "Patiente quelques secondes.",
            windows.next_minute_seconds,
        )
    return None


def ensure_schema(connection: Any) -> None:
    with connection.cursor() as cursor:
        for statement in SCHEMA_SQL:
            cursor.execute(statement)
    connection.commit()


def increment(connection: Any, buckets: dict[str, tuple[str, datetime]]) -> dict[str, int]:
    names = {bucket: name for name, (bucket, _window) in buckets.items()}
    values = [value for bucket, window in buckets.values() for value in (bucket, window)]
    placeholders = ", ".join(["(%s, %s, 1)"] * len(buckets))
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO guard.request_counts (bucket, window_start, count)
            VALUES {placeholders}
            ON CONFLICT (bucket, window_start)
            DO UPDATE SET count = guard.request_counts.count + 1
            RETURNING bucket, count
            """,
            values,
        )
        counts = {names[bucket]: int(count) for bucket, count in cursor.fetchall()}
        if random.random() < PURGE_PROBABILITY:
            cursor.execute(
                "DELETE FROM guard.request_counts WHERE window_start < now() - interval '2 days'"
            )
    connection.commit()
    return counts


_SCHEMA_READY = False


def check_request(client_ip: str | None, *, now: datetime | None = None) -> Rejection | None:
    """Count one paid request and return a rejection when a limit is exceeded."""
    global _SCHEMA_READY
    from interface.backend.database import connect_database

    now = now or datetime.now(timezone.utc)
    windows = windows_at(now)
    try:
        with connect_database() as connection:
            if not _SCHEMA_READY:
                ensure_schema(connection)
                _SCHEMA_READY = True
            counts = increment(connection, buckets_for(hashed_client(client_ip), windows))
    except Exception as exc:
        # Fail open: an unavailable counter must not take the site down.
        print(f"[rate_limit] counter unavailable, request allowed: {exc}", file=sys.stderr, flush=True)
        return None
    return evaluate(counts, limits_from_env(client_ip), windows)
