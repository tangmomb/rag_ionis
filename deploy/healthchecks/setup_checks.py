"""Crée ou met à jour sur healthchecks.io un check par tâche planifiée du RAG.

Chaque tâche systemd envoie un ping au début, à la fin et en cas d'échec
(deploy/systemd/rag-ionis-healthcheck.sh). Healthchecks connaît l'horaire de chaque
tâche : si le ping de fin n'arrive pas dans le délai de grâce, il alerte. Une tâche
qui ne tourne plus du tout ne reste donc pas silencieuse.

Le script est idempotent (upsert sur le slug) : on le relance après avoir changé
un horaire. Il se lance d'où l'on veut, avec la clé d'API (lecture-écriture) du
projet healthchecks :

    HEALTHCHECKS_API_KEY=... python deploy/healthchecks/setup_checks.py
    HEALTHCHECKS_API_KEY=... python deploy/healthchecks/setup_checks.py --dry-run

Les slugs doivent correspondre aux unités systemd : rag-ionis-monitor@quality.service
pingue le slug rag-ionis-monitor-quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass

API_URL = os.getenv("HEALTHCHECKS_API_URL", "https://healthchecks.io/api/v3")
TIMEZONE = "Europe/Paris"


@dataclass(frozen=True)
class Check:
    slug: str
    name: str
    # Même horaire que le timer systemd, au format cron.
    schedule: str
    # Délai après l'heure prévue avant l'alerte : durée normale de la tâche + marge.
    grace_minutes: int
    desc: str

    @property
    def unit(self) -> str:
        """Unité systemd qui pingue ce check."""
        prefix, _, instance = self.slug.rpartition("-")
        if prefix in {"rag-ionis-monitor", "rag-ionis-update"}:
            return f"{prefix}@{instance}.service"
        return f"{self.slug}.service"


CHECKS: tuple[Check, ...] = (
    Check(
        "rag-ionis-backup",
        "Sauvegarde nocturne",
        "30 3 * * *",
        60,
        "Sauvegarde Postgres + Phoenix vers S3 (rag-ionis-backup.service).",
    ),
    Check(
        "rag-ionis-update-stats",
        "Statistiques",
        "30 1 * * *",
        120,
        "Rafraîchissement des statistiques si besoin (rag-ionis-stats.timer).",
    ),
    Check(
        "rag-ionis-update-videos",
        "Mise à jour des vidéos",
        "30 4 * * 0",
        360,
        "Nouvelles vidéos, transcription et indexation (rag-ionis-videos.timer).",
    ),
    Check(
        "rag-ionis-monitor-quality",
        "Rapport qualité",
        "0 5 * * *",
        180,
        "Canari, LLM juge et rapport du matin (rag-ionis-quality.timer).",
    ),
    Check(
        "rag-ionis-monitor-recall",
        "Recall de prod",
        "30 2 1,15 * *",
        180,
        "Recall du golden dataset sur la prod (rag-ionis-recall.timer).",
    ),
    Check(
        "rag-ionis-monitor-restore",
        "Test de restauration",
        "0 4 * * 1",
        60,
        "Restauration de la dernière sauvegarde dans une base jetable (rag-ionis-restore.timer).",
    ),
)


def payload(check: Check) -> dict:
    return {
        "name": check.name,
        "slug": check.slug,
        "desc": check.desc,
        "tags": "rag-ionis",
        "schedule": check.schedule,
        "tz": TIMEZONE,
        "grace": check.grace_minutes * 60,
        # Les alertes utilisent les intégrations configurées dans le projet.
        "channels": "*",
        "unique": ["slug"],
    }


def upsert(api_key: str, check: Check) -> int:
    request = urllib.request.Request(
        f"{API_URL}/checks/",
        data=json.dumps(payload(check)).encode(),
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="affiche les checks sans rien envoyer")
    args = parser.parse_args()
    # Accents et symboles lisibles aussi dans une console Windows.
    sys.stdout.reconfigure(encoding="utf-8")

    if args.dry_run:
        for check in CHECKS:
            print(json.dumps(payload(check), ensure_ascii=False))
        return 0
    api_key = os.getenv("HEALTHCHECKS_API_KEY", "")
    if not api_key:
        print("HEALTHCHECKS_API_KEY est requis (clé d'API du projet, pas la clé de ping).", file=sys.stderr)
        return 2
    for check in CHECKS:
        status = upsert(api_key, check)
        print(f"{check.slug}: {'créé' if status == 201 else 'mis à jour'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
