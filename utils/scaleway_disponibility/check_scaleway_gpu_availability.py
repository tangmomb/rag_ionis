"""Display GPU Instance availability across Scaleway zones.

Usage:
    python utils/scaleway_disponibility/check_scaleway_gpu_availability.py
    python utils/scaleway_disponibility/check_scaleway_gpu_availability.py --resource gpu --model L4
    python utils/scaleway_disponibility/check_scaleway_gpu_availability.py --gpu L4 --zone pl-waw-2
    python utils/scaleway_disponibility/check_scaleway_gpu_availability.py --record --resource gpu --model L4
    python utils/scaleway_disponibility/check_scaleway_gpu_availability.py --summary --resource gpu --model L4

The Scaleway API token is read from SCW_SECRET_KEY in the selected environment file.
It is never printed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.support.environment import load_project_env

DEFAULT_HISTORY_FILE = Path(__file__).resolve().with_name(
    "scaleway_gpu_availability.csv"
)
HISTORY_FIELDS = (
    "timestamp_utc",
    "resource",
    "model",
    "zone",
    "instance_type",
    "gpu_name",
    "vram_gib",
    "availability",
)
DEFAULT_ZONES = (
    "fr-par-1",
    "fr-par-2",
    "fr-par-3",
    "nl-ams-1",
    "nl-ams-2",
    "nl-ams-3",
    "pl-waw-1",
    "pl-waw-2",
    "pl-waw-3",
    "it-mil-1",
)


def server_types(zone: str, token: str) -> list[dict]:
    """Return all Instance types exposed for one zone."""
    url = f"https://api.scaleway.com/instance/v2alpha1/zones/{zone}/server-types"
    headers = {"X-Auth-Token": token}
    results: list[dict] = []
    page_token = ""

    while True:
        params = {"page_size": 100}
        if page_token:
            params["page_token"] = page_token
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        results.extend(payload.get("server_types", []))
        page_token = str(payload.get("next_page_token") or "")
        if not page_token:
            return results


def matching_types(
    zone: str,
    token: str,
    resource: str,
    model: str,
) -> list[dict]:
    wanted = model.strip().lower()
    matches = []
    for item in server_types(zone, token):
        info = item.get("gpu_info") or {}
        name = str(item.get("name") or "")
        actual_gpu = str(info.get("name") or "")
        is_gpu = bool(item.get("gpu_count") or info)
        if resource == "gpu" and is_gpu and (
            wanted in name.lower() or wanted in actual_gpu.lower()
        ):
            matches.append(item)
        elif resource == "cpu" and not is_gpu and wanted in name.lower():
            matches.append(item)
    return matches


def format_memory(bytes_value: object) -> str:
    try:
        gib = float(bytes_value) / (1024**3)
    except (TypeError, ValueError):
        return "-"
    return f"{gib:.0f} GiB"


def memory_gib(bytes_value: object) -> str:
    try:
        return f"{float(bytes_value) / (1024**3):.1f}"
    except (TypeError, ValueError):
        return ""


def append_history(
    history_file: Path,
    rows: list[tuple[str, dict]],
    resource: str,
    model: str,
) -> int:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_file.exists() and history_file.stat().st_size > 0
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with history_file.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        for zone, item in rows:
            info = item.get("gpu_info") or {}
            writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    "resource": resource,
                    "model": model,
                    "zone": zone,
                    "instance_type": item.get("name") or "",
                    "gpu_name": info.get("name") or ("CPU" if resource == "cpu" else ""),
                    "vram_gib": memory_gib(info.get("memory")) if resource == "gpu" else "",
                    "availability": item.get("availability") or "unknown",
                }
            )
    return len(rows)


def read_history(
    history_file: Path,
    resource: str,
    model: str,
    zones: tuple[str, ...],
    days: int,
) -> list[dict[str, str]]:
    if not history_file.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    wanted_zones = set(zones)
    rows: list[dict[str, str]] = []
    with history_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("resource") != resource or row.get("model", "").lower() != model.lower():
                continue
            if row.get("zone") not in wanted_zones:
                continue
            try:
                timestamp = datetime.fromisoformat(row["timestamp_utc"])
            except (KeyError, ValueError):
                continue
            if timestamp >= cutoff:
                rows.append(row)
    return rows


def print_summary(rows: list[dict[str, str]], days: int) -> None:
    if not rows:
        print(f"Aucun relevé disponible sur les {days} derniers jours.")
        return

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["zone"], row["instance_type"])].append(row)

    headers = (
        "ZONE",
        "TYPE",
        "RELEVES",
        "AVAILABLE",
        "LOW_STOCK",
        "OUT_OF_STOCK",
        "DERNIER ETAT",
    )
    values = []
    for (zone, instance_type), group in sorted(groups.items()):
        counts = Counter(row.get("availability", "unknown") for row in group)
        total = len(group)
        percent = lambda status: f"{100 * counts[status] / total:.1f}%"
        latest = max(group, key=lambda row: row.get("timestamp_utc", ""))
        values.append(
            (
                zone,
                instance_type,
                str(total),
                percent("available"),
                percent("low_stock"),
                percent("out_of_stock"),
                latest.get("availability", "unknown"),
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    print(f"Moyenne de disponibilité sur {days} jours ({len(rows)} relevés) :")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in values:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_results(rows: list[tuple[str, dict]], resource: str) -> None:
    if not rows:
        print("Aucun type GPU correspondant trouvé.")
        return

    headers = ("ZONE", "TYPE", "GPU/CPU", "VRAM", "DISPONIBILITE")
    values = []
    for zone, item in rows:
        info = item.get("gpu_info") or {}
        values.append(
            (
                zone,
                str(item.get("name") or "-"),
                str(info.get("name") or ("CPU" if resource == "cpu" else "-")),
                format_memory(info.get("memory")) if resource == "gpu" else "-",
                str(item.get("availability") or "unknown"),
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in values:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vérifie la disponibilité des types d'instances Scaleway par zone."
    )
    parser.add_argument(
        "--resource",
        choices=("gpu", "cpu"),
        help="Ressource à vérifier (GPU ou CPU). Sans option, le script demande.",
    )
    parser.add_argument(
        "--model",
        "--gpu",
        dest="model",
        help="Type d'instance à rechercher (ex. L4 ou DEV1-S).",
    )
    parser.add_argument(
        "--zone",
        action="append",
        dest="zones",
        help="Zone à vérifier ; peut être répété. Par défaut : toutes les zones.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enregistre le relevé courant dans l'historique local.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Affiche la moyenne calculée depuis l'historique local.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Fenêtre de calcul pour --summary (défaut : 7 jours).",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_FILE,
        help="Fichier CSV d'historique.",
    )
    return parser.parse_args()


def select_resource_and_model(
    args: argparse.Namespace,
    *,
    interactive: bool = True,
) -> tuple[str, str]:
    resource = args.resource
    model = args.model
    if interactive and resource is None and model is None:
        resource = input("Ressource à vérifier [gpu/cpu] (défaut : gpu) : ").strip().lower()
    resource = resource or "gpu"
    if resource not in {"gpu", "cpu"}:
        raise ValueError("La ressource doit être gpu ou cpu.")

    default_model = "L4" if resource == "gpu" else "DEV1-S"
    if interactive and model is None and args.resource is None:
        model = input(f"Type d'instance (défaut : {default_model}) : ").strip()
    model = model or default_model
    return resource, model


def main() -> int:
    load_project_env(PROJECT_ROOT, override=False)
    args = parse_args()
    token = os.getenv("SCW_SECRET_KEY", "").strip()
    if not token:
        if not args.summary:
            print("SCW_SECRET_KEY manquant dans l'environnement ou .env.", file=sys.stderr)
            return 2

    try:
        resource, model = select_resource_and_model(
            args,
            interactive=not (args.record or args.summary),
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    zones = tuple(args.zones or DEFAULT_ZONES)

    if args.days <= 0:
        print("--days doit être supérieur à zéro.", file=sys.stderr)
        return 2

    if args.summary and not args.record:
        rows = read_history(args.history_file, resource, model, zones, args.days)
        print_summary(rows, args.days)
        return 0

    rows: list[tuple[str, dict]] = []
    for zone in zones:
        try:
            rows.extend(
                (zone, item)
                for item in matching_types(zone, token, resource, model)
            )
        except requests.RequestException as error:
            print(f"{zone}: erreur API Scaleway ({error})", file=sys.stderr)
            return 1

    if args.record:
        count = append_history(args.history_file, rows, resource, model)
        print(f"{count} relevé(s) ajouté(s) dans {args.history_file}")
    if not args.summary:
        print(f"Disponibilité {resource.upper()} pour {model} :")
        print_results(rows, resource)
    else:
        print_summary(
            read_history(args.history_file, resource, model, zones, args.days),
            args.days,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
