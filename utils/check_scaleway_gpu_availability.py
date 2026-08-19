"""Display GPU Instance availability across Scaleway zones.

Usage:
    python utils/check_scaleway_gpu_availability.py
    python utils/check_scaleway_gpu_availability.py --resource gpu --model L40S
    python utils/check_scaleway_gpu_availability.py --gpu L40S --zone pl-waw-2

The Scaleway API token is read from SCW_SECRET_KEY in the environment or .env.
It is never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        help="Type d'instance à rechercher (ex. L40S ou DEV1-S).",
    )
    parser.add_argument(
        "--zone",
        action="append",
        dest="zones",
        help="Zone à vérifier ; peut être répété. Par défaut : toutes les zones.",
    )
    return parser.parse_args()


def select_resource_and_model(args: argparse.Namespace) -> tuple[str, str]:
    resource = args.resource
    model = args.model
    if resource is None and model is None:
        resource = input("Ressource à vérifier [gpu/cpu] (défaut : gpu) : ").strip().lower()
    resource = resource or "gpu"
    if resource not in {"gpu", "cpu"}:
        raise ValueError("La ressource doit être gpu ou cpu.")

    default_model = "L40S" if resource == "gpu" else "DEV1-S"
    if model is None and args.resource is None:
        model = input(f"Type d'instance (défaut : {default_model}) : ").strip()
    model = model or default_model
    return resource, model


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    token = os.getenv("SCW_SECRET_KEY", "").strip()
    if not token:
        print("SCW_SECRET_KEY manquant dans l'environnement ou .env.", file=sys.stderr)
        return 2

    args = parse_args()
    try:
        resource, model = select_resource_and_model(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    zones = tuple(args.zones or DEFAULT_ZONES)
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

    print(f"Disponibilité {resource.upper()} pour {model} :")
    print_results(rows, resource)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
