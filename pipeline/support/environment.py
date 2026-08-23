from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv


SUPPORTED_ENVIRONMENTS = frozenset({"local", "production"})


def _prefer_ipv4_for_local_postgres() -> None:
    """Avoid Windows' slow failed IPv6 attempt against Docker's IPv4-only port."""
    database_url = os.getenv("DATABASE_URL")
    if os.name != "nt" or not database_url:
        return

    parsed = urlsplit(database_url)
    if parsed.hostname != "localhost":
        return

    user_info, separator, host_port = parsed.netloc.rpartition("@")
    if not separator:
        host_port = parsed.netloc
    if not host_port.startswith("localhost"):
        return

    ipv4_netloc = f"127.0.0.1{host_port[len('localhost') :]}"
    if separator:
        ipv4_netloc = f"{user_info}@{ipv4_netloc}"
    os.environ["DATABASE_URL"] = urlunsplit(
        (parsed.scheme, ipv4_netloc, parsed.path, parsed.query, parsed.fragment)
    )


def load_project_env(project_root: Path, *, override: bool = True) -> Path:
    """Load the selected project environment and return its path.

    ``RAG_IONIS_ENV`` is deliberately read before loading a dotenv file so a
    VPS can select production from systemd, cron, or its shell environment.
    Local development defaults to ``.env.local`` and falls back to the legacy
    ``.env`` file while projects migrate.
    """
    environment_name = os.getenv("RAG_IONIS_ENV", "local").strip().lower()
    if environment_name not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
        raise RuntimeError(
            f"RAG_IONIS_ENV doit valoir l'une de ces valeurs: {supported}."
        )

    selected_path = Path(project_root) / f".env.{environment_name}"
    if not selected_path.exists() and environment_name == "local":
        legacy_path = Path(project_root) / ".env"
        if legacy_path.exists():
            selected_path = legacy_path
    environment_file_optional = os.getenv(
        "RAG_IONIS_ENV_FILE_OPTIONAL",
        "",
    ).strip().lower() in {"1", "true", "yes"}
    if not selected_path.exists() and environment_file_optional:
        _prefer_ipv4_for_local_postgres()
        return selected_path
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Fichier d'environnement introuvable: {selected_path}"
        )

    load_dotenv(selected_path, override=override)
    _prefer_ipv4_for_local_postgres()
    return selected_path
