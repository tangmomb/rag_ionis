from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


SUPPORTED_ENVIRONMENTS = frozenset({"local", "production"})


def load_project_env(project_root: Path, *, override: bool = True) -> Path:
    """Load the selected environment for the API process.

    Production Docker deployments inject variables with Compose's ``env_file``
    instead of mounting the secret dotenv file into the API image.
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
    if not selected_path.exists():
        if os.getenv("RAG_IONIS_ENV_FILE_OPTIONAL", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            return selected_path
        raise FileNotFoundError(
            f"Fichier d'environnement introuvable: {selected_path}"
        )

    load_dotenv(selected_path, override=override)
    return selected_path
