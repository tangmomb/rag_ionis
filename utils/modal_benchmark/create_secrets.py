"""Create the two Modal secrets used by deploy/modal_app.py from .env.local.

Only the variables the GPU worker needs are sent to Modal; values are never
printed. Run it yourself: it sends credentials to Modal.

    python utils/modal_benchmark/create_secrets.py

Registry: set SCW_REGISTRY_PASSWORD to a Scaleway API key restricted to
ContainerRegistryReadOnly. Without it, SCW_SECRET_KEY is used, which gives
Modal far wider access to the Scaleway project.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_KEYS = (
    "S3_BUCKET_NAME",
    "S3_REGION",
    "S3_ENDPOINT_URL",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "YOUTUBE_API_KEY",
    "OPENAI_API_KEY",
    "HUGGINGFACE_TOKEN",
)


def create_secret(modal_cli: str, name: str, values: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="modal-secret-") as directory:
        dotenv_path = Path(directory) / "secret.env"
        dotenv_path.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )
        subprocess.run(
            [modal_cli, "secret", "create", name, "--from-dotenv", str(dotenv_path), "--force"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    print(f"Secret Modal {name!r} cree ({', '.join(values)}).")


def main() -> None:
    env_path = PROJECT_ROOT / os.getenv("RAG_IONIS_ENV_FILE", ".env.local")
    env = {key: value for key, value in dotenv_values(env_path).items() if value}
    missing = [key for key in WORKER_KEYS if key not in env]
    if missing:
        sys.exit(f"Variables absentes de {env_path.name}: {', '.join(missing)}")
    modal_cli = shutil.which("modal") or str(Path(sys.executable).with_name("modal"))

    registry_password = os.getenv("SCW_REGISTRY_PASSWORD") or env.get("SCW_REGISTRY_PASSWORD")
    if not registry_password:
        registry_password = env.get("SCW_SECRET_KEY", "")
        print("Attention : SCW_REGISTRY_PASSWORD absent, SCW_SECRET_KEY est utilise.")
    if not registry_password:
        sys.exit("Aucun mot de passe de registre disponible.")

    create_secret(
        modal_cli,
        "rag-ionis-registry",
        {"REGISTRY_USERNAME": "nologin", "REGISTRY_PASSWORD": registry_password},
    )
    create_secret(modal_cli, "rag-ionis-worker", {key: env[key] for key in WORKER_KEYS})


if __name__ == "__main__":
    main()
