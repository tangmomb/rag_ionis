"""Worker GPU sur Modal (prototype), alternative a la VM Scaleway.

La fonction reprend l'image GPU deja publiee sur le registre Scaleway et appelle
``pipeline.workers.scaleway_ingestion.process_job`` avec le meme ``job_input``.
Le code execute est donc celui de l'image, comme sur la VM.

Deploiement, depuis un poste ou ``modal`` est installe et authentifie :

    SCALEWAY_CONTAINER_IMAGE=rg.fr-par.scw.cloud/NAMESPACE/rag-ionis-scaleway:VERSION \\
        modal deploy deploy/modal_app.py

Test du GPU, sans S3 ni pipeline :

    SCALEWAY_CONTAINER_IMAGE=... modal run deploy/modal_app.py

Voir deploy/MODAL.md pour les secrets a creer.
"""

from __future__ import annotations

import os

import modal


APP_NAME = os.environ.get("MODAL_APP_NAME", "rag-ionis-gpu")
# Only read when deploying: in the container the image is already built.
IMAGE_TAG = os.environ.get("SCALEWAY_CONTAINER_IMAGE", "").strip()
REGION = os.environ.get("MODAL_REGION", "").strip() or None
MAX_CONTAINERS = int(os.environ.get("MODAL_MAX_CONTAINERS", "1"))

if modal.is_local() and not IMAGE_TAG:
    raise RuntimeError(
        "SCALEWAY_CONTAINER_IMAGE doit designer l'image GPU publiee pour deployer."
    )

app = modal.App(APP_NAME)

# REGISTRY_USERNAME=nologin, REGISTRY_PASSWORD=<cle secrete Scaleway>.
registry_secret = modal.Secret.from_name("rag-ionis-registry")
# Same variables as /etc/rag-ionis/scaleway-worker.env (S3, OpenAI, Hugging Face...).
worker_secret = modal.Secret.from_name("rag-ionis-worker")
# Replaces /var/lib/rag-ionis/models on the VM: Whisper, Paddle and HF caches.
models_volume = modal.Volume.from_name("rag-ionis-models", create_if_missing=True)

image = modal.Image.from_registry(
    IMAGE_TAG or "unused-in-container",
    secret=registry_secret,
    # The CUDA image only ships python3; Modal looks for ``python``.
    setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/local/bin/python"],
).env(
    {
        "PYTHONPATH": "/app",
        "SCALEWAY_WORKER": "1",
        "PIPELINE_EXECUTION_BACKEND": "local",
    }
)


@app.function(
    image=image,
    gpu="L4",
    timeout=6 * 3600,
    volumes={"/models": models_volume},
    secrets=[worker_secret],
    max_containers=MAX_CONTAINERS,
    region=REGION,
)
def process_job(job_input: dict) -> dict:
    # The pipeline resolves models/ relative to the image's working directory.
    os.chdir("/app")
    from pipeline.workers.scaleway_ingestion import process_job as run_job

    try:
        return run_job(job_input)
    finally:
        # Keep models downloaded during this job for the next containers.
        models_volume.commit()


@app.local_entrypoint()
def healthcheck() -> None:
    print(process_job.remote({"operation": "healthcheck"}))
