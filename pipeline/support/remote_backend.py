"""Choix du backend d'execution du pipeline.

- ``local`` : tout tourne sur la machine courante.
- ``modal`` : le pipeline tourne sur la machine courante et ses taches GPU
  partent sur Modal (voir pipeline/gpu_offload.py).
- ``scaleway`` : le job entier part sur la VM GPU Scaleway.
"""

from __future__ import annotations

import os


REMOTE_BACKENDS = frozenset({"scaleway"})
EXECUTION_BACKENDS = REMOTE_BACKENDS | {"local", "modal"}
BACKEND_ERROR = "PIPELINE_EXECUTION_BACKEND doit valoir 'local', 'modal' ou 'scaleway'."


def configured_backend() -> str:
    backend = os.getenv("PIPELINE_EXECUTION_BACKEND", "scaleway").strip().lower()
    if backend not in EXECUTION_BACKENDS:
        raise RuntimeError(BACKEND_ERROR)
    return backend


def remote_job_client(backend: str):
    """Return a client whose ``run(job_input, object_store=, bucket=)`` runs one job."""
    if backend == "scaleway":
        from pipeline.support.scaleway_jobs import ScalewayClient, ScalewayConfig

        return ScalewayClient(ScalewayConfig.from_env())
    raise RuntimeError(f"Backend distant inconnu: {backend!r}")
