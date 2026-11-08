"""Choix du client qui execute les jobs GPU distants."""

from __future__ import annotations

import os


REMOTE_BACKENDS = frozenset({"scaleway", "modal"})
EXECUTION_BACKENDS = REMOTE_BACKENDS | {"local"}
BACKEND_ERROR = "PIPELINE_EXECUTION_BACKEND doit valoir 'local', 'scaleway' ou 'modal'."


def configured_backend() -> str:
    backend = os.getenv("PIPELINE_EXECUTION_BACKEND", "scaleway").strip().lower()
    if backend not in EXECUTION_BACKENDS:
        raise RuntimeError(BACKEND_ERROR)
    return backend


def remote_job_client(backend: str):
    """Return a client whose ``run(job_input, object_store=, bucket=)`` runs one job."""
    if backend == "modal":
        from pipeline.support.modal_jobs import ModalClient, ModalConfig

        return ModalClient(ModalConfig.from_env())
    if backend == "scaleway":
        from pipeline.support.scaleway_jobs import ScalewayClient, ScalewayConfig

        return ScalewayClient(ScalewayConfig.from_env())
    raise RuntimeError(f"Backend distant inconnu: {backend!r}")
