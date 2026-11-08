"""Execution des jobs GPU sur Modal (prototype).

Le VPS appelle une fonction Modal deployee par ``deploy/modal_app.py`` avec le
meme ``job_input`` que celui ecrit pour la VM Scaleway. Modal alloue un GPU L4,
execute ``process_job`` et renvoie sa sortie : il n'y a ni VM a allumer, ni
file d'attente S3 a surveiller, ni arret a declencher.

L'authentification du client lit ``MODAL_TOKEN_ID`` et ``MODAL_TOKEN_SECRET``.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

from pipeline.support.scaleway_jobs import ScalewayJobError


DEFAULT_APP_NAME = "rag-ionis-gpu"
DEFAULT_FUNCTION_NAME = "process_job"


class ModalJobError(ScalewayJobError):
    """Raised when a Modal GPU job cannot complete."""


@dataclass(frozen=True)
class ModalConfig:
    app_name: str = DEFAULT_APP_NAME
    function_name: str = DEFAULT_FUNCTION_NAME
    environment_name: str = ""
    poll_seconds: float = 60.0
    timeout_seconds: float = 21600.0

    @classmethod
    def from_env(cls) -> "ModalConfig":
        poll_seconds = float(os.getenv("MODAL_POLL_SECONDS", "60"))
        timeout_seconds = float(os.getenv("MODAL_JOB_TIMEOUT_SECONDS", "21600"))
        if poll_seconds < 1:
            raise ValueError("MODAL_POLL_SECONDS doit valoir au moins 1.")
        if timeout_seconds <= 0:
            raise ValueError("MODAL_JOB_TIMEOUT_SECONDS doit etre positif.")
        return cls(
            app_name=os.getenv("MODAL_APP_NAME", DEFAULT_APP_NAME).strip()
            or DEFAULT_APP_NAME,
            function_name=os.getenv("MODAL_FUNCTION_NAME", DEFAULT_FUNCTION_NAME).strip()
            or DEFAULT_FUNCTION_NAME,
            environment_name=os.getenv("MODAL_ENVIRONMENT", "").strip(),
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )


def _is_wait_timeout(error: BaseException) -> bool:
    """True when ``FunctionCall.get`` only stopped waiting; the job still runs."""
    # FunctionTimeoutError means the remote function itself hit its timeout.
    if type(error).__name__ == "FunctionTimeoutError":
        return False
    if isinstance(error, TimeoutError):
        return True
    try:
        from modal.exception import TimeoutError as ModalTimeoutError
    except ImportError:
        return False
    return isinstance(error, ModalTimeoutError)


class ModalClient:
    def __init__(self, config: ModalConfig, *, function=None, clock=time.monotonic):
        self.config = config
        self._function = function
        self.clock = clock

    def function(self):
        if self._function is None:
            try:
                import modal
            except ImportError as error:
                raise RuntimeError(
                    "Le paquet 'modal' est requis pour PIPELINE_EXECUTION_BACKEND=modal."
                ) from error
            options = {}
            if self.config.environment_name:
                options["environment_name"] = self.config.environment_name
            self._function = modal.Function.from_name(
                self.config.app_name,
                self.config.function_name,
                **options,
            )
        return self._function

    def _wait(self, call, job_id: str) -> object:
        deadline = self.clock() + self.config.timeout_seconds
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise ModalJobError(
                    f"Delai depasse pour le job Modal {job_id} "
                    f"({self.config.timeout_seconds:g} s)."
                )
            wait_seconds = min(self.config.poll_seconds, remaining)
            started = self.clock()
            try:
                return call.get(timeout=wait_seconds)
            except Exception as error:
                waited = self.clock() - started
                # A remote TimeoutError (socket timeout in the pipeline, say) is
                # re-raised at once, whereas a wait timeout uses the full delay.
                if _is_wait_timeout(error) and waited >= wait_seconds * 0.5:
                    print(f"[modal] job={job_id} toujours en cours", flush=True)
                    continue
                raise ModalJobError(
                    f"Job Modal {job_id} en echec: {type(error).__name__}: {error}"
                ) from error

    def run(self, job_input: dict, *, object_store=None, bucket: str = "") -> tuple[str, dict]:
        """Run one job and return ``(job_id, output)`` like ``ScalewayClient.run``.

        ``object_store`` and ``bucket`` are accepted for interface parity: the
        Modal worker reads and writes S3 itself and needs no control channel.
        """
        control = job_input.get("control")
        job_id = str((control or {}).get("job_id") or "").strip() or uuid.uuid4().hex
        call = self.function().spawn(job_input)
        print(
            f"[modal] job soumis: job={job_id} call={getattr(call, 'object_id', '?')} "
            f"app={self.config.app_name}",
            flush=True,
        )
        try:
            output = self._wait(call, job_id)
        except BaseException:
            try:
                call.cancel()
            except Exception as cancel_error:
                print(
                    f"[modal] avertissement annulation job={job_id}: {cancel_error}",
                    flush=True,
                )
            raise
        if not isinstance(output, dict):
            raise ModalJobError(f"Le job Modal {job_id} a renvoye une sortie invalide.")
        print(f"[modal] job termine: {job_id}", flush=True)
        return job_id, output
