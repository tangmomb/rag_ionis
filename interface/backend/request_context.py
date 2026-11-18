"""Identifiant de la requête HTTP en cours, partagé entre le middleware et le RAG.

Le middleware (interface/app.py) le fixe pour chaque appel ; run_rag le lit pour
le joindre à la trace Phoenix et écrire une ligne de log qui relie request_id,
trace_id, conversation et message. Dans Grafana, une erreur mène ainsi à sa
trace complète, et inversement.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

REQUEST_LOGGER_NAME = "rag_ionis.http"
current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)


def log_request_event(event: str, **fields: Any) -> None:
    """Write one JSON log line (formatted by interface.request_logging)."""
    logging.getLogger(REQUEST_LOGGER_NAME).info(
        event,
        extra={"request_id": current_request_id.get(), **fields},
    )
