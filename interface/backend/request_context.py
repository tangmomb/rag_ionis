"""Contexte de la requête HTTP en cours, partagé entre le middleware et le RAG.

Le middleware (interface/app.py) fixe pour chaque appel un identifiant et un
dictionnaire de détails. run_rag y dépose le trace_id Phoenix, la conversation et
le message ; le middleware écrit ensuite une seule ligne de log, à la fin de la
réponse (streaming compris), avec la vraie durée et ces détails. Dans Grafana,
une erreur mène ainsi à sa trace complète, et inversement.

Hors requête HTTP (canari, expériences), les détails sont écrits sur une ligne à part.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

REQUEST_LOGGER_NAME = "rag_ionis.http"
current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)
# Mutable on purpose: the streaming worker thread runs in a copy of the request
# context, so it shares this same dict object with the middleware.
current_request_details: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_request_details", default=None
)


def record_request_details(event: str, **fields: Any) -> None:
    """Attach RAG details to the request's log line, or log them on their own."""
    details = current_request_details.get()
    if details is not None:
        details.update({"rag_event": event, **fields})
        return
    logging.getLogger(REQUEST_LOGGER_NAME).info(
        event,
        extra={"request_id": current_request_id.get(), **fields},
    )
