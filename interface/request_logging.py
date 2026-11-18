"""Journal JSON des requêtes HTTP de l'API.

Une ligne JSON par appel /api (méthode, route, statut, durée, client haché,
request_id), lisible par `docker logs` comme par un collecteur de logs. Le
request_id est renvoyé dans l'en-tête X-Request-ID : une erreur signalée par un
utilisateur se retrouve dans les logs par cet identifiant.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

LOGGER_NAME = "rag_ionis.http"
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in _RESERVED}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_json_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # The JSON line replaces uvicorn's plain-text access log.
    logging.getLogger("uvicorn.access").disabled = True
    return logger


def should_log(path: str) -> bool:
    return path.startswith("/api/")
