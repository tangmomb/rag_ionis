from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from pipeline.support.json_io import read_json, write_json


COMPLETED_BATCH_STATUS = "completed"
TERMINAL_BATCH_STATUSES = frozenset(
    {
        COMPLETED_BATCH_STATUS,
        "failed",
        "expired",
        "cancelled",
    }
)

BatchState = dict[str, Any]
WaitCallback = Callable[[BatchState, float], None]
FINGERPRINT_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_request_fingerprint(
    model: str,
    *,
    sources: Iterable[Path | str] = (),
    custom_ids: Iterable[object] = (),
    options: Mapping[str, Any] | None = None,
) -> str:
    """Return a deterministic digest for one logical Batch submission.

    Source contents are hashed rather than relying on mtimes, so copying a
    source unchanged does not invalidate a resumable batch. Callers should put
    every request-shaping option (prompt/profile/limit) in ``options``.
    """

    source_payload = []
    for source in sources:
        path = Path(source)
        stat = path.stat()
        source_payload.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload = {
        "version": FINGERPRINT_VERSION,
        "model": str(model),
        "sources": source_payload,
        "custom_ids": [str(custom_id) for custom_id in custom_ids],
        "options": dict(options or {}),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def batch_state_matches(
    state: Mapping[str, Any] | None,
    request_fingerprint: str,
) -> bool:
    """Whether a persisted state belongs to the current logical request."""

    return bool(
        state
        and state.get("request_fingerprint") == request_fingerprint
    )


def is_terminal_batch_status(status: object) -> bool:
    return status in TERMINAL_BATCH_STATUSES


def save_batch_state(path: Path | str, state: Mapping[str, Any]) -> Path:
    """Persist a batch state atomically so an interrupted write stays readable."""
    return write_json(path, dict(state))


def load_batch_state(path: Path | str) -> BatchState | None:
    target = Path(path)
    if not target.exists():
        return None
    payload = read_json(target)
    if not isinstance(payload, dict):
        raise ValueError(f"Etat batch invalide dans {target}: un objet JSON est attendu.")
    return payload


def _request_counts_payload(request_counts: object) -> dict[str, Any]:
    if hasattr(request_counts, "model_dump"):
        return request_counts.model_dump()
    if isinstance(request_counts, Mapping):
        return dict(request_counts)
    try:
        return dict(request_counts)
    except (TypeError, ValueError):
        return vars(request_counts)


def refresh_batch_state(
    client: Any,
    state: Mapping[str, Any],
    state_path: Path | str,
) -> BatchState:
    batch = client.batches.retrieve(state["batch_id"])
    refreshed = dict(state)
    refreshed.update(
        {
            "status": batch.status,
            "input_file_id": getattr(batch, "input_file_id", refreshed.get("input_file_id")),
            "output_file_id": getattr(batch, "output_file_id", refreshed.get("output_file_id")),
            "error_file_id": getattr(batch, "error_file_id", refreshed.get("error_file_id")),
        }
    )
    request_counts = getattr(batch, "request_counts", None)
    if request_counts is not None:
        refreshed["request_counts"] = _request_counts_payload(request_counts)
    save_batch_state(state_path, refreshed)
    return refreshed


def poll_batch_state(
    client: Any,
    state: Mapping[str, Any],
    state_path: Path | str,
    *,
    wait: bool = False,
    poll_interval_seconds: float = 30,
    on_wait: WaitCallback | None = None,
) -> BatchState:
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds doit etre positif ou nul.")

    refreshed = refresh_batch_state(client, state, state_path)
    while wait and not is_terminal_batch_status(refreshed.get("status")):
        if on_wait is not None:
            on_wait(refreshed, poll_interval_seconds)
        time.sleep(poll_interval_seconds)
        refreshed = refresh_batch_state(client, refreshed, state_path)
    return refreshed


def parse_jsonl(path: Path | str) -> list[Any]:
    target = Path(path)
    if not target.exists():
        return []

    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def records_by_custom_id(records: list[Any]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        custom_id = record.get("custom_id")
        if custom_id:
            indexed[str(custom_id)] = record
    return indexed


def download_batch_files(
    client: Any,
    state: Mapping[str, Any],
    output_path: Path | str,
    error_path: Path | str | None = None,
) -> None:
    def download(file_id: str, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            client.files.content(file_id).write_to_file(temporary_path)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    output_file_id = state.get("output_file_id")
    if output_file_id:
        download(str(output_file_id), output_path)

    error_file_id = state.get("error_file_id")
    if error_file_id and error_path is not None:
        download(str(error_file_id), error_path)
