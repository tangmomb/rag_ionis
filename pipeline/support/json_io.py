from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


_MISSING = object()
_REPLACE_ATTEMPTS = 6
_REPLACE_INITIAL_DELAY_SECONDS = 0.05


def _replace_with_retry(source: Path, target: Path) -> None:
    delay = _REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


def read_json(
    path: str | Path,
    *,
    default: Any = _MISSING,
    encoding: str = "utf-8",
) -> Any:
    """Read a JSON document, optionally returning a fallback on read errors."""
    try:
        with Path(path).open("r", encoding=encoding) as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        if default is _MISSING:
            raise
        return default


def write_json(
    path: str | Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    sort_keys: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write one JSON document in the target directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=ensure_ascii,
                indent=indent,
                sort_keys=sort_keys,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def write_jsonl(
    path: str | Path,
    records: Iterable[Any],
    *,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write newline-delimited JSON records."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                json.dump(record, handle, ensure_ascii=ensure_ascii)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target
