from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


_MISSING = object()


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
        os.replace(temporary_path, target)
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
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target
