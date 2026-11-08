"""Helpers S3 du VPS : client et envoi d'un dossier sous un prefixe."""

from __future__ import annotations

import os
from pathlib import Path

import boto3


class S3TransferError(RuntimeError):
    """Raised when a directory holds nothing to upload."""


def s3_client(region: str | None = None):
    access_key = os.getenv("S3_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    options = {}
    if region:
        options["region_name"] = region
    if endpoint_url:
        options["endpoint_url"] = endpoint_url
    if access_key and secret_key:
        options["aws_access_key_id"] = access_key
        options["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **options)


def upload_s3_directory(
    client,
    bucket: str,
    source: Path,
    prefix: str,
    *,
    excluded_suffixes: set[str] | frozenset[str] = frozenset(),
    included_roots: set[str] | frozenset[str] | None = None,
) -> int:
    """Upload one directory below an exact job prefix."""
    source = Path(source).resolve()
    normalized = prefix.strip("/")
    if not source.is_dir():
        raise FileNotFoundError(f"Dossier introuvable: {source}")
    if not normalized:
        raise ValueError("Le prefixe S3 ne peut pas etre vide.")
    excluded = {suffix.lower() for suffix in excluded_suffixes}
    included = set(included_roots) if included_roots is not None else None
    files = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file()
        and path.suffix.lower() not in excluded
        and (included is None or path.relative_to(source).parts[0] in included)
    ]
    if not files:
        raise S3TransferError(f"Aucun fichier a envoyer depuis {source}")
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(source).as_posix()
        client.upload_file(str(path), bucket, f"{normalized}/{relative}")
        print(
            f"\r[s3 upload] {index}/{len(files)}",
            end="" if index < len(files) else "\n",
            flush=True,
        )
    return len(files)
