from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import boto3
import requests


TERMINAL_JOB_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class RunpodJobError(RuntimeError):
    """Raised when a Runpod job cannot be submitted or does not complete."""


@dataclass(frozen=True)
class RunpodConfig:
    api_key: str
    endpoint_id: str
    poll_seconds: float = 5.0
    timeout_seconds: float = 21600.0

    @classmethod
    def from_env(cls) -> "RunpodConfig":
        api_key = os.getenv("RUNPOD_API_KEY", "").strip()
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
        if not api_key:
            raise RuntimeError("RUNPOD_API_KEY manquant dans l'environnement.")
        if not endpoint_id:
            raise RuntimeError("RUNPOD_ENDPOINT_ID manquant dans l'environnement.")
        return cls(
            api_key=api_key,
            endpoint_id=endpoint_id,
            poll_seconds=float(os.getenv("RUNPOD_POLL_SECONDS", "5")),
            timeout_seconds=float(os.getenv("RUNPOD_JOB_TIMEOUT_SECONDS", "21600")),
        )


class RunpodClient:
    def __init__(self, config: RunpodConfig, *, session=None, sleep=time.sleep):
        self.config = config
        self.session = session or requests.Session()
        self.sleep = sleep
        self.base_url = f"https://api.runpod.ai/v2/{config.endpoint_id}"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": self.config.api_key,
            "Content-Type": "application/json",
        }

    def submit(self, job_input: dict) -> str:
        response = self.session.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json={"input": job_input},
            timeout=30,
        )
        response.raise_for_status()
        job_id = str(response.json().get("id") or "").strip()
        if not job_id:
            raise RunpodJobError("Runpod n'a renvoye aucun identifiant de job.")
        return job_id

    def status(self, job_id: str) -> dict:
        response = self.session.get(
            f"{self.base_url}/status/{job_id}",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RunpodJobError("Reponse de statut Runpod invalide.")
        return payload

    def wait(self, job_id: str) -> dict:
        deadline = time.monotonic() + self.config.timeout_seconds
        previous_status = None
        while True:
            payload = self.status(job_id)
            status = str(payload.get("status") or "").upper()
            if status != previous_status:
                print(f"[runpod] job={job_id} status={status or 'UNKNOWN'}", flush=True)
                previous_status = status
            if status == "COMPLETED":
                output = payload.get("output")
                if not isinstance(output, dict):
                    raise RunpodJobError("Le job Runpod a termine sans sortie JSON exploitable.")
                return output
            if status in TERMINAL_JOB_STATES:
                detail = payload.get("error") or payload.get("output") or payload
                raise RunpodJobError(f"Job Runpod {job_id} termine avec {status}: {detail}")
            if time.monotonic() >= deadline:
                raise RunpodJobError(
                    f"Delai d'attente depasse pour le job Runpod {job_id} "
                    f"({self.config.timeout_seconds:g} s)."
                )
            self.sleep(max(0.1, self.config.poll_seconds))

    def run(self, job_input: dict) -> tuple[str, dict]:
        job_id = self.submit(job_input)
        print(f"[runpod] job soumis: {job_id}", flush=True)
        return job_id, self.wait(job_id)


def s3_client(region: str | None = None):
    access_key = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    options = {}
    if region:
        options["region_name"] = region
    if access_key and secret_key:
        options["aws_access_key_id"] = access_key
        options["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **options)


def download_s3_prefix(
    client,
    bucket: str,
    prefix: str,
    destination: Path,
) -> int:
    """Download a prefix while preserving paths below that exact prefix."""
    normalized = prefix.strip("/")
    if not normalized:
        raise ValueError("Le prefixe S3 de telechargement ne peut pas etre vide.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{normalized}/"):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            relative = key.removeprefix(f"{normalized}/")
            relative_path = PurePosixPath(relative)
            parts = relative_path.parts
            if (
                not parts
                or relative_path.is_absolute()
                or "\\" in relative
                or any(part in {"", ".", ".."} for part in parts)
            ):
                continue
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    if count == 0:
        raise RunpodJobError(f"Aucun artefact trouve dans s3://{bucket}/{normalized}/")
    return count


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
        raise FileNotFoundError(f"Dossier Runpod introuvable: {source}")
    if not normalized:
        raise ValueError("Le prefixe S3 Runpod ne peut pas etre vide.")
    excluded = {suffix.lower() for suffix in excluded_suffixes}
    included = set(included_roots) if included_roots is not None else None
    files = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file()
        and path.suffix.lower() not in excluded
        and (
            included is None
            or path.relative_to(source).parts[0] in included
        )
    ]
    if not files:
        raise RunpodJobError(f"Aucun fichier a envoyer depuis {source}")
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(source).as_posix()
        client.upload_file(str(path), bucket, f"{normalized}/{relative}")
        print(
            f"\r[runpod upload] {index}/{len(files)}",
            end="" if index < len(files) else "\n",
            flush=True,
        )
    return len(files)
