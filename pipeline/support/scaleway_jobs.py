from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import boto3
import requests
from botocore.exceptions import ClientError


DEFAULT_FORWARDED_ENV = frozenset(
    {
        "S3_BUCKET_NAME", "S3_REGION", "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "YOUTUBE_API_KEY", "OPENAI_API_KEY", "OPENAI_SERVICE_TIER",
        "MISTRAL_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY",
        "HUGGINGFACE_TOKEN", "HF_TOKEN", "WHISPERX_MODEL",
        "WHISPERX_LANGUAGE", "WHISPERX_MIN_SPEAKERS",
        "WHISPERX_MAX_SPEAKERS", "WHISPERX_DIARIZATION_MODEL",
        "TRANSCRIPT_RETRIES", "TRANSCRIPT_RETRY_SECONDS",
        "TRANSCRIPT_SLEEP_SECONDS",
    }
)
SAFE_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9._:/@-]+$")


class ScalewayJobError(RuntimeError):
    """Raised when a Scaleway GPU job cannot complete."""


def scaleway_registry_endpoint(container_image: str) -> str:
    parts = container_image.split("/")
    if len(parts) < 3 or not parts[0].startswith("rg.") or not parts[0].endswith(
        ".scw.cloud"
    ):
        raise ValueError(
            "SCALEWAY_CONTAINER_IMAGE doit pointer vers un namespace "
            "Container Registry Scaleway."
        )
    return "/".join(parts[:2])


@dataclass(frozen=True)
class ScalewayConfig:
    secret_key: str
    project_id: str
    container_image: str = ""
    zone: str = "fr-par-2"
    instance_type: str = "L40S-1-48G"
    image_label: str = "ubuntu_noble_gpu_os_13_nvidia"
    root_volume_gb: int = 125
    poll_seconds: float = 10.0
    timeout_seconds: float = 21600.0
    keep_instance: bool = False
    server_id: str = ""
    stop_after_job: bool = True

    @classmethod
    def from_env(cls) -> "ScalewayConfig":
        secret_key = os.getenv("SCW_SECRET_KEY", "").strip()
        project_id = (
            os.getenv("SCW_DEFAULT_PROJECT_ID")
            or os.getenv("SCW_PROJECT_ID")
            or ""
        ).strip()
        container_image = os.getenv("SCALEWAY_CONTAINER_IMAGE", "").strip()
        if not secret_key:
            raise RuntimeError("SCW_SECRET_KEY manquant dans l'environnement.")
        if not project_id:
            raise RuntimeError("SCW_DEFAULT_PROJECT_ID manquant dans l'environnement.")
        if container_image and not SAFE_IMAGE_PATTERN.fullmatch(container_image):
            raise RuntimeError("SCALEWAY_CONTAINER_IMAGE contient des caracteres invalides.")
        server_id = os.getenv("SCALEWAY_SERVER_ID", "").strip()
        if not server_id:
            raise RuntimeError("SCALEWAY_SERVER_ID manquant dans l'environnement.")
        return cls(
            secret_key=secret_key,
            project_id=project_id,
            container_image=container_image,
            zone=os.getenv("SCW_DEFAULT_ZONE", "fr-par-2").strip(),
            instance_type=os.getenv("SCALEWAY_INSTANCE_TYPE", "L40S-1-48G").strip(),
            image_label=os.getenv(
                "SCALEWAY_IMAGE_LABEL", "ubuntu_noble_gpu_os_13_nvidia"
            ).strip(),
            root_volume_gb=int(os.getenv("SCALEWAY_ROOT_VOLUME_GB", "125")),
            poll_seconds=float(os.getenv("SCALEWAY_POLL_SECONDS", "10")),
            timeout_seconds=float(os.getenv("SCALEWAY_JOB_TIMEOUT_SECONDS", "21600")),
            keep_instance=os.getenv("SCALEWAY_KEEP_INSTANCE", "0").strip().lower()
            in {"1", "true", "yes"},
            server_id=server_id,
            stop_after_job=os.getenv("SCALEWAY_STOP_AFTER_JOB", "1").strip().lower()
            in {"1", "true", "yes"},
        )


class ScalewayClient:
    def __init__(self, config: ScalewayConfig, *, session=None, sleep=time.sleep):
        self.config = config
        self.session = session or requests.Session()
        self.sleep = sleep
        self.base_url = (
            f"https://api.scaleway.com/instance/v2alpha1/zones/{config.zone}"
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-Auth-Token": self.config.secret_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs):
        return self._request_url(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            **kwargs,
        )

    def _request_url(self, method: str, url: str, *, headers: dict[str, str], **kwargs):
        response = self.session.request(
            method,
            url,
            headers=headers,
            timeout=30,
            **kwargs,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            status_code = getattr(response, "status_code", "inconnu")
            request_id = str(getattr(response, "headers", {}).get("x-request-id") or "")
            request_suffix = f", requete {request_id}" if request_id else ""
            # Scaleway can echo the submitted cloud-init in an error body. Never
            # include it here because it contains the worker's forwarded secrets.
            raise ScalewayJobError(
                f"API Scaleway: HTTP {status_code}{request_suffix}."
            ) from error
        return response

    def create_server(self, name: str) -> str:
        payload = {
            "project_id": self.config.project_id,
            "name": name,
            "tags": ["rag-ionis", "dedicated-gpu"],
            "server_type": self.config.instance_type,
            "volumes": [
                {
                    "volume_type": "sbs",
                    "new_volume": {
                        "name": f"{name}-root",
                        "size": self.config.root_volume_gb * 1_000_000_000,
                        "image_label": self.config.image_label,
                        "perf_iops": 5000,
                    },
                }
            ],
            "public_network_interface": {
                "ips": [{"new_ip": {"type": "zonal_ipv4", "tags": ["rag-ionis"]}}]
            },
        }
        response = self._request("POST", "/servers", json=payload)
        server_id = str(response.json().get("id") or "").strip()
        if not server_id:
            raise ScalewayJobError("Scaleway n'a renvoye aucun identifiant d'instance.")
        return server_id

    def set_cloud_init(self, server_id: str, content: str) -> None:
        # v2alpha1 represents protobuf `bytes` as base64 in its JSON API.
        self._request(
            "PUT",
            f"/servers/{server_id}/user-data/cloud-init",
            json={
                "content": base64.b64encode(content.encode("utf-8")).decode("ascii")
            },
        )

    def start_server(self, server_id: str) -> None:
        self._request("POST", f"/servers/{server_id}/start", json={})

    def server_status(self, server_id: str) -> str:
        response = self._request("GET", f"/servers/{server_id}")
        return str(response.json().get("status") or "unknown_status").lower()

    def wait_until_ready(self, server_id: str) -> None:
        deadline = time.monotonic() + min(self.config.timeout_seconds, 900.0)
        while True:
            status = self.server_status(server_id)
            if status == "stopped":
                return
            if status == "started":
                raise ScalewayJobError(
                    f"L'instance Scaleway {server_id} a demarre avant son cloud-init."
                )
            if time.monotonic() >= deadline:
                raise ScalewayJobError(
                    f"L'instance Scaleway {server_id} n'est pas devenue disponible."
                )
            self.sleep(max(0.1, self.config.poll_seconds))

    def wait_until_started(self, server_id: str) -> None:
        deadline = time.monotonic() + min(self.config.timeout_seconds, 900.0)
        while True:
            status = self.server_status(server_id)
            if status == "started":
                return
            if status in {"stopped", "paused"}:
                raise ScalewayJobError(
                    f"L'instance Scaleway {server_id} s'est arretee avant le demarrage "
                    "du worker."
                )
            if status not in {"starting", "stopping", "pausing", "locked"}:
                raise ScalewayJobError(
                    f"Etat inattendu de l'instance Scaleway {server_id}: {status}."
                )
            if time.monotonic() >= deadline:
                raise ScalewayJobError(
                    f"L'instance Scaleway {server_id} n'est pas devenue disponible."
                )
            self.sleep(max(0.1, self.config.poll_seconds))

    def ensure_started(self, server_id: str) -> None:
        deadline = time.monotonic() + min(self.config.timeout_seconds, 900.0)
        while True:
            status = self.server_status(server_id)
            if status == "started":
                return
            if status in {"stopped", "paused"}:
                self.start_server(server_id)
                self.wait_until_started(server_id)
                return
            if status == "starting":
                self.wait_until_started(server_id)
                return
            if status not in {"stopping", "pausing", "locked"}:
                raise ScalewayJobError(
                    f"Etat inattendu avant demarrage de l'instance {server_id}: {status}."
                )
            if time.monotonic() >= deadline:
                raise ScalewayJobError(
                    f"Impossible de demarrer l'instance {server_id} dans le delai imparti."
                )
            self.sleep(max(0.1, self.config.poll_seconds))

    def stop_server(self, server_id: str) -> None:
        deadline = time.monotonic() + 300.0
        stop_requested = False
        while True:
            status = self.server_status(server_id)
            if status == "stopped":
                return
            if status in {"started", "paused"} and not stop_requested:
                self._request("POST", f"/servers/{server_id}/stop", json={})
                stop_requested = True
            elif status not in {"starting", "stopping", "pausing", "locked", "started", "paused"}:
                raise ScalewayJobError(
                    f"Etat inattendu pendant l'arret de l'instance {server_id}: {status}."
                )
            if time.monotonic() >= deadline:
                raise ScalewayJobError(
                    f"Impossible d'arreter l'instance {server_id} dans le delai imparti."
                )
            self.sleep(max(0.1, self.config.poll_seconds))

    def destroy_server(self, server_id: str) -> None:
        deadline = time.monotonic() + 300.0
        status = self.server_status(server_id)
        while status in {"starting", "stopping", "pausing", "locked"}:
            if time.monotonic() >= deadline:
                raise ScalewayJobError(
                    f"Impossible de stabiliser l'instance {server_id} avant suppression."
                )
            self.sleep(max(0.1, self.config.poll_seconds))
            status = self.server_status(server_id)
        if status in {"started", "paused"}:
            self._request("POST", f"/servers/{server_id}/stop", json={})
            status = self.server_status(server_id)
            while status != "stopped":
                if status not in {"stopping", "paused"}:
                    raise ScalewayJobError(
                        f"Etat inattendu pendant l'arret de l'instance "
                        f"{server_id}: {status}."
                    )
                if time.monotonic() >= deadline:
                    raise ScalewayJobError(
                        f"Impossible d'arreter l'instance {server_id} avant suppression."
                    )
                self.sleep(max(0.1, self.config.poll_seconds))
                status = self.server_status(server_id)
        if status == "stopped":
            self._request(
                "DELETE",
                f"/servers/{server_id}",
                params={
                    "delete_all_ips": "true",
                    "delete_all_volumes": "true",
                    "keep_all_private_nics": "false",
                },
            )

    def wait_for_job(self, server_id: str, object_store, bucket: str, status_key: str) -> dict:
        deadline = time.monotonic() + self.config.timeout_seconds
        previous_status = None
        while True:
            try:
                response = object_store.get_object(Bucket=bucket, Key=status_key)
            except ClientError as error:
                code = str(error.response.get("Error", {}).get("Code") or "")
                if code not in {"404", "NoSuchKey", "NotFound"}:
                    raise
            else:
                payload = json.loads(response["Body"].read().decode("utf-8"))
                status = str(payload.get("status") or "").lower()
                if status == "completed" and isinstance(payload.get("output"), dict):
                    return payload["output"]
                if status == "failed":
                    raise ScalewayJobError(
                        f"Job Scaleway {server_id} en echec: {payload.get('error')}"
                    )

            instance_status = self.server_status(server_id)
            if instance_status != previous_status:
                print(
                    f"[scaleway] instance={server_id} status={instance_status}",
                    flush=True,
                )
                previous_status = instance_status
            if instance_status in {"stopped", "stopping"}:
                raise ScalewayJobError(
                    f"L'instance Scaleway {server_id} s'est arretee sans resultat."
                )
            if time.monotonic() >= deadline:
                raise ScalewayJobError(
                    f"Delai depasse pour le job Scaleway {server_id} "
                    f"({self.config.timeout_seconds:g} s)."
                )
            self.sleep(max(0.1, self.config.poll_seconds))

    def run(self, job_input: dict, *, object_store, bucket: str) -> tuple[str, dict]:
        control = job_input.get("control")
        if not isinstance(control, dict):
            raise ValueError("Le job Scaleway doit definir son canal de controle S3.")
        status_key = str(control.get("status_key") or "").strip().strip("/")
        if not status_key:
            raise ValueError("Cle de statut S3 Scaleway manquante.")
        job_key = str(control.get("job_key") or "").strip().strip("/")
        if not job_key:
            job_key = f"{status_key.removesuffix('/status.json')}/job.json"
        job_id = str(control.get("job_id") or "").strip()
        if not job_id:
            job_id = status_key.removesuffix("/status.json").rsplit("/", 1)[-1]
        server_id = self.config.server_id.strip()
        if not server_id:
            raise RuntimeError("SCALEWAY_SERVER_ID manquant dans la configuration.")
        try:
            object_store.delete_object(Bucket=bucket, Key=status_key)
        except ClientError:
            pass
        object_store.put_object(
            Bucket=bucket,
            Key=job_key,
            Body=json.dumps(job_input, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        try:
            self.ensure_started(server_id)
            print(
                f"[scaleway] job soumis: job={job_id} instance={server_id} "
                f"type={self.config.instance_type} zone={self.config.zone}",
                flush=True,
            )
            return job_id, self.wait_for_job(
                server_id, object_store, bucket, status_key
            )
        finally:
            if self.config.stop_after_job:
                active_error = sys.exc_info()[1]
                try:
                    self.stop_server(server_id)
                except Exception as cleanup_error:
                    if active_error is None:
                        raise
                    print(
                        f"[scaleway] avertissement arret instance={server_id}: "
                        f"{cleanup_error}",
                        flush=True,
                    )
                else:
                    print(f"[scaleway] instance arretee: {server_id}", flush=True)


def forwarded_environment() -> dict[str, str]:
    extra = {
        name.strip()
        for name in os.getenv("SCALEWAY_FORWARD_ENV", "").split(",")
        if name.strip()
    }
    values: dict[str, str] = {
        "PIPELINE_EXECUTION_BACKEND": "local",
        "SCALEWAY_WORKER": "1",
    }
    for name in sorted(DEFAULT_FORWARDED_ENV | extra):
        value = os.getenv(name)
        if value is None or value == "":
            continue
        if "\n" in value or "\r" in value:
            raise ValueError(f"La variable {name} ne peut pas contenir de saut de ligne.")
        values[name] = value
    return values


def build_cloud_init(job_input: dict, config: ScalewayConfig) -> str:
    job_b64 = base64.b64encode(
        json.dumps(job_input, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    env_text = "\n".join(
        f"{name}={value}" for name, value in forwarded_environment().items()
    ) + "\n"
    env_b64 = base64.b64encode(env_text.encode("utf-8")).decode("ascii")
    registry_endpoint = scaleway_registry_endpoint(config.container_image)
    registry_secret_b64 = base64.b64encode(
        f"{config.secret_key}\n".encode("utf-8")
    ).decode("ascii")
    runner = f"""#!/bin/bash
set -euo pipefail
cleanup() {{
  rm -f /opt/rag-ionis/registry.secret /opt/rag-ionis/worker.env /opt/rag-ionis/job.json
  docker logout {registry_endpoint} >/dev/null 2>&1 || true
  shutdown -h now
}}
trap cleanup EXIT
docker login {registry_endpoint} -u nologin --password-stdin \\
  < /opt/rag-ionis/registry.secret
rm -f /opt/rag-ionis/registry.secret
docker pull {config.container_image}
docker logout {registry_endpoint} >/dev/null 2>&1 || true
docker run --rm --gpus all \\
  --env-file /opt/rag-ionis/worker.env \\
  -v /opt/rag-ionis:/job:ro \\
  {config.container_image} \\
  python3 -u -m pipeline.workers.scaleway_ingestion --job-file /job/job.json
"""
    runner_b64 = base64.b64encode(runner.encode("utf-8")).decode("ascii")
    return f"""#cloud-config
write_files:
  - path: /opt/rag-ionis/job.json
    permissions: '0600'
    encoding: b64
    content: {job_b64}
  - path: /opt/rag-ionis/worker.env
    permissions: '0600'
    encoding: b64
    content: {env_b64}
  - path: /opt/rag-ionis/registry.secret
    permissions: '0600'
    encoding: b64
    content: {registry_secret_b64}
  - path: /opt/rag-ionis/run-job.sh
    permissions: '0700'
    encoding: b64
    content: {runner_b64}
runcmd:
  - [bash, /opt/rag-ionis/run-job.sh]
final_message: "rag-ionis Scaleway GPU job finished"
"""


def s3_client(region: str | None = None):
    access_key = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    options = {}
    if region:
        options["region_name"] = region
    if endpoint_url:
        options["endpoint_url"] = endpoint_url
    if access_key and secret_key:
        options["aws_access_key_id"] = access_key
        options["aws_secret_access_key"] = secret_key
    if session_token:
        options["aws_session_token"] = session_token
    return boto3.client("s3", **options)


def download_s3_prefix(client, bucket: str, prefix: str, destination: Path) -> int:
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
        raise ScalewayJobError(f"Aucun artefact trouve dans s3://{bucket}/{normalized}/")
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
        raise FileNotFoundError(f"Dossier Scaleway introuvable: {source}")
    if not normalized:
        raise ValueError("Le prefixe S3 Scaleway ne peut pas etre vide.")
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
        raise ScalewayJobError(f"Aucun fichier a envoyer depuis {source}")
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(source).as_posix()
        client.upload_file(str(path), bucket, f"{normalized}/{relative}")
        print(
            f"\r[scaleway upload] {index}/{len(files)}",
            end="" if index < len(files) else "\n",
            flush=True,
        )
    return len(files)
