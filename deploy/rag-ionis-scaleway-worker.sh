#!/usr/bin/env bash
set -euo pipefail

: "${SCALEWAY_CONTAINER_IMAGE:?SCALEWAY_CONTAINER_IMAGE must be set by systemd}"
worker_env_file="${SCALEWAY_WORKER_ENV_FILE:-/etc/rag-ionis/scaleway-worker.env}"
registry_secret_file="${SCALEWAY_REGISTRY_SECRET_FILE:-/etc/rag-ionis/registry.secret}"
registry_endpoint="$(printf '%s' "${SCALEWAY_CONTAINER_IMAGE}" | cut -d/ -f1-2)"

login_registry() {
  if [[ -f "${registry_secret_file}" ]]; then
    /usr/bin/docker login "${registry_endpoint}" -u nologin \
      --password-stdin < "${registry_secret_file}"
  fi
}

logout_registry() {
  if [[ -f "${registry_secret_file}" ]]; then
    /usr/bin/docker logout "${registry_endpoint}" >/dev/null 2>&1 || true
  fi
}

if [[ "${SCALEWAY_WORKER_PULL_IMAGE:-0}" == "1" ]]; then
  login_registry
  /usr/bin/docker pull "${SCALEWAY_CONTAINER_IMAGE}"
  logout_registry
elif ! /usr/bin/docker image inspect "${SCALEWAY_CONTAINER_IMAGE}" >/dev/null 2>&1; then
  login_registry
  /usr/bin/docker pull "${SCALEWAY_CONTAINER_IMAGE}"
  logout_registry
fi

exec /usr/bin/docker run \
  --rm \
  --name rag-ionis-scaleway-worker \
  --gpus all \
  --env-file "${worker_env_file}" \
  -e SCALEWAY_WORKER=1 \
  -e PIPELINE_EXECUTION_BACKEND=local \
  "${SCALEWAY_CONTAINER_IMAGE}" \
  python3 -u -m pipeline.workers.scaleway_ingestion --poll
