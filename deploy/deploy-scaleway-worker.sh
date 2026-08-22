#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: sudo %s <full-image-reference-with-immutable-tag>\n' "$0" >&2
  exit 2
fi
if [[ "${EUID}" -ne 0 ]]; then
  printf 'This deployment script must run as root.\n' >&2
  exit 1
fi

image="$1"
env_file="${SCALEWAY_WORKER_ENV_FILE:-/etc/rag-ionis/scaleway-worker.env}"
registry_secret_file="${SCALEWAY_REGISTRY_SECRET_FILE:-/etc/rag-ionis/registry.secret}"
drain_timeout="${SCALEWAY_DEPLOY_DRAIN_TIMEOUT_SECONDS:-21600}"
start_worker="${SCALEWAY_DEPLOY_START_WORKER:-1}"

if [[ "${image}" != */*:* || "${image}" == *:latest ]]; then
  printf 'A full image reference with an immutable tag is required: %s\n' "${image}" >&2
  exit 2
fi
if [[ ! -f "${env_file}" ]]; then
  printf 'Worker environment file not found: %s\n' "${env_file}" >&2
  exit 1
fi
if [[ ! "${drain_timeout}" =~ ^[0-9]+$ || "${drain_timeout}" -lt 1 ]]; then
  printf 'SCALEWAY_DEPLOY_DRAIN_TIMEOUT_SECONDS must be a positive integer.\n' >&2
  exit 2
fi
if [[ "${start_worker}" != "0" && "${start_worker}" != "1" ]]; then
  printf 'SCALEWAY_DEPLOY_START_WORKER must be 0 or 1.\n' >&2
  exit 2
fi

get_env_value() {
  local key="$1"
  awk -F= -v wanted="${key}" '$1 == wanted {sub(/^[^=]*=/, ""); sub(/\r$/, ""); print; exit}' "${env_file}"
}

set_env_value() {
  local key="$1"
  local value="$2"
  local temporary_file
  temporary_file="$(mktemp "${env_file}.tmp.XXXXXX")"
  awk -v wanted="${key}" -v replacement="${value}" '
    BEGIN { found = 0 }
    index($0, wanted "=") == 1 {
      print wanted "=" replacement
      found = 1
      next
    }
    { print }
    END {
      if (!found) print wanted "=" replacement
    }
  ' "${env_file}" >"${temporary_file}"
  chmod --reference="${env_file}" "${temporary_file}"
  mv -- "${temporary_file}" "${env_file}"
}

# Let the current worker finish naturally. It exits after its queue becomes idle,
# so a deployment never kills a transcription already in progress.
deadline=$((SECONDS + drain_timeout))
while true; do
  service_state="$(systemctl is-active rag-ionis-scaleway-worker.service 2>/dev/null || true)"
  if [[ "${service_state}" == "inactive" || "${service_state}" == "failed" || "${service_state}" == "unknown" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    printf 'The GPU worker is still busy after %s seconds; deployment aborted.\n' "${drain_timeout}" >&2
    exit 75
  fi
  printf 'Waiting for the current GPU worker to become idle...\n'
  sleep 10
done

registry_endpoint="$(printf '%s' "${image}" | cut -d/ -f1-2)"
logged_in=0
logout_registry() {
  if [[ "${logged_in}" == "1" ]]; then
    docker logout "${registry_endpoint}" >/dev/null 2>&1 || true
  fi
}
trap logout_registry EXIT

if [[ -f "${registry_secret_file}" ]]; then
  docker login "${registry_endpoint}" -u nologin \
    --password-stdin <"${registry_secret_file}"
  logged_in=1
fi

docker pull "${image}"
docker run --rm --gpus all --entrypoint python3 "${image}" -c \
  'import paddle, torch, whisperx; assert torch.cuda.is_available(); paddle.set_device("gpu:0"); print(torch.cuda.get_device_name(0))'

previous_image="$(get_env_value SCALEWAY_CONTAINER_IMAGE)"
set_env_value SCALEWAY_CONTAINER_IMAGE "${image}"
set_env_value SCALEWAY_WORKER_PULL_IMAGE 0

if [[ "${start_worker}" == "0" ]]; then
  systemctl stop rag-ionis-scaleway-worker.service || true
  printf 'Scaleway worker prepared with %s; it will start on the next VM boot.\n' "${image}"
  exit 0
fi

if systemctl restart rag-ionis-scaleway-worker.service; then
  sleep 3
  if docker ps --filter 'name=^/rag-ionis-scaleway-worker$' --format '{{.ID}}' | grep -q .; then
    printf 'Scaleway worker started with %s\n' "${image}"
    exit 0
  fi
fi

printf 'Worker startup failed; rolling back to %s\n' "${previous_image:-<none>}" >&2
if [[ -n "${previous_image}" ]]; then
  set_env_value SCALEWAY_CONTAINER_IMAGE "${previous_image}"
  systemctl restart rag-ionis-scaleway-worker.service || true
else
  systemctl stop rag-ionis-scaleway-worker.service || true
fi
exit 1
