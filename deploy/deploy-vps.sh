#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s <api|updater> <image-tag>\n' "$0" >&2
  exit 2
fi

component="$1"
image_tag="$2"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${RAG_IONIS_PRODUCTION_ENV_FILE:-${repo_root}/.env.production}"
registry_secret_file="${REGISTRY_SECRET_FILE:-/etc/rag-ionis/registry.secret}"
health_timeout="${DEPLOY_HEALTH_TIMEOUT_SECONDS:-180}"
lock_file="${DEPLOY_LOCK_FILE:-${repo_root}/.deploy.lock}"

if [[ ! "${image_tag}" =~ ^[A-Za-z0-9._-]+$ || "${image_tag}" == "latest" ]]; then
  printf 'An immutable image tag is required, got: %s\n' "${image_tag}" >&2
  exit 2
fi
if [[ ! -f "${env_file}" ]]; then
  printf 'Production environment file not found: %s\n' "${env_file}" >&2
  exit 1
fi
if [[ ! "${health_timeout}" =~ ^[0-9]+$ || "${health_timeout}" -lt 1 ]]; then
  printf 'DEPLOY_HEALTH_TIMEOUT_SECONDS must be a positive integer.\n' >&2
  exit 2
fi

case "${component}" in
  api)
    tag_key="API_IMAGE_TAG"
    image_name="rag-ionis-api"
    ;;
  updater)
    tag_key="UPDATER_IMAGE_TAG"
    image_name="rag-ionis-updater"
    ;;
  *)
    printf 'Unknown component: %s (expected api or updater)\n' "${component}" >&2
    exit 2
    ;;
esac

exec 9>"${lock_file}"
if ! flock -n 9; then
  printf 'Another deployment is already running.\n' >&2
  exit 1
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

container_registry="$(get_env_value CONTAINER_REGISTRY)"
if [[ -z "${container_registry}" ]]; then
  printf 'CONTAINER_REGISTRY is missing from %s\n' "${env_file}" >&2
  exit 1
fi
container_registry="${container_registry%/}"
image="${container_registry}/${image_name}:${image_tag}"
previous_tag="$(get_env_value "${tag_key}")"
registry_endpoint="${container_registry}"
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

printf 'Pulling %s\n' "${image}"
docker pull "${image}"
set_env_value "${tag_key}" "${image_tag}"

compose=(
  docker compose
  --project-directory "${repo_root}"
  --file "${repo_root}/docker-compose.yml"
  --file "${repo_root}/docker-compose.prod.yml"
  --env-file "${env_file}"
)

if [[ "${component}" == "updater" ]]; then
  printf 'Updater prepared with immutable tag %s\n' "${image_tag}"
  exit 0
fi

rollback_api() {
  if [[ -n "${previous_tag}" && "${previous_tag}" != "${image_tag}" ]]; then
    printf 'API healthcheck failed; rolling back to %s\n' "${previous_tag}" >&2
    set_env_value "${tag_key}" "${previous_tag}"
    "${compose[@]}" up -d --no-deps api || true
  fi
}

# This is a no-op on an already running stack and makes the first deployment
# self-contained without rebuilding or recreating healthy dependencies.
"${compose[@]}" up -d postgres phoenix
"${compose[@]}" up -d --no-deps api
deadline=$((SECONDS + health_timeout))
while (( SECONDS < deadline )); do
  container_id="$("${compose[@]}" ps -q api)"
  if [[ -n "${container_id}" ]]; then
    health_status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${container_id}")"
    case "${health_status}" in
      healthy)
        "${compose[@]}" up -d --no-deps caddy
        printf 'API %s is healthy.\n' "${image_tag}"
        exit 0
        ;;
      unhealthy|exited|dead)
        break
        ;;
    esac
  fi
  sleep 3
done

"${compose[@]}" logs --tail=100 api >&2 || true
rollback_api
exit 1
