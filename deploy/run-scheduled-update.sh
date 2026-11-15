#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <stats|videos>\n' "$0" >&2
  exit 2
fi

update_type="$1"
case "${update_type}" in
  stats)
    module="pipeline.update_stats"
    minimum_interval_seconds=$((3 * 24 * 60 * 60))
    ;;
  videos)
    module="pipeline.update_videos"
    minimum_interval_seconds=0
    ;;
  *)
    printf 'Unknown update type: %s (expected stats or videos)\n' "${update_type}" >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${RAG_IONIS_PRODUCTION_ENV_FILE:-${repo_root}/.env.production}"
state_dir="${RAG_IONIS_UPDATE_STATE_DIR:-/var/lib/rag-ionis-updater}"
lock_file="${RAG_IONIS_UPDATE_LOCK_FILE:-/run/lock/rag-ionis-update.lock}"
success_stamp="${state_dir}/last-${update_type}-success"

[[ -f "${env_file}" ]] || {
  printf 'Production environment file not found: %s\n' "${env_file}" >&2
  exit 1
}

install -d -m 0755 "${state_dir}"
exec 9>"${lock_file}"
flock 9

if (( minimum_interval_seconds > 0 )) && [[ -f "${success_stamp}" ]]; then
  now="$(date +%s)"
  last_success="$(stat -c %Y "${success_stamp}")"
  age=$((now - last_success))
  if (( age < minimum_interval_seconds )); then
    printf '%s update skipped: last successful run was %s seconds ago.\n' \
      "${update_type}" "${age}"
    exit 0
  fi
fi

compose=(
  docker compose
  --project-directory "${repo_root}"
  --file "${repo_root}/docker-compose.yml"
  --file "${repo_root}/docker-compose.prod.yml"
  --env-file "${env_file}"
)

printf 'Starting scheduled %s update.\n' "${update_type}"
"${compose[@]}" --profile jobs run --rm updater python -m "${module}"
touch "${success_stamp}"
printf 'Scheduled %s update completed successfully.\n' "${update_type}"
