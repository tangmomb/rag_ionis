#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <deployment-directory>\n' "$0" >&2
  exit 2
fi

deployment_dir="$1"

# The deployment workflow only accepts a simple directory name. Resolve it here
# before removing legacy files so cleanup can never escape the VPS deployment
# directory.
[[ "${deployment_dir}" =~ ^[A-Za-z0-9_-]+$ ]] || {
  printf 'Invalid deployment directory: %s\n' "${deployment_dir}" >&2
  exit 2
}

deployment_dir="${HOME}/${deployment_dir}"
[[ -d "${deployment_dir}" ]] || {
  printf 'Deployment directory does not exist: %s\n' "${deployment_dir}" >&2
  exit 1
}

# Application code is built into immutable container images. Keep only runtime
# configuration, deployment scripts, and the server-managed .env.production.
legacy_paths=(
  .git
  .gitignore
  Dockerfile.api
  Dockerfile.scaleway
  Dockerfile.vps
  README.md
  interface
  models
  pipeline
  requirements-api.txt
  requirements-gpu.txt
  requirements-paddle-bootstrap.txt
  requirements-vps.txt
  requirements.txt
  start_app_local.bat
  tests
  utils
)

for legacy_path in "${legacy_paths[@]}"; do
  rm -rf -- "${deployment_dir:?}/${legacy_path}"
done

printf 'Deployment directory synchronized: %s\n' "${deployment_dir}"
