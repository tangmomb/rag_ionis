#!/usr/bin/env bash
set -euo pipefail

# Publishes the same image under an immutable commit tag and the moving tag
# consumed by the persistent Scaleway VM.
: "${SCALEWAY_IMAGE_REPOSITORY:?Set SCALEWAY_IMAGE_REPOSITORY, for example rg.fr-par.scw.cloud/NAMESPACE/rag-ionis-scaleway}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ $# -gt 1 ]]; then
  printf 'Usage: SCALEWAY_IMAGE_REPOSITORY=... %s [commit-tag]\n' "$0" >&2
  exit 2
fi

commit_tag="${1:-$(git -C "${repo_root}" rev-parse --short=7 HEAD)}"
if [[ ! "${commit_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'Invalid commit tag: %s\n' "${commit_tag}" >&2
  exit 2
fi

image_repository="${SCALEWAY_IMAGE_REPOSITORY%/}"
if [[ -z "${image_repository}" || "${image_repository}" == *:* ]]; then
  printf 'SCALEWAY_IMAGE_REPOSITORY must be an image repository without a tag.\n' >&2
  exit 2
fi

commit_image="${image_repository}:${commit_tag}"
latest_image="${image_repository}:latest"

docker build \
  --platform linux/amd64 \
  -f "${repo_root}/Dockerfile.scaleway" \
  -t "${commit_image}" \
  -t "${latest_image}" \
  "${repo_root}"

docker push "${commit_image}"
docker push "${latest_image}"

printf 'Published %s and %s\n' "${commit_image}" "${latest_image}"
