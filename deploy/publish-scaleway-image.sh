#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper around the shared Buildx publisher.
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

IMAGE_REPOSITORY="${image_repository}" \
  bash "${script_dir}/publish-image.sh" scaleway "${commit_tag}"
