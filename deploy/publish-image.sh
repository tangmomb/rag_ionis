#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: IMAGE_REGISTRY=... %s <api|updater|scaleway> [commit-tag]\n' "$0" >&2
  exit 2
fi

component="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
commit_tag="${2:-$(git -C "${repo_root}" rev-parse --short=7 HEAD)}"

if [[ ! "${commit_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'Invalid commit tag: %s\n' "${commit_tag}" >&2
  exit 2
fi

case "${component}" in
  api)
    dockerfile="Dockerfile.api"
    image_name="rag-ionis-api"
    ;;
  updater)
    dockerfile="Dockerfile.vps"
    image_name="rag-ionis-updater"
    ;;
  scaleway)
    dockerfile="Dockerfile.scaleway"
    image_name="rag-ionis-scaleway"
    ;;
  *)
    printf 'Unknown component: %s (expected api, updater or scaleway)\n' "${component}" >&2
    exit 2
    ;;
esac

if [[ -n "${IMAGE_REPOSITORY:-}" ]]; then
  image_repository="${IMAGE_REPOSITORY%/}"
else
  : "${IMAGE_REGISTRY:?Set IMAGE_REGISTRY, for example rg.fr-par.scw.cloud/NAMESPACE}"
  image_registry="${IMAGE_REGISTRY%/}"
  image_repository="${image_registry}/${image_name}"
fi

if [[ -z "${image_repository}" || "${image_repository}" == *:* ]]; then
  printf 'The image repository must not contain a tag: %s\n' "${image_repository}" >&2
  exit 2
fi

commit_image="${image_repository}:${commit_tag}"
latest_image="${image_repository}:latest"
cache_image="${image_repository}:buildcache"

docker buildx build \
  --progress plain \
  --platform linux/amd64 \
  --file "${repo_root}/${dockerfile}" \
  --tag "${commit_image}" \
  --tag "${latest_image}" \
  --cache-from "type=registry,ref=${cache_image}" \
  --cache-to "type=registry,ref=${cache_image},mode=max" \
  --push \
  "${repo_root}"

printf 'Published %s and %s (cache: %s)\n' \
  "${commit_image}" "${latest_image}" "${cache_image}"
