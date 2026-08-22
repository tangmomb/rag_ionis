#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE_REGISTRY:?Set IMAGE_REGISTRY, for example rg.fr-par.scw.cloud/NAMESPACE}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ $# -gt 1 ]]; then
  printf 'Usage: IMAGE_REGISTRY=... %s [commit-tag]\n' "$0" >&2
  exit 2
fi

commit_tag="${1:-$(git -C "${repo_root}" rev-parse --short=7 HEAD)}"

for component in api updater scaleway; do
  IMAGE_REGISTRY="${IMAGE_REGISTRY}" \
    bash "${script_dir}/publish-image.sh" "${component}" "${commit_tag}"
done
