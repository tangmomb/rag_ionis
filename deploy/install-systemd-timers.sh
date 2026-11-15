#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <deployment-directory>\n' "$0" >&2
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  printf 'This installer must run as root.\n' >&2
  exit 1
fi

deployment_dir="$1"
[[ "${deployment_dir}" =~ ^/srv/[A-Za-z0-9_-]+$ ]] || {
  printf 'Invalid deployment directory: %s\n' "${deployment_dir}" >&2
  exit 2
}

unit_source_dir="${deployment_dir}/deploy/systemd"
units=(
  rag-ionis-update@.service
  rag-ionis-stats.timer
  rag-ionis-videos.timer
)

for unit in "${units[@]}"; do
  [[ -f "${unit_source_dir}/${unit}" ]] || {
    printf 'Missing systemd unit: %s\n' "${unit_source_dir}/${unit}" >&2
    exit 1
  }
  install -m 0644 "${unit_source_dir}/${unit}" "/etc/systemd/system/${unit}"
done

install -d -m 0755 /var/lib/rag-ionis-updater

# A persistent timer otherwise considers its first, already elapsed calendar
# event missed and can run immediately during deployment. Seed only new timers;
# existing timer state is kept so genuine downtime is still caught up.
install -d -m 0755 /var/lib/systemd/timers
for timer in rag-ionis-stats.timer rag-ionis-videos.timer; do
  stamp="/var/lib/systemd/timers/stamp-${timer}"
  [[ -e "${stamp}" ]] || touch "${stamp}"
done

systemctl daemon-reload
systemd-analyze verify \
  /etc/systemd/system/rag-ionis-update@.service \
  /etc/systemd/system/rag-ionis-stats.timer \
  /etc/systemd/system/rag-ionis-videos.timer
systemctl enable --now rag-ionis-stats.timer rag-ionis-videos.timer

printf 'RAG IONIS update timers installed and enabled.\n'
