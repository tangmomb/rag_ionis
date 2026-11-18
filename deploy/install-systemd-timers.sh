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
  rag-ionis-notify-failure@.service
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

# Alertes Telegram en cas d'échec. Le token reste dans /etc/rag-ionis/telegram.env,
# créé à la main sur le VPS ; sans ce fichier, l'alerte est seulement journalisée.
install -m 0755 "${unit_source_dir}/rag-ionis-notify-failure.sh" /usr/local/bin/rag-ionis-notify-failure.sh
install -d -m 0700 /etc/rag-ionis
# La sauvegarde nocturne est installée hors de ce dépôt : on lui ajoute seulement l'alerte.
if [[ -f /etc/systemd/system/rag-ionis-backup.service ]]; then
  install -d -m 0755 /etc/systemd/system/rag-ionis-backup.service.d
  printf '[Unit]\nOnFailure=rag-ionis-notify-failure@%%n.service\n' \
    > /etc/systemd/system/rag-ionis-backup.service.d/notify-failure.conf
fi

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
  /etc/systemd/system/rag-ionis-notify-failure@.service \
  /etc/systemd/system/rag-ionis-stats.timer \
  /etc/systemd/system/rag-ionis-videos.timer
systemctl enable --now rag-ionis-stats.timer rag-ionis-videos.timer

printf 'RAG IONIS update timers installed and enabled.\n'
