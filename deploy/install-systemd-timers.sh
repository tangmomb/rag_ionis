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
  rag-ionis-monitor@.service
  rag-ionis-stats.timer
  rag-ionis-videos.timer
  rag-ionis-quality.timer
  rag-ionis-recall.timer
  rag-ionis-restore.timer
)
timers=(
  rag-ionis-stats.timer
  rag-ionis-videos.timer
  rag-ionis-quality.timer
  rag-ionis-recall.timer
  rag-ionis-restore.timer
)

for unit in "${units[@]}"; do
  [[ -f "${unit_source_dir}/${unit}" ]] || {
    printf 'Missing systemd unit: %s\n' "${unit_source_dir}/${unit}" >&2
    exit 1
  }
  install -m 0644 "${unit_source_dir}/${unit}" "/etc/systemd/system/${unit}"
done

quality_drop_in=rag-ionis-monitor@quality.service.d/timeout.conf
install -D -m 0644 "${unit_source_dir}/${quality_drop_in}" "/etc/systemd/system/${quality_drop_in}"

install -d -m 0755 /var/lib/rag-ionis-updater /var/lib/rag-ionis-monitoring

# Les réglages SMTP restent dans /etc/rag-ionis/smtp.ini, créé à la main sur le VPS.
# Pings healthchecks.io : la clé reste dans /etc/rag-ionis/healthchecks.env, créé à
# la main ; sans ce fichier, les pings sont seulement journalisés.
for script in rag-ionis-email.py rag-ionis-notify-failure.sh rag-ionis-healthcheck.sh; do
  install -m 0755 "${unit_source_dir}/${script}" "/usr/local/bin/${script}"
done
install -d -m 0700 /etc/rag-ionis
# La sauvegarde nocturne est installée hors de ce dépôt : on lui ajoute seulement
# l'alerte et les pings healthchecks (service oneshot, comme les autres tâches).
if [[ -f /etc/systemd/system/rag-ionis-backup.service ]]; then
  install -d -m 0755 /etc/systemd/system/rag-ionis-backup.service.d
  printf '[Unit]\nOnFailure=rag-ionis-notify-failure@%%n.service\n' \
    > /etc/systemd/system/rag-ionis-backup.service.d/notify-failure.conf
  # Seuls les pings tournent en root pour lire la clé, pas la sauvegarde.
  printf '[Service]\nExecStartPre=-+/usr/bin/bash /usr/local/bin/rag-ionis-healthcheck.sh %%n start\nExecStartPost=-+/usr/bin/bash /usr/local/bin/rag-ionis-healthcheck.sh %%n success\n' \
    > /etc/systemd/system/rag-ionis-backup.service.d/healthcheck.conf
fi

# A persistent timer otherwise considers its first, already elapsed calendar
# event missed and can run immediately during deployment. Seed only new timers;
# existing timer state is kept so genuine downtime is still caught up.
install -d -m 0755 /var/lib/systemd/timers
for timer in "${timers[@]}"; do
  stamp="/var/lib/systemd/timers/stamp-${timer}"
  [[ -e "${stamp}" ]] || touch "${stamp}"
done

systemctl daemon-reload
installed_units=()
for unit in "${units[@]}"; do
  installed_units+=("/etc/systemd/system/${unit}")
done
systemd-analyze verify "${installed_units[@]}"
systemctl enable --now "${timers[@]}"

printf 'RAG IONIS update and monitoring timers installed and enabled.\n'
