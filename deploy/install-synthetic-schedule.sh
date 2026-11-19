#!/usr/bin/env bash
set -euo pipefail

cd /srv/rag_ionis_synthetic

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check \
  mistralai==2.9.0 python-dotenv==1.2.2 requests==2.34.2 numpy==2.2.6

timers=(
  rag-ionis-synthetic-night.timer
  rag-ionis-synthetic-day.timer
  rag-ionis-synthetic-followup-batch.timer
  rag-ionis-synthetic-followup-send.timer
)
units=(rag-ionis-synthetic@.service "${timers[@]}")
sources=("${units[@]/#/deploy/systemd/}")
systemd-analyze verify "${sources[@]}"
for unit in "${units[@]}"; do
  sudo -n install -m 0644 "deploy/systemd/$unit" "/etc/systemd/system/$unit"
done

sudo -n systemctl daemon-reload
for timer in "${timers[@]}"; do
  # Arm a new timer at installation without generating an extra daytime run.
  stamp="/var/lib/systemd/timers/stamp-$timer"
  if ! sudo -n test -e "$stamp"; then
    sudo -n touch "$stamp"
  fi
done
sudo -n systemctl enable --now "${timers[@]}"
