#!/usr/bin/env bash
# Envoie sur Telegram l'échec d'un service systemd et la fin de son journal.
# Appelé par rag-ionis-notify-failure@.service (OnFailure=) avec le nom du service en échec.
set -euo pipefail

unit="${1:?Usage: rag-ionis-notify-failure.sh <unit>}"

# Healthchecks d'abord : l'échec est signalé même si Telegram est cassé.
/usr/bin/bash /usr/local/bin/rag-ionis-healthcheck.sh "${unit}" fail || true

# Fin du journal du dernier lancement, tronquée sous la limite de Telegram.
logs="$(journalctl -u "${unit}" -n 15 --no-pager -o cat 2>/dev/null | tail -c 3000 || true)"

printf '🔴 %s : échec de %s\n%s\n\n%s\n' \
  "$(hostname)" "${unit}" "$(date '+%Y-%m-%d %H:%M %Z')" "${logs:-(journal vide)}" |
  /usr/bin/bash /usr/local/bin/rag-ionis-telegram.sh

printf 'Telegram alert handled for %s.\n' "${unit}"
