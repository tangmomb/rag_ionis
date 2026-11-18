#!/usr/bin/env bash
# Envoie sur Telegram l'échec d'un service systemd et la fin de son journal.
# Appelé par rag-ionis-notify-failure@.service (OnFailure=) avec le nom du service en échec.
# Le token et le chat vivent hors du dépôt, dans /etc/rag-ionis/telegram.env (root, 0600) :
#   TELEGRAM_BOT_TOKEN=...
#   TELEGRAM_CHAT_ID=...
set -euo pipefail

unit="${1:?Usage: rag-ionis-notify-failure.sh <unit>}"
config_file="${RAG_IONIS_TELEGRAM_ENV_FILE:-/etc/rag-ionis/telegram.env}"

if [[ ! -r "${config_file}" ]]; then
  printf 'Telegram alert skipped for %s: %s is missing.\n' "${unit}" "${config_file}" >&2
  exit 0
fi
# shellcheck disable=SC1090
source "${config_file}"
: "${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN is required in ${config_file}}"
: "${TELEGRAM_CHAT_ID:?TELEGRAM_CHAT_ID is required in ${config_file}}"

# Fin du journal du dernier lancement, tronquée sous la limite de 4096 caractères de Telegram.
logs="$(journalctl -u "${unit}" -n 15 --no-pager -o cat 2>/dev/null | tail -c 3000 || true)"
message="🔴 $(hostname) : échec de ${unit}
$(date '+%Y-%m-%d %H:%M %Z')

${logs:-(journal vide)}"

# Le token passe par la config de curl sur stdin, pas par la ligne de commande visible dans ps.
curl --silent --show-error --fail --max-time 20 --retry 3 \
  --config - \
  --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
  --data-urlencode "text=${message}" \
  --output /dev/null \
  <<<"url = \"https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage\""

printf 'Telegram alert sent for %s.\n' "${unit}"
