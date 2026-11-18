#!/usr/bin/env bash
# Envoie sur Telegram le texte lu sur l'entrée standard.
# Le token et le chat vivent hors du dépôt, dans /etc/rag-ionis/telegram.env (root, 0600) :
#   TELEGRAM_BOT_TOKEN=...
#   TELEGRAM_CHAT_ID=...
# Sans ce fichier, le message est seulement écrit dans le journal.
set -euo pipefail

config_file="${RAG_IONIS_TELEGRAM_ENV_FILE:-/etc/rag-ionis/telegram.env}"
# Limite Telegram : 4096 caractères par message.
message="$(head -c 4000)"

if [[ ! -r "${config_file}" ]]; then
  printf 'Telegram skipped (%s missing):\n%s\n' "${config_file}" "${message}" >&2
  exit 0
fi
# shellcheck disable=SC1090
source "${config_file}"
: "${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN is required in ${config_file}}"
: "${TELEGRAM_CHAT_ID:?TELEGRAM_CHAT_ID is required in ${config_file}}"

# Le token passe par la config de curl sur stdin, pas par la ligne de commande visible dans ps.
curl --silent --show-error --fail --max-time 20 --retry 3 \
  --config - \
  --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
  --data-urlencode "text=${message}" \
  --output /dev/null \
  <<<"url = \"https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage\""
