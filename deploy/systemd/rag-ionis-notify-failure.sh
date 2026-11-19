#!/usr/bin/env bash
# Envoie par e-mail l'échec d'un service systemd et la fin de son journal.
# Appelé par rag-ionis-notify-failure@.service (OnFailure=) avec le nom du service en échec.
set -euo pipefail

unit="${1:?Usage: rag-ionis-notify-failure.sh <unit>}"

# Healthchecks signale l'échec indépendamment de l'envoi SMTP.
/usr/bin/bash /usr/local/bin/rag-ionis-healthcheck.sh "${unit}" fail || true

# Limite l'extrait du journal inclus dans l'alerte.
logs="$(journalctl -u "${unit}" -n 15 --no-pager -o cat 2>/dev/null | tail -c 3000 || true)"

printf '🔴 %s : échec de %s\n%s\n\n%s\n' \
  "$(hostname)" "${unit}" "$(date '+%Y-%m-%d %H:%M %Z')" "${logs:-(journal vide)}" |
  /usr/bin/python3 /usr/local/bin/rag-ionis-email.py "[RAG IONIS] Échec de ${unit}"

printf 'Email alert sent for %s.\n' "${unit}"
