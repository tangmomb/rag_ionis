#!/usr/bin/env bash
# Signale à healthchecks.io le début, la réussite ou l'échec d'une tâche systemd.
# Healthchecks alerte quand un ping attendu n'arrive pas : une tâche qui ne tourne
# plus (timer désactivé, VPS éteint à l'heure prévue) ne reste pas
# silencieuse, contrairement à OnFailure= qui ne voit que les échecs.
#
#   rag-ionis-healthcheck.sh <unit> [start|success|fail]
#
# <unit> est le nom systemd (rag-ionis-monitor@quality.service) ; le check
# correspondant a pour slug rag-ionis-monitor-quality (voir deploy/healthchecks/).
# En cas d'échec, la fin du journal est jointe au ping.
#
# La clé de ping vit hors du dépôt, dans /etc/rag-ionis/healthchecks.env (root, 0600) :
#   HEALTHCHECKS_PING_KEY=...
# Sans ce fichier, rien n'est envoyé. Le script ne fait jamais échouer la tâche.
set -uo pipefail

unit="${1:?Usage: rag-ionis-healthcheck.sh <unit> [start|success|fail]}"
event="${2:-success}"
config_file="${RAG_IONIS_HEALTHCHECKS_ENV_FILE:-/etc/rag-ionis/healthchecks.env}"

case "${event}" in
  start) suffix="/start" ;;
  success) suffix="" ;;
  fail) suffix="/fail" ;;
  *) printf 'Unknown healthcheck event: %s\n' "${event}" >&2; exit 0 ;;
esac

# rag-ionis-monitor@quality.service -> rag-ionis-monitor-quality
slug="${unit%.service}"
slug="${slug//@/-}"
[[ "${slug}" =~ ^[a-z0-9-]+$ ]] || { printf 'Invalid healthcheck slug: %s\n' "${slug}" >&2; exit 0; }

if [[ ! -r "${config_file}" ]]; then
  printf 'Healthcheck skipped (%s missing): %s %s\n' "${config_file}" "${slug}" "${event}" >&2
  exit 0
fi
# shellcheck disable=SC1090
source "${config_file}"
if [[ -z "${HEALTHCHECKS_PING_KEY:-}" ]]; then
  printf 'Healthcheck skipped (HEALTHCHECKS_PING_KEY empty): %s %s\n' "${slug}" "${event}" >&2
  exit 0
fi
base_url="${HEALTHCHECKS_PING_URL:-https://hc-ping.com}"

body=""
if [[ "${event}" == "fail" ]]; then
  body="$(journalctl -u "${unit}" -n 30 --no-pager -o cat 2>/dev/null | tail -c 10000 || true)"
fi

# La clé passe par la config de curl sur stdin, pas par la ligne de commande visible dans ps.
curl --silent --show-error --fail --max-time 10 --retry 3 \
  --config - \
  --data-raw "${body}" \
  --output /dev/null \
  <<<"url = \"${base_url}/${HEALTHCHECKS_PING_KEY}/${slug}${suffix}\"" ||
  printf 'Healthcheck ping failed: %s %s\n' "${slug}" "${event}" >&2
exit 0
