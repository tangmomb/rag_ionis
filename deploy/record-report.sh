#!/usr/bin/env bash
# Enregistre un rapport de surveillance dans la table monitoring.reports de la base
# de prod, sauvegardée chaque nuit avec le reste et vérifiée par le test de restauration.
#
#   record-report.sh <kind> <status> <metrics.json> [message.txt]
#
#   kind    : quality | recall | recall_gate | restore
#   status  : ok | anomaly | failed
#   metrics : fichier JSON (objet) des chiffres du rapport
#   message : fichier texte du message envoyé, facultatif
#
# Les valeurs passent par des variables psql (:'var'), jamais concaténées dans le SQL.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: %s <kind> <status> <metrics.json> [message.txt]\n' "$0" >&2
  exit 2
fi
kind="$1"
status="$2"
metrics_file="$3"
message_file="${4:-}"
[[ "${kind}" =~ ^(quality|recall|recall_gate|restore)$ ]] || { printf 'Unknown report kind: %s\n' "${kind}" >&2; exit 2; }
[[ "${status}" =~ ^(ok|anomaly|failed)$ ]] || { printf 'Unknown report status: %s\n' "${status}" >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${RAG_IONIS_PRODUCTION_ENV_FILE:-${repo_root}/.env.production}"
cd "${repo_root}"
POSTGRES_USER="$(sed -n 's/^POSTGRES_USER=//p' "${env_file}" | tail -1 | tr -d '\r"')"
POSTGRES_DB="$(sed -n 's/^POSTGRES_DB=//p' "${env_file}" | tail -1 | tr -d '\r"')"

metrics="$(cat "${metrics_file}")"
message=""
[[ -n "${message_file}" && -f "${message_file}" ]] && message="$(cat "${message_file}")"

docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file "${env_file}" \
  exec -T postgres psql -v ON_ERROR_STOP=1 -q -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
  -v kind="${kind}" -v status="${status}" -v metrics="${metrics}" -v message="${message}" \
  -f - <<'SQL'
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE TABLE IF NOT EXISTS monitoring.reports (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_monitoring_reports_kind_created
    ON monitoring.reports (kind, created_at);
INSERT INTO monitoring.reports (kind, status, metrics, message)
VALUES (:'kind', :'status', :'metrics'::jsonb, NULLIF(:'message', ''));
SQL
printf 'Report recorded: %s (%s).\n' "${kind}" "${status}"
