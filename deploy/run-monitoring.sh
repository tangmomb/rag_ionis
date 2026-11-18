#!/usr/bin/env bash
# Tâches de surveillance de la qualité, lancées par rag-ionis-monitor@.service :
#   quality : canari, LLM juge et rapport de la veille (chaque matin)
#   recall  : recall du golden dataset sur la prod (le 1er et le 15 du mois)
#   restore : restauration de la dernière sauvegarde dans une base jetable (chaque lundi)
# Chaque tâche envoie son résultat sur Telegram et l'enregistre dans la table
# monitoring.reports (deploy/record-report.sh). Un crash fait échouer le service,
# ce qui déclenche l'alerte OnFailure.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <quality|recall|restore>\n' "$0" >&2
  exit 2
fi

task="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${RAG_IONIS_PRODUCTION_ENV_FILE:-${repo_root}/.env.production}"
state_dir="${RAG_IONIS_MONITORING_STATE_DIR:-/var/lib/rag-ionis-monitoring}"
telegram="${RAG_IONIS_TELEGRAM_SCRIPT:-/usr/local/bin/rag-ionis-telegram.sh}"

[[ -f "${env_file}" ]] || {
  printf 'Production environment file not found: %s\n' "${env_file}" >&2
  exit 1
}
install -d -m 0755 "${state_dir}"
cd "${repo_root}"
compose=(docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file "${env_file}")
# Conteneur ponctuel : il partage le réseau de la prod (postgres, phoenix) sans
# publier de port ni redémarrer les services dont dépend l'API.
run_api=("${compose[@]}" run --rm --no-deps -T)
workdir="$(mktemp -d)"
cleanup() { rm -rf "${workdir}"; }
trap cleanup EXIT

# Dernière ligne JSON d'une sortie, vide s'il n'y en a pas.
last_json() {
  python3 -c '
import json, sys
for line in reversed(sys.stdin.read().splitlines()):
    line = line.strip()
    if line.startswith("{"):
        try:
            json.loads(line)
        except ValueError:
            continue
        print(line)
        break
'
}

# Champ d'un objet JSON lu sur stdin ; les objets sont réécrits en JSON.
json_field() {
  python3 -c '
import json, sys
value = json.load(sys.stdin).get(sys.argv[1])
print(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else ("" if value is None else value))
' "$1"
}

# Envoie le message sur Telegram et enregistre le rapport. Un échec d'enregistrement
# est signalé sans empêcher l'alerte.
publish() {
  local kind="$1" status="$2" metrics="$3" message="$4"
  printf '%s\n' "${metrics}" >"${workdir}/metrics.json"
  printf '%s\n' "${message}" >"${workdir}/message.txt"
  printf '%s\n' "${message}" | /usr/bin/bash "${telegram}"
  /usr/bin/bash "${script_dir}/record-report.sh" "${kind}" "${status}" \
    "${workdir}/metrics.json" "${workdir}/message.txt" ||
    printf 'Warning: %s report not recorded in monitoring.reports.\n' "${kind}" >&2
}

record_failure() {
  local kind="$1" exit_code="$2"
  printf '{"exit_code": %s}\n' "${exit_code}" >"${workdir}/metrics.json"
  /usr/bin/bash "${script_dir}/record-report.sh" "${kind}" failed "${workdir}/metrics.json" || true
}

case "${task}" in
  quality)
    set +e
    output="$("${run_api[@]}" -e PHOENIX_PROJECT_NAME=rag-ionis-canary \
      api python -m interface.monitoring.quality daily)"
    status=$?
    set -e
    printf '%s\n' "${output}"
    result="$(printf '%s\n' "${output}" | last_json)"
    if [[ ${status} -ne 0 || -z "${result}" ]]; then
      printf 'Quality run failed (exit %s, no result).\n' "${status}" >&2
      record_failure quality "${status}"
      exit 1
    fi
    anomaly="$(printf '%s\n' "${result}" | json_field anomaly)"
    publish quality "$([[ "${anomaly}" == "True" ]] && echo anomaly || echo ok)" \
      "$(printf '%s\n' "${result}" | json_field metrics)" \
      "$(printf '%s\n' "${result}" | json_field message)"
    ;;
  recall)
    baseline="$(cat .golden-recall-baseline 2>/dev/null || true)"
    baseline_args=()
    if [[ "${baseline}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
      baseline_args=(--baseline-recall "${baseline}")
    fi
    # Recherche seule, sans trace Phoenix : la mesure ne pollue pas les métriques de prod.
    set +e
    output="$("${run_api[@]}" -e PHOENIX_ENABLED=false \
      api python -m interface.backend.golden_recall "${baseline_args[@]}")"
    status=$?
    set -e
    printf '%s\n' "${output}"
    result="$(printf '%s\n' "${output}" | last_json)"
    if [[ -z "${result}" ]]; then
      printf 'Recall run failed (exit %s, no result).\n' "${status}" >&2
      record_failure recall "${status}"
      exit 1
    fi
    # Copie locale, une ligne par mesure (la référence est monitoring.reports).
    printf '%s\n' "${result}" | python3 -c '
import json, sys, datetime
row = json.load(sys.stdin)
row["measured_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
print(json.dumps(row, ensure_ascii=False))
' >>"${state_dir}/recall-history.jsonl"
    # golden_recall sort en erreur si le plancher ou la baisse maximale est franchi.
    summary="$(printf '%s\n' "${result}" | python3 -c '
import json, sys
row = json.load(sys.stdin)
ok, baseline = sys.argv[1] == "0", sys.argv[2]
fmt = lambda v: "—" if v is None else format(v, ".1%")
video, chunk = fmt(row.get("video_recall")), fmt(row.get("chunk_recall"))
reference = " (référence " + fmt(float(baseline)) + ")" if baseline else ""
measured = row.get("measured_case_count")
failed = len(row.get("failed_cases") or [])
lines = [
    ("🟢" if ok else "🔴") + " Recall prod RAG IONIS",
    f"Vidéos : {video}{reference}",
    f"Chunks : {chunk}",
    f"{measured} questions mesurées, {failed} en échec",
]
if not ok:
    lines.append("Plancher ou baisse maximale franchi : voir journalctl -u rag-ionis-monitor@recall")
row["baseline_recall"] = float(baseline) if baseline else None
print(json.dumps({"status": "ok" if ok else "anomaly", "message": "\n".join(lines), "metrics": row}, ensure_ascii=False))
' "${status}" "${baseline}")"
    publish recall "$(printf '%s\n' "${summary}" | json_field status)" \
      "$(printf '%s\n' "${summary}" | json_field metrics)" \
      "$(printf '%s\n' "${summary}" | json_field message)"
    ;;
  restore)
    # Restaure la dernière sauvegarde S3 dans un PostgreSQL jetable, compare son
    # contenu à la prod et vérifie la base Phoenix : une sauvegarde n'est prouvée
    # que restaurée. Mesure aussi la durée de reprise (RTO).
    set -a
    # shellcheck disable=SC1090
    . "${env_file}"
    set +a
    export RCLONE_CONFIG=/dev/null RCLONE_CONFIG_SCW_TYPE=s3 RCLONE_CONFIG_SCW_PROVIDER=Scaleway \
      RCLONE_CONFIG_SCW_ACCESS_KEY_ID="${S3_ACCESS_KEY_ID}" \
      RCLONE_CONFIG_SCW_SECRET_ACCESS_KEY="${S3_SECRET_ACCESS_KEY}" \
      RCLONE_CONFIG_SCW_REGION="${S3_REGION}" RCLONE_CONFIG_SCW_ENDPOINT="${S3_ENDPOINT_URL}" \
      RCLONE_CONFIG_SCW_NO_CHECK_BUCKET=true
    prefix="scw:${S3_BUCKET_NAME}/backups/rag_ionis"
    latest="$(rclone lsf --dirs-only "${prefix}" | sort | tail -1)"
    latest="${latest%/}"
    [[ -n "${latest}" ]] || { printf 'No backup found under %s\n' "${prefix}" >&2; exit 1; }

    container="rag-ionis-restore-test"
    cleanup() { docker rm -f "${container}" >/dev/null 2>&1 || true; rm -rf "${workdir}"; }
    trap cleanup EXIT
    restore_test_started=$SECONDS
    docker rm -f "${container}" >/dev/null 2>&1 || true
    # Même image que la prod, mémoire plafonnée : le VPS est partagé.
    docker run -d --name "${container}" --memory 512m --memory-swap 512m \
      -e POSTGRES_PASSWORD=restore-test -e POSTGRES_DB=restore \
      pgvector/pgvector:0.8.2-pg16 >/dev/null
    for _ in $(seq 60); do
      docker exec "${container}" pg_isready -h 127.0.0.1 -U postgres -d restore >/dev/null 2>&1 && break
      sleep 1
    done
    rclone copyto "${prefix}/${latest}/postgres.dump" "${workdir}/postgres.dump"
    rclone copyto "${prefix}/${latest}/phoenix.db" "${workdir}/phoenix.db"
    restore_started=$SECONDS
    # Sans propriétaires ni droits : les rôles de la prod n'existent pas ici.
    set +e
    docker exec -i "${container}" pg_restore --no-owner --no-privileges --exit-on-error \
      -U postgres -d restore <"${workdir}/postgres.dump"
    restore_status=$?
    set -e
    restore_seconds=$((SECONDS - restore_started))

    count_sql="SELECT (SELECT count(*) FROM data.videos) || ' ' || (SELECT count(*) FROM data.chunks) || ' ' || (SELECT count(*) FROM data.transcripts) || ' ' || (SELECT count(*) FROM chat.messages)"
    restored_counts="$(docker exec "${container}" psql -At -U postgres -d restore -c "${count_sql}" 2>/dev/null || true)"
    prod_counts="$("${compose[@]}" exec -T postgres sh -c 'psql -At -U "$POSTGRES_USER" "$POSTGRES_DB" -c "$1"' sh "${count_sql}" </dev/null)"
    phoenix_check="$(python3 - "${workdir}/phoenix.db" <<'PY'
import sqlite3, sys
try:
    connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    spans = connection.execute("SELECT count(*) FROM spans").fetchone()[0]
    print(f"{integrity} {spans}")
except Exception as exc:
    print(f"error {type(exc).__name__}")
PY
)"
    total_seconds=$((SECONDS - restore_test_started))

    summary="$(python3 - "${latest}" "${restore_status}" "${restored_counts}" "${prod_counts}" \
      "${phoenix_check}" "${restore_seconds}" "${total_seconds}" <<'PY'
import json, sys
from datetime import datetime, timezone

stamp, status, restored, prod, phoenix, restore_s, total_s = sys.argv[1:8]
names = ("vidéos", "chunks", "transcriptions", "messages")
keys = ("videos", "chunks", "transcripts", "messages")
restored_values = [int(v) for v in restored.split()] if len(restored.split()) == 4 else []
prod_values = [int(v) for v in prod.split()]
phoenix_ok = phoenix.split()[0] == "ok"
problems = []
if status != "0":
    problems.append(f"pg_restore a échoué (code {status})")
if not restored_values:
    problems.append("tables introuvables après restauration")
else:
    # The backup is from the night: allow content added since, not missing content.
    for name, saved, live in zip(names[:3], restored_values, prod_values):
        if saved == 0 or saved < 0.9 * live:
            problems.append(f"{name} : {saved} restaurés pour {live} en prod")
if not phoenix_ok:
    problems.append(f"base Phoenix invalide ({phoenix})")
age_h = (datetime.now(timezone.utc) - datetime.strptime(stamp, "%Y-%m-%dT%H%MZ").replace(tzinfo=timezone.utc)).total_seconds() / 3600
if age_h > 36:
    problems.append(f"dernière sauvegarde vieille de {age_h:.0f} h")
phoenix_spans = int(phoenix.split()[1]) if phoenix_ok else None
lines = [
    ("🔴" if problems else "🟢") + " Test de restauration RAG IONIS",
    f"Sauvegarde {stamp} (il y a {age_h:.0f} h)",
]
if restored_values:
    lines.append(", ".join(f"{v} {n}" for n, v in zip(names, restored_values)))
lines.append(f"Phoenix : {phoenix_spans if phoenix_ok else '—'} spans, intégrité {phoenix.split()[0]}")
lines.append(f"Restauration Postgres : {restore_s} s · reprise complète : {total_s} s")
lines += problems
metrics = {
    "backup": stamp,
    "backup_age_hours": round(age_h, 1),
    "restored": dict(zip(keys, restored_values)),
    "production": dict(zip(keys, prod_values)),
    "phoenix_integrity": phoenix.split()[0],
    "phoenix_spans": phoenix_spans,
    "restore_seconds": int(restore_s),
    "total_seconds": int(total_s),
    "problems": problems,
}
print(json.dumps({"status": "anomaly" if problems else "ok", "message": "\n".join(lines), "metrics": metrics}, ensure_ascii=False))
PY
)"
    publish restore "$(printf '%s\n' "${summary}" | json_field status)" \
      "$(printf '%s\n' "${summary}" | json_field metrics)" \
      "$(printf '%s\n' "${summary}" | json_field message)"
    ;;
  *)
    printf 'Unknown monitoring task: %s (expected quality, recall or restore)\n' "${task}" >&2
    exit 2
    ;;
esac
