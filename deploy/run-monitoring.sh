#!/usr/bin/env bash
# Tâches de surveillance de la qualité, lancées par rag-ionis-monitor@.service :
#   quality : canari, LLM juge et rapport de la veille (chaque matin)
#   recall  : recall du golden dataset sur la prod (le 1er et le 15 du mois)
# Chaque tâche tourne dans un conteneur ponctuel de l'image de l'API et envoie son
# résultat sur Telegram. Un crash (aucun résultat) fait échouer le service, ce qui
# déclenche l'alerte OnFailure.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <quality|recall>\n' "$0" >&2
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
      exit 1
    fi
    printf '%s\n' "${result}" | python3 -c 'import json, sys; print(json.load(sys.stdin)["message"])' |
      /usr/bin/bash "${telegram}"
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
      exit 1
    fi
    # Historique local, une ligne par mesure.
    printf '%s\n' "${result}" | python3 -c '
import json, sys, datetime
row = json.load(sys.stdin)
row["measured_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
print(json.dumps(row, ensure_ascii=False))
' >>"${state_dir}/recall-history.jsonl"
    # golden_recall sort en erreur si le plancher ou la baisse maximale est franchi.
    printf '%s\n' "${result}" | python3 -c '
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
print("\n".join(lines))
' "${status}" "${baseline}" | /usr/bin/bash "${telegram}"
    ;;
  *)
    printf 'Unknown monitoring task: %s (expected quality or recall)\n' "${task}" >&2
    exit 2
    ;;
esac
