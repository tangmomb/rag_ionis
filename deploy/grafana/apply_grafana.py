"""Installe dans Grafana Cloud les alertes et le dashboard « santé » du RAG IONIS.

Ce que le script crée ou met à jour (idempotent, on le relance après une modification) :
  - le dossier « RAG IONIS » ;
  - le point de contact Telegram « rag-ionis-telegram » (même bot que les tâches systemd) ;
  - le groupe d'alertes « rag-ionis », évalué toutes les 5 min sur les logs Loki ;
  - le dashboard « RAG IONIS — santé » : trafic, erreurs et latence (Loki),
    recall et fidélité dans le temps (/api/monitoring/history, via Infinity).

Les alertes passent directement par le point de contact (routage simplifié de la
règle) : la politique de notification du stack n'est pas modifiée.

Prérequis côté Grafana Cloud :
  - un compte de service avec le rôle Editor et son jeton ;
  - le plugin Infinity installé (catalogue des plugins) pour les courbes de qualité ;
    sans lui, le dashboard est créé sans ces deux panneaux.

Lancement depuis le poste de dev, avec le bot et le chat de /etc/rag-ionis/telegram.env :

    export GRAFANA_URL=https://<stack>.grafana.net GRAFANA_TOKEN=glsa_...
    export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
    python deploy/grafana/apply_grafana.py

    python deploy/grafana/apply_grafana.py --dry-run   # affiche le JSON, n'envoie rien
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

FOLDER_UID = "rag-ionis"
FOLDER_TITLE = "RAG IONIS"
RULE_GROUP = "rag-ionis"
CONTACT_POINT = "rag-ionis-telegram"
DASHBOARD_UID = "rag-ionis-health"
INFINITY_TYPE = "yesoreyeram-infinity-datasource"
HISTORY_URL = os.getenv("RAG_IONIS_HISTORY_URL", "https://ionis.tanguym.fr/api/monitoring/history")
RUNBOOK_URL = "https://github.com/tangmomb/rag_ionis/blob/master/docs/exploitation.md"

API = '{app="rag-ionis", service="api"}'

# --------------------------------------------------------------------------- alertes


def _rule(
    uid: str,
    title: str,
    expr: str,
    *,
    window_seconds: int,
    evaluator: str,
    threshold: float,
    summary: str,
    runbook: str,
    pending: str = "0s",
    no_data: str = "OK",
) -> dict[str, Any]:
    """Une règle Grafana-managed : requête Loki instantanée, dernière valeur, seuil."""
    return {
        "uid": uid,
        "title": title,
        "condition": "C",
        "data": [
            {
                "refId": "A",
                "queryType": "instant",
                "relativeTimeRange": {"from": window_seconds, "to": 0},
                "datasourceUid": "${loki}",
                "model": {"refId": "A", "expr": expr, "queryType": "instant", "instant": True},
            },
            {
                "refId": "B",
                "datasourceUid": "__expr__",
                "model": {"refId": "B", "type": "reduce", "expression": "A", "reducer": "last"},
            },
            {
                "refId": "C",
                "datasourceUid": "__expr__",
                "model": {
                    "refId": "C",
                    "type": "threshold",
                    "expression": "B",
                    "conditions": [{"evaluator": {"type": evaluator, "params": [threshold]}}],
                },
            },
        ],
        # Sans ligne correspondante, Loki ne renvoie aucune série : pas d'erreur = OK,
        # sauf pour la règle qui surveille justement l'absence de logs.
        "noDataState": no_data,
        "execErrState": "Error",
        "for": pending,
        "labels": {"app": "rag-ionis"},
        "annotations": {
            "summary": summary,
            "description": runbook,
            "runbook_url": RUNBOOK_URL,
        },
        "notification_settings": {"receiver": CONTACT_POINT},
    }


def alert_rules() -> list[dict[str, Any]]:
    return [
        _rule(
            "rag-ionis-api-5xx",
            "API : réponses 5xx",
            f"sum(count_over_time({API} | status >= 500 [15m]))",
            window_seconds=900,
            evaluator="gt",
            threshold=2,
            summary="Au moins 3 réponses 5xx de l'API en 15 min.",
            runbook="Grafana Explore : {app=\"rag-ionis\", service=\"api\"} | status >= 500, "
            "puis la trace Phoenix via trace_id. Runbook : « Site ou API en panne ».",
        ),
        _rule(
            "rag-ionis-rag-failed",
            "RAG : questions en échec",
            f'sum(count_over_time({API} | rag_event="rag_failed" [30m]))',
            window_seconds=1800,
            evaluator="gt",
            threshold=1,
            summary="Au moins 2 questions en échec (rag_failed) en 30 min.",
            runbook="Le champ error de la ligne de log donne la cause (fournisseur LLM, base, quota). "
            "Ouvrir la trace Phoenix correspondante.",
        ),
        _rule(
            "rag-ionis-answer-ready-p95",
            "RAG : latence p95 > 10 s",
            "quantile_over_time(0.95, "
            f'{API} | rag_event="rag_answered" | json ready="rag_answer_ready_ms" '
            '| __error__="" | unwrap ready [3h]) by ()',
            window_seconds=10800,
            evaluator="gt",
            threshold=10_000,
            pending="30m",
            summary="Latence « réponse prête » p95 au-dessus de 10 s sur 3 h (SLO).",
            runbook="Comparer les étapes dans Phoenix (retrieval, rerank, LLM) ; "
            "un fournisseur lent se voit sur le span LLM. Beszel pour le CPU/temps volé.",
        ),
        _rule(
            "rag-ionis-rate-limited",
            "API : beaucoup de questions refusées",
            f"sum(count_over_time({API} | status = 429 [1h]))",
            window_seconds=3600,
            evaluator="gt",
            threshold=20,
            summary="Plus de 20 questions refusées (limite par IP ou budget) en 1 h.",
            runbook="Un robot ou le budget quotidien atteint. "
            "Runbook : « Un utilisateur reçoit Limite … atteinte ou budget ».",
        ),
        _rule(
            "rag-ionis-logs-absent",
            "Logs : plus rien reçu",
            'sum(count_over_time({app="rag-ionis", source="systemd"} [26h]))',
            window_seconds=93600,
            evaluator="lt",
            threshold=1,
            no_data="Alerting",
            summary="Aucun log des tâches systemd depuis 26 h.",
            runbook="Le rapport du matin écrit chaque jour : soit l'agent Alloy est arrêté "
            "(docker ps, profil observability), soit les tâches ne tournent plus (healthchecks). "
            "Runbook : « Logs absents de Grafana ».",
        ),
    ]


def rule_group(loki_uid: str) -> dict[str, Any]:
    rules = json.loads(json.dumps(alert_rules()).replace("${loki}", loki_uid))
    for rule in rules:
        rule.update(folderUID=FOLDER_UID, ruleGroup=RULE_GROUP)
    return {"title": RULE_GROUP, "folderUid": FOLDER_UID, "interval": 300, "rules": rules}


def contact_point(bot_token: str, chat_id: str) -> dict[str, Any]:
    return {
        "uid": CONTACT_POINT,
        "name": CONTACT_POINT,
        "type": "telegram",
        "settings": {"bottoken": bot_token, "chatid": chat_id},
        "disableResolveMessage": False,
    }


# --------------------------------------------------------------------------- dashboard


def _loki(uid: str) -> dict[str, str]:
    return {"type": "loki", "uid": uid}


def _target(uid: str, expr: str, legend: str = "", instant: bool = False) -> dict[str, Any]:
    return {
        "refId": "A",
        "datasource": _loki(uid),
        "expr": expr,
        "legendFormat": legend,
        "queryType": "instant" if instant else "range",
    }


def _stat(uid: str, title: str, expr: str, x: int, unit: str = "short", red_above: float | None = None) -> dict:
    steps = [{"color": "green", "value": None}]
    if red_above is not None:
        steps.append({"color": "red", "value": red_above})
    return {
        "type": "stat",
        "title": title,
        "gridPos": {"x": x, "y": 0, "w": 4, "h": 4},
        "datasource": _loki(uid),
        "targets": [_target(uid, expr, instant=True)],
        "fieldConfig": {
            "defaults": {"unit": unit, "noValue": "0", "thresholds": {"mode": "absolute", "steps": steps}},
            "overrides": [],
        },
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "none"},
    }


def _timeseries(title: str, grid: dict, datasource: dict, targets: list, unit: str = "short", **defaults) -> dict:
    return {
        "type": "timeseries",
        "title": title,
        "gridPos": grid,
        "datasource": datasource,
        "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit, **defaults}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
    }


def _quality_panel(infinity_uid: str, title: str, root: str, columns: list[tuple[str, str]], x: int) -> dict:
    datasource = {"type": INFINITY_TYPE, "uid": infinity_uid}
    target = {
        "refId": "A",
        "datasource": datasource,
        "type": "json",
        "source": "url",
        "format": "table",
        "parser": "backend",
        "url": HISTORY_URL,
        "url_options": {"method": "GET"},
        "root_selector": root,
        "columns": [{"selector": "at", "text": "Date", "type": "timestamp"}]
        + [{"selector": selector, "text": text, "type": "number"} for selector, text in columns],
    }
    return _timeseries(
        title,
        {"x": x, "y": 13, "w": 12, "h": 8},
        datasource,
        [target],
        unit="percentunit",
        min=0,
        max=1,
        custom={"drawStyle": "line", "showPoints": "always", "spanNulls": True, "thresholdsStyle": {"mode": "line+area"}},
        thresholds={"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "transparent", "value": 0.8}]},
    )


LINKS_MARKDOWN = f"""\
**Alertes** : Telegram. **Runbook** : [docs/exploitation.md]({RUNBOOK_URL})

- [Page de statut (Upptime)](https://tangmomb.github.io/status/) — disponibilité vue de l'extérieur
- [Beszel](https://monitor.tanguym.fr) — CPU, RAM, disque du VPS
- [Healthchecks](https://healthchecks.io/projects/) — tâches planifiées (sauvegarde, rapports, restauration)
- [Phoenix](http://contabo:6006) — réseau privé Tailscale (secours : `ssh -L 6006:127.0.0.1:6006 contabo`)
- Retrouver une requête : `{{app="rag-ionis", service="api"}} | request_id=~"<référence>.*"`
"""


def dashboard(loki_uid: str, infinity_uid: str | None) -> dict[str, Any]:
    loki = _loki(loki_uid)
    answered = f'{API} | rag_event="rag_answered"'
    ready = f'{answered} | json ready="rag_answer_ready_ms" | __error__="" | unwrap ready'
    panels: list[dict[str, Any]] = [
        _stat(loki_uid, "Questions répondues", f"sum(count_over_time({answered} [$__range]))", 0),
        _stat(loki_uid, "Questions en échec", f'sum(count_over_time({API} | rag_event="rag_failed" [$__range]))', 4, red_above=1),
        _stat(loki_uid, "Réponses 5xx", f"sum(count_over_time({API} | status >= 500 [$__range]))", 8, red_above=1),
        _stat(loki_uid, "Refusées (429)", f"sum(count_over_time({API} | status = 429 [$__range]))", 12, red_above=20),
        _stat(loki_uid, "Réponse prête p95", f"quantile_over_time(0.95, {ready} [$__range]) by ()", 16, unit="ms", red_above=10_000),
        {
            "type": "text",
            "title": "Liens",
            "gridPos": {"x": 20, "y": 0, "w": 4, "h": 13},
            "options": {"mode": "markdown", "content": LINKS_MARKDOWN},
        },
        _timeseries(
            "Questions par heure",
            {"x": 0, "y": 4, "w": 10, "h": 9},
            loki,
            [_target(loki_uid, f'sum by (rag_event) (count_over_time({API} | rag_event=~"rag_answered|rag_failed" [1h]))', "{{rag_event}}")],
            custom={"drawStyle": "bars", "fillOpacity": 60, "stacking": {"mode": "normal"}},
        ),
        _timeseries(
            "Latence « réponse prête » (fenêtre 1 h)",
            {"x": 10, "y": 4, "w": 10, "h": 9},
            loki,
            [
                _target(loki_uid, f"quantile_over_time(0.5, {ready} [1h]) by ()", "p50"),
                {**_target(loki_uid, f"quantile_over_time(0.95, {ready} [1h]) by ()", "p95"), "refId": "B"},
            ],
            unit="ms",
            custom={"thresholdsStyle": {"mode": "line"}},
            thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 10_000}]},
        ),
    ]
    if infinity_uid:
        panels += [
            _quality_panel(
                infinity_uid,
                "Recall du golden dataset (seuil 80 %)",
                "recall",
                [("video_recall", "Recall vidéo"), ("chunk_recall", "Recall chunks")],
                0,
            ),
            _quality_panel(
                infinity_uid,
                "Fidélité des réponses (seuil 80 %)",
                "faithfulness",
                [("canary_faithfulness", "Canari"), ("traffic_faithfulness", "Trafic réel")],
                12,
            ),
        ]
    panels += [
        _timeseries(
            "Codes HTTP ≥ 400",
            {"x": 0, "y": 21, "w": 12, "h": 8},
            loki,
            [_target(loki_uid, f"sum by (status) (count_over_time({API} | status >= 400 [$__interval]))", "{{status}}")],
            custom={"drawStyle": "bars", "fillOpacity": 60, "stacking": {"mode": "normal"}},
        ),
        {
            "type": "logs",
            "title": "Erreurs récentes (API et tâches)",
            "gridPos": {"x": 12, "y": 21, "w": 12, "h": 8},
            "datasource": loki,
            "targets": [
                _target(
                    loki_uid,
                    '{app="rag-ionis"} |~ "(?i)traceback|\\"level\\": ?\\"(error|critical)\\"|échec|failed"',
                )
            ],
            "options": {"showTime": True, "wrapLogMessage": True, "sortOrder": "Descending"},
        },
    ]
    return {
        "uid": DASHBOARD_UID,
        "title": "RAG IONIS — santé",
        "tags": ["rag-ionis"],
        "timezone": "Europe/Paris",
        "time": {"from": "now-24h", "to": "now"},
        "refresh": "5m",
        "schemaVersion": 39,
        "panels": panels,
    }


# --------------------------------------------------------------------------- API Grafana


class Grafana:
    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token

    def call(self, method: str, path: str, body: Any = None, ok_missing: bool = False) -> Any:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                # Les objets restent modifiables dans l'interface.
                "X-Disable-Provenance": "true",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if ok_missing and error.code == 404:
                return None
            detail = error.read().decode(errors="replace")[:500]
            raise SystemExit(f"{method} {path} : HTTP {error.code} {detail}") from None
        return json.loads(raw) if raw else None


def pick_datasources(datasources: list[dict[str, Any]]) -> tuple[str, str | None]:
    lokis = [d for d in datasources if d.get("type") == "loki"]
    if not lokis:
        raise SystemExit("Aucune source Loki dans ce stack Grafana.")
    # Grafana Cloud nomme la source de logs du stack grafanacloud-<stack>-logs.
    loki = next((d for d in lokis if d.get("name", "").endswith("-logs")), lokis[0])
    infinity = next((d for d in datasources if d.get("type") == INFINITY_TYPE), None)
    return loki["uid"], infinity["uid"] if infinity else None


def ensure_infinity(grafana: Grafana, infinity_uid: str | None) -> str | None:
    if infinity_uid:
        return infinity_uid
    plugin = grafana.call("GET", f"/api/plugins/{INFINITY_TYPE}/settings", ok_missing=True)
    if plugin is None:
        print("⚠ Plugin Infinity absent : dashboard sans les courbes de recall et de fidélité.", file=sys.stderr)
        return None
    created = grafana.call(
        "POST",
        "/api/datasources",
        {"name": "rag-ionis-public-api", "type": INFINITY_TYPE, "access": "proxy", "jsonData": {}},
    )
    return created["datasource"]["uid"]


def apply(grafana: Grafana, bot_token: str, chat_id: str) -> None:
    loki_uid, infinity_uid = pick_datasources(grafana.call("GET", "/api/datasources"))
    print(f"Loki : {loki_uid}")

    # La liste plutôt que GET /api/folders/<uid> : avec des droits par dossier,
    # un dossier absent répond 403 et non 404.
    folders = grafana.call("GET", "/api/folders") or []
    if not any(folder.get("uid") == FOLDER_UID for folder in folders):
        grafana.call("POST", "/api/folders", {"uid": FOLDER_UID, "title": FOLDER_TITLE})
    print(f"Dossier : {FOLDER_TITLE}")

    existing = grafana.call("GET", f"/api/v1/provisioning/contact-points?name={CONTACT_POINT}") or []
    point = contact_point(bot_token, chat_id)
    if existing:
        grafana.call("PUT", f"/api/v1/provisioning/contact-points/{existing[0]['uid']}", point)
    else:
        grafana.call("POST", "/api/v1/provisioning/contact-points", point)
    print(f"Point de contact : {CONTACT_POINT}")

    group = rule_group(loki_uid)
    grafana.call("PUT", f"/api/v1/provisioning/folder/{FOLDER_UID}/rule-groups/{RULE_GROUP}", group)
    print(f"Alertes : {len(group['rules'])} règles dans le groupe {RULE_GROUP}")

    infinity_uid = ensure_infinity(grafana, infinity_uid)
    result = grafana.call(
        "POST",
        "/api/dashboards/db",
        {
            "dashboard": dashboard(loki_uid, infinity_uid),
            "folderUid": FOLDER_UID,
            "overwrite": True,
            "message": "deploy/grafana/apply_grafana.py",
        },
    )
    print(f"Dashboard : {grafana.url}{result.get('url', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="affiche le JSON sans rien envoyer")
    args = parser.parse_args()
    # Accents et symboles lisibles aussi dans une console Windows.
    sys.stdout.reconfigure(encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"rule_group": rule_group("<loki>"), "dashboard": dashboard("<loki>", "<infinity>")},
                         ensure_ascii=False, indent=2))
        return 0
    missing = [
        name
        for name in ("GRAFANA_URL", "GRAFANA_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
        if not os.getenv(name)
    ]
    if missing:
        print(f"Variables manquantes : {', '.join(missing)}", file=sys.stderr)
        return 2
    apply(
        Grafana(os.environ["GRAFANA_URL"], os.environ["GRAFANA_TOKEN"]),
        os.environ["TELEGRAM_BOT_TOKEN"],
        os.environ["TELEGRAM_CHAT_ID"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
