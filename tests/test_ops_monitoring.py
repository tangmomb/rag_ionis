"""Healthchecks (tâches planifiées) et Grafana (alertes, dashboard) restent alignés avec systemd."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = ROOT_DIR / "deploy" / "systemd"
INSTALLER = ROOT_DIR / "deploy" / "install-systemd-timers.sh"


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Les dataclasses relisent leur module dans sys.modules.
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


healthchecks = load(ROOT_DIR / "deploy" / "healthchecks" / "setup_checks.py")
grafana = load(ROOT_DIR / "deploy" / "grafana" / "apply_grafana.py")


def slug_of(unit: str) -> str:
    """Même dérivation que rag-ionis-healthcheck.sh."""
    return unit.removesuffix(".service").replace("@", "-")


class HealthchecksTests(unittest.TestCase):
    def test_every_timer_has_a_check_on_the_same_schedule(self) -> None:
        checks = {check.unit: check for check in healthchecks.CHECKS}
        timers = sorted(SYSTEMD_DIR.glob("*.timer"))
        self.assertTrue(timers)
        for timer in timers:
            text = timer.read_text(encoding="utf-8")
            unit = re.search(r"^Unit=(\S+)$", text, re.M).group(1)
            self.assertIn(unit, checks, f"{timer.name} n'a pas de check healthchecks")
            check = checks[unit]
            self.assertEqual(slug_of(unit), check.slug)
            hour, minute = re.search(r"(\d\d):(\d\d):00", text).groups()
            cron_minute, cron_hour = check.schedule.split()[:2]
            self.assertEqual((int(minute), int(hour)), (int(cron_minute), int(cron_hour)), timer.name)
        # La sauvegarde est installée hors du dépôt mais a aussi son check.
        self.assertIn("rag-ionis-backup.service", checks)

    def test_payload_upserts_on_slug_in_paris_time(self) -> None:
        payload = healthchecks.payload(healthchecks.CHECKS[0])
        self.assertEqual(payload["unique"], ["slug"])
        self.assertEqual(payload["tz"], "Europe/Paris")
        self.assertGreater(payload["grace"], 0)

    def test_jobs_ping_start_success_and_failure(self) -> None:
        for name in ("rag-ionis-monitor@.service", "rag-ionis-update@.service"):
            unit = (SYSTEMD_DIR / name).read_text(encoding="utf-8")
            self.assertIn("ExecStartPre=-/usr/bin/bash /usr/local/bin/rag-ionis-healthcheck.sh %n start", unit)
            self.assertIn("ExecStartPost=-/usr/bin/bash /usr/local/bin/rag-ionis-healthcheck.sh %n success", unit)
        notify = (SYSTEMD_DIR / "rag-ionis-notify-failure.sh").read_text(encoding="utf-8")
        # Le ping d'échec part avant Telegram, qui peut être cassé.
        self.assertLess(notify.index("rag-ionis-healthcheck.sh"), notify.index("rag-ionis-telegram.sh"))
        installer = INSTALLER.read_text(encoding="utf-8")
        self.assertIn("rag-ionis-backup.service.d/healthcheck.conf", installer)
        self.assertIn("ExecStartPre=-+/usr/bin/bash /usr/local/bin/rag-ionis-healthcheck.sh %%n start", installer)
        self.assertIn("ExecStartPost=-+/usr/bin/bash /usr/local/bin/rag-ionis-healthcheck.sh %%n success", installer)
        script = (SYSTEMD_DIR / "rag-ionis-healthcheck.sh").read_text(encoding="utf-8")
        # La clé de ping ne doit jamais apparaître sur la ligne de commande (ps).
        self.assertIn("--config -", script)

    @unittest.skipIf(os.name == "nt" or shutil.which("bash") is None, "bash POSIX requis")
    def test_ping_script_never_fails_the_job(self) -> None:
        script = SYSTEMD_DIR / "rag-ionis-healthcheck.sh"
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "RAG_IONIS_HEALTHCHECKS_ENV_FILE": str(Path(directory) / "absent.env")}
            result = subprocess.run(
                ["bash", str(script), "rag-ionis-monitor@quality.service", "fail"],
                env=env, capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn("rag-ionis-monitor-quality", result.stderr)


class GrafanaTests(unittest.TestCase):
    def test_rules_target_loki_and_the_telegram_contact_point(self) -> None:
        group = grafana.rule_group("loki-uid")
        self.assertEqual(group["interval"], 300)
        uids = [rule["uid"] for rule in group["rules"]]
        self.assertEqual(len(uids), len(set(uids)))
        serialized = json.dumps(group)
        self.assertNotIn("${loki}", serialized)
        for rule in group["rules"]:
            self.assertEqual(rule["data"][0]["datasourceUid"], "loki-uid")
            self.assertEqual(rule["notification_settings"]["receiver"], grafana.CONTACT_POINT)
            self.assertEqual(rule["folderUID"], grafana.FOLDER_UID)

    def test_only_the_absent_logs_rule_alerts_on_no_data(self) -> None:
        no_data = {rule["uid"]: rule["noDataState"] for rule in grafana.alert_rules()}
        self.assertEqual(no_data.pop("rag-ionis-logs-absent"), "Alerting")
        self.assertEqual(set(no_data.values()), {"OK"})

    def test_latency_rule_matches_the_slo(self) -> None:
        rule = next(r for r in grafana.alert_rules() if r["uid"] == "rag-ionis-answer-ready-p95")
        self.assertIn("rag_answer_ready_ms", rule["data"][0]["model"]["expr"])
        self.assertEqual(rule["data"][2]["model"]["conditions"][0]["evaluator"]["params"], [10_000])

    def test_dashboard_skips_quality_panels_without_infinity(self) -> None:
        with_quality = grafana.dashboard("loki-uid", "infinity-uid")
        without = grafana.dashboard("loki-uid", None)
        self.assertEqual(len(with_quality["panels"]) - len(without["panels"]), 2)
        self.assertNotIn(grafana.INFINITY_TYPE, json.dumps(without))
        roots = {
            target["root_selector"]
            for panel in with_quality["panels"]
            for target in panel.get("targets", [])
            if "root_selector" in target
        }
        # Les deux séries renvoyées par /api/monitoring/history.
        self.assertEqual(roots, {"recall", "faithfulness"})

    def test_pick_datasources_prefers_the_stack_logs(self) -> None:
        loki, infinity = grafana.pick_datasources(
            [
                {"type": "loki", "uid": "other", "name": "grafanacloud-usage"},
                {"type": "loki", "uid": "logs", "name": "grafanacloud-stack-logs"},
                {"type": grafana.INFINITY_TYPE, "uid": "inf", "name": "infinity"},
            ]
        )
        self.assertEqual((loki, infinity), ("logs", "inf"))


if __name__ == "__main__":
    unittest.main()
