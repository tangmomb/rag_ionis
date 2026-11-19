"""Run the four scheduled phases of the daily VPS conversations."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import date, datetime, time as local_time, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from dotenv import dotenv_values
from mistralai.client import Mistral

from utils.model_embedding.benchmark import DEFAULT_VIDEO_DIR
from utils.model_embedding.generate_synthetic_questions import VideoChunks, generate_questions, write_text
from utils.model_embedding.simulate_rag_users import (
    DEFAULT_MAX_TURNS,
    followup_batch,
    play_turns,
    recover_interrupted_turns,
    request_quota_reached,
    save_state,
    simulate,
    wait_until,
)

PARIS = ZoneInfo("Europe/Paris")
API_URL = "https://ionis.tanguym.fr/api/rag/stream"
OPENING_HOURS = (8, 12)
FOLLOWUP_HOURS = (16, 20)
DAYTIME_PHASE_STARTS = {
    "openings": local_time(OPENING_HOURS[0]),
    "followup-batch": local_time(12, 30),
    "followup-send": local_time(FOLLOWUP_HOURS[0]),
}
LATE_TOLERANCE = timedelta(minutes=15)
CORPUS_SQL = """
SELECT json_build_object(
    'video', v.title,
    'chunks', CASE WHEN v.is_long_video
        THEN json_build_array((array_agg(c.content ORDER BY c.chunk_index))[1])
        ELSE json_agg(c.content ORDER BY c.chunk_index, c.chunk_level)
    END
)
FROM data.chunks c JOIN data.videos v ON v.id = c.video_id
WHERE (NOT v.is_long_video OR c.chunk_level = 'global') AND length(trim(c.content)) > 0
GROUP BY v.id
ORDER BY v.youtube_video_id
"""


def load_corpus() -> list[VideoChunks]:
    rows = subprocess.run(
        [
            "docker", "exec", "rag_ionis_postgres", "sh", "-c",
            'exec psql -At -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" "$POSTGRES_DB" -c "$1"',
            "psql", CORPUS_SQL,
        ],
        capture_output=True, encoding="utf-8", check=True,
    ).stdout.splitlines()
    return [json.loads(line) for line in rows]


def build_plan(questions: list[dict], day: date) -> list[list[dict]]:
    rng = random.Random(day.toordinal())
    shuffled = questions.copy()
    rng.shuffle(shuffled)
    start, end = OPENING_HOURS
    start_time = datetime.combine(day, local_time(start), tzinfo=PARIS)
    available_minutes = (end - start) * 60 - 5
    slots = sorted(rng.sample(range(available_minutes), len(shuffled)))
    return [[
        {**question, "send_at": (start_time + timedelta(minutes=minute)).isoformat()}
        for question, minute in zip(shuffled, slots)
    ]]


def prepare_day(api_key: str, day: date, directory: Path, count: int) -> int:
    plan = directory / "day-plan.json"
    if plan.exists():
        groups = json.loads(plan.read_text(encoding="utf-8"))
        if sum(map(len, groups)) != count:
            raise ValueError("Le plan enregistré utilise un nombre de sessions différent.")
        if len(groups) == 1 or any(directory.glob("vague-*/simulation.json")):
            return 0
        rows = [item for group in groups for item in group]
    else:
        with Mistral(api_key=api_key) as client:
            if generate_questions(client, day, DEFAULT_VIDEO_DIR, count, directory, videos=load_corpus):
                return 2
        rows = [json.loads(line) for line in (directory / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != count:
        raise ValueError(f"{len(rows)} questions utilisables pour {count} sessions.")
    groups = build_plan(rows, day)
    write_text(plan, json.dumps(groups, ensure_ascii=False))
    slots = [item["send_at"] for group in groups for item in group]
    print(f"{count} ouvertures prêtes, de {slots[0]} à {slots[-1]}.", flush=True)
    return 0


def open_sessions(
    manifest: Path, group: list[dict], *, request_limit: int | None = None, min_interval: float = 0,
) -> dict:
    if manifest.exists():
        state = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        state = {
            "api_url": API_URL, "max_turns": DEFAULT_MAX_TURNS, "run_id": uuid4().hex[:12],
            "sessions": [{**item, "turns": [], "next_question": item["question"]} for item in group],
        }
        manifest.parent.mkdir(parents=True, exist_ok=True)
    if state["api_url"] != API_URL and any(session["turns"] for session in state["sessions"]):
        raise RuntimeError("Une conversation déjà ouverte doit rester attachée à son API.")
    state.update(api_url=API_URL, request_limit=request_limit, min_interval=min_interval)
    recover_interrupted_turns(state["sessions"])
    save_state(manifest, state)

    for session in state["sessions"]:
        if session["turns"] or "outcome" in session:
            continue
        target = datetime.fromisoformat(session["send_at"])
        if not OPENING_HOURS[0] <= target.hour < OPENING_HOURS[1]:
            session["outcome"] = "missed_opening"
            save_state(manifest, state)
            continue
        wait_until(target)
        now = datetime.now(PARIS)
        if now - target > LATE_TOLERANCE or now.hour >= OPENING_HOURS[1]:
            session["outcome"] = "missed_opening"
            save_state(manifest, state)
            continue
        print(f"Envoi {session['id']} prévu à {session['send_at']}.", flush=True)
        play_turns(state, manifest, [session], parallelism=1, timeout=180, api_url=API_URL)
    return state


def planned_waves(directory: Path):
    plan = json.loads((directory / "day-plan.json").read_text(encoding="utf-8"))
    for index, group in enumerate(plan, start=1):
        if group:
            yield directory / f"vague-{index}" / "simulation.json", group


def dispatch_openings(directory: Path, *, request_limit: int, min_interval: float) -> int:
    failed = False
    for manifest, group in planned_waves(directory):
        state = open_sessions(
            manifest, group, request_limit=request_limit, min_interval=min_interval,
        )
        failed |= any(
            session.get("outcome") in {"rag_error", "missed_opening", "request_quota_reached"}
            for session in state["sessions"]
        )
    return 2 if failed else 0


def followup_states(directory: Path):
    for manifest, _group in planned_waves(directory):
        state = json.loads(manifest.read_text(encoding="utf-8"))
        if state["api_url"] != API_URL:
            raise RuntimeError("Une conversation déjà ouverte doit rester attachée à son API.")
        recover_interrupted_turns(state["sessions"])
        for session in state["sessions"]:
            if not session["turns"] and "outcome" not in session:
                session["outcome"] = "missed_opening"
        save_state(manifest, state)
        yield manifest, state


def pending_followups(state: dict) -> list[dict]:
    sessions = state["sessions"]
    if request_quota_reached(state, sum(len(session["turns"]) for session in sessions)):
        return []
    return [
        session for session in sessions
        if "outcome" not in session and len(session["turns"]) == 1 and "next_question" not in session
    ]


def submit_followups(client: Mistral, day: date, directory: Path) -> int:
    for manifest, state in followup_states(directory):
        waiting = pending_followups(state)
        if waiting and not state.get("turn_2_job_1"):
            followup_batch(client, state, manifest, day, 2, 1, waiting, wait=False)
    return 0


def dispatch_followups(client: Mistral, day: date, directory: Path) -> int:
    start, end = FOLLOWUP_HOURS
    send_not_before = datetime.combine(day, local_time(start), tzinfo=PARIS)
    deadline = datetime.combine(day, local_time(end), tzinfo=PARIS)
    failed = False
    for manifest, state in followup_states(directory):
        if pending_followups(state) and not state.get("turn_2_job_1") and datetime.now(PARIS) < deadline:
            raise RuntimeError("Aucun lot de relances enregistré pour cette journée.")
        status = simulate(
            client, state, manifest, day, parallelism=1, timeout=180,
            send_not_before=send_not_before, send_deadline=deadline,
        )
        failed |= bool(status) or any(
            session.get("outcome") == "followup_window_closed" for session in state["sessions"]
        )
    return 2 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("night", "openings", "followup-batch", "followup-send"))
    parser.add_argument("--env-file", type=Path, default=Path("/srv/rag_ionis/.env.production"))
    parser.add_argument("--state-dir", type=Path, default=Path("/var/lib/rag-ionis-synthetic"))
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.count <= 30:
        parser.error("--count doit être compris entre 1 et 30")
    now = datetime.now(PARIS)
    day = args.date or now.date()
    phase_start = DAYTIME_PHASE_STARTS.get(args.phase)
    # A persistent timer can catch up yesterday's missed activation before today's slot.
    if phase_start is not None and now < datetime.combine(day, phase_start, tzinfo=PARIS):
        print(f"Phase {args.phase} prévue à {phase_start:%H:%M} pour le {day}.", flush=True)
        return 0
    directory = args.state_dir / day.isoformat()
    environment = dotenv_values(args.env_file)
    print(f"Simulation du {day} · phase {args.phase}", flush=True)
    if args.phase == "openings":
        if environment.get("RAG_RATE_LIMIT_VPS_IP"):
            daily_limit = int(environment.get("RAG_RATE_LIMIT_VPS_PER_DAY") or 100)
        else:
            daily_limit = int(environment.get("RAG_RATE_LIMIT_PER_DAY") or 30)
        minute_limit = int(environment.get("RAG_RATE_LIMIT_PER_MINUTE") or 10)
        return dispatch_openings(directory, request_limit=daily_limit, min_interval=60 / max(1, minute_limit))
    api_key = environment.get("MISTRAL_API_KEY")
    if not api_key:
        parser.error("MISTRAL_API_KEY manquante")
    if args.phase == "night":
        return prepare_day(api_key, day, directory, args.count)
    with Mistral(api_key=api_key) as client:
        if args.phase == "followup-batch":
            return submit_followups(client, day, directory)
        return dispatch_followups(client, day, directory)


if __name__ == "__main__":
    raise SystemExit(main())
