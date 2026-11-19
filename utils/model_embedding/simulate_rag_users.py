"""Replay synthetic conversations over RAG SSE and generate follow-ups in batches."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

import requests
from mistralai.client import Mistral

from pipeline.support.environment import load_project_env
from utils.model_embedding.benchmark import DEFAULT_VIDEO_DIR, PROJECT_DIR
from utils.model_embedding.generate_synthetic_questions import (
    DEFAULT_OUTPUT_DIR,
    batch_json,
    batch_results,
    chat_request,
    generate_questions,
    question_signature,
    run_batch,
    write_jsonl,
    write_text,
)

DEFAULT_MAX_TURNS = 2

FOLLOWUP_PROMPT = """Tu es un utilisateur du RAG. À partir de l'échange fourni,
écris la prochaine question que tu poserais naturellement.
Renvoie {"question": "..."} ou {"question": null} si l'échange est terminé.
"""


def save_state(path: Path, state: dict[str, Any]) -> None:
    write_text(path, json.dumps(state, ensure_ascii=False))


def sse_events(response):
    # SSE uses blank lines as event delimiters and permits several data lines.
    data = []
    for raw in response.iter_lines(chunk_size=256):
        line = raw.decode("utf-8")
        if not line:
            if data:
                yield json.loads("\n".join(data))
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield json.loads("\n".join(data))


def ask_rag(api_url: str, turn: dict[str, Any], conversation_id: int | None, timeout: float):
    payload = {
        "question": turn["question"],
        "conversationId": conversation_id,
        "embeddingModel": "meme modele que la base",
        "rerankModel": "cohere rerank",
        "useSql": True,
        "useRerank": True,
        "topK": 40,
        "finalK": 5,
    }
    started = time.monotonic()
    result: dict[str, Any] = {"status": "error"}
    try:
        with requests.post(
            api_url, json=payload, headers={"X-Request-ID": turn["request_id"]},
            stream=True, timeout=(10, timeout), allow_redirects=False,
        ) as response:
            result["request_id"] = response.headers.get("X-Request-ID", turn["request_id"])
            if response.status_code != 200:
                result["http_status"] = response.status_code
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdecimal():
                    result["retry_after_seconds"] = int(retry_after)
                raise ValueError(f"HTTP {response.status_code}")
            if not response.headers.get("Content-Type", "").startswith("text/event-stream"):
                raise ValueError("réponse SSE attendue")
            for event in sse_events(response):
                elapsed_ms = round((time.monotonic() - started) * 1000)
                event_type = event.get("type")
                if event_type == "chunk" and event.get("text"):
                    result.setdefault("first_chunk_ms", elapsed_ms)
                elif event_type == "done":
                    answer = event["response"]
                    returned_id = answer.get("conversation_id")
                    if type(returned_id) is not int or returned_id <= 0:
                        raise ValueError("conversation_id absent ou invalide")
                    if conversation_id is not None and returned_id != conversation_id:
                        raise ValueError("le RAG a changé de conversation")
                    result.update(response=answer, done_ms=elapsed_ms)
                elif event_type == "memory":
                    result["memory"] = event.get("status")
                    result["memory_ms"] = elapsed_ms
                elif event_type == "error":
                    raise ValueError(str(event.get("detail") or "erreur SSE"))
            if "response" not in result:
                raise ValueError("flux interrompu avant done")
            if result.get("memory") == "failed":
                raise ValueError("mise à jour de la mémoire en échec")
            result["status"] = "completed"
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["error"] = str(exc)
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    return result


def video_key(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/") or None
    return parse_qs(parsed.query).get("v", [None])[0]


def check_opening_sources(session: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    expected = {(row["video_key"], str(row["chunk_index"])) for row in session["relevant"]}
    cited = answer.get("sources", [])
    retrieved = answer.get("retrieval", {}).get("retrieved_sources")
    result = {"expected_chunks": len(expected)}
    result["cited_chunks_found"] = len(expected & {
        (video_key(source.get("video_url")), str(source.get("chunk_index"))) for source in cited
    })
    if retrieved is not None:
        result["retrieved_videos_found"] = len(
            {key for key, _ in expected} & {video_key(source.get("video_url")) for source in retrieved}
        )
        if all(source.get("chunk_index") is not None for source in retrieved):
            result["retrieved_chunks_found"] = len(expected & {
                (video_key(source.get("video_url")), str(source["chunk_index"])) for source in retrieved
            })
    return result


def generation_sources(response: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = response.get("retrieval") or {}
    telemetry = retrieval.get("telemetry") or {}
    project = telemetry.get("project")
    trace_id = telemetry.get("trace_id")
    if project and trace_id and retrieval.get("retrieved_sources") != []:
        base_url = os.getenv("SYNTHETIC_PHOENIX_URL", "http://127.0.0.1:6006").rstrip("/")
        try:
            # The generation span holds every source, including uncited ones.
            for attempt in range(3):
                if attempt:
                    time.sleep(2.5)  # A just-completed trace may still be buffered.
                with requests.get(
                    f"{base_url}/v1/projects/{quote(project, safe='')}/spans",
                    params={"name": "generation", "trace_id": trace_id},
                    timeout=3,
                ) as result:
                    result.raise_for_status()
                    spans = result.json()["data"]
                for span in spans:
                    value = span.get("attributes", {}).get("input.value")
                    if isinstance(value, str):
                        value = json.loads(value)
                    if isinstance(value, dict) and isinstance(value.get("sources"), list):
                        return value["sources"]
        except (requests.RequestException, ValueError, KeyError, TypeError):
            pass
    return response.get("sources", [])


def followup_request(session: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    turn = session["turns"][-1]
    response = turn["response"]
    chunks = []
    for source in generation_sources(response):
        chunks.extend(
            chunk for chunk in (
                source.get("text"),
                (source.get("section_context") or {}).get("text"),
                (source.get("global_context") or {}).get("text"),
            ) if chunk
        )
    context = {
        "question_initiale": turn["question"],
        "chunks_envoyes_par_le_systeme": list(dict.fromkeys(chunks)),
        "reponse_initiale": response["answer"],
    }
    return chat_request(
        session["id"], FOLLOWUP_PROMPT, context,
        max_tokens=max_tokens,
    )


def apply_followups(sessions, results) -> list[dict[str, Any]]:
    rejected = []
    for session in sessions:
        try:
            if session["id"] not in results:
                raise ValueError("résultat absent du lot Mistral")
            question = batch_json(results[session["id"]])["question"]
            if question is None:
                session["outcome"] = "user_finished"
                continue
            if not isinstance(question, str) or not question.strip().endswith("?"):
                raise ValueError("relance invalide")
            signature = question_signature(question)
            if signature in {question_signature(turn["question"]) for turn in session["turns"]}:
                raise ValueError("question déjà posée")
            session["next_question"] = question.strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            session["followup_error"] = str(exc)
            rejected.append(session)
    return rejected


def followup_batch(client, state, manifest, day, number, attempt, sessions, *, wait=True, deadline=None):
    key = f"turn_{number}_job_{attempt}"
    requests = [] if state.get(key) else [
        followup_request(session, max_tokens=4096 if attempt == 1 else 8192)
        for session in sessions
    ]
    return run_batch(
        client, state, key, requests, manifest, day, purpose="synthetic-rag-followups",
        wait=wait, deadline=deadline,
    )


def generate_followups(client, state, manifest, day, number, sessions, *, deadline=None) -> None:
    pending = sessions
    for attempt in (1, 2):
        if not pending:
            break
        try:
            job = followup_batch(
                client, state, manifest, day, number, attempt, pending, deadline=deadline,
            )
        except TimeoutError:
            for session in pending:
                session["outcome"] = "followup_window_closed"
            break
        if job.status != "SUCCESS":
            raise RuntimeError(f"Batch des relances {job.id} terminé avec le statut {job.status}.")
        pending = apply_followups(pending, batch_results(client, job))
    for session in sessions:
        if session in pending:
            session.setdefault("outcome", "followup_error")
        else:
            session.pop("followup_error", None)
    save_state(manifest, state)


def request_quota_reached(state, sent: int) -> bool:
    request_limit = state.get("request_limit")
    return bool(state.get("quota_blocked")) or (request_limit is not None and sent >= request_limit)


def play_turns(state, manifest, sessions, parallelism, timeout, api_url, *, send_deadline=None) -> None:
    # Persist each dispatch before sending it, since retrying a POST can duplicate a message.
    sent = sum(len(session["turns"]) for session in state["sessions"])
    min_interval = state.get("min_interval", 0)
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        for offset in range(0, len(sessions), parallelism):
            tasks = {}
            for session in sessions[offset : offset + parallelism]:
                if request_quota_reached(state, sent):
                    session["outcome"] = "request_quota_reached"
                    save_state(manifest, state)
                    continue
                if min_interval:
                    delay = state.get("last_dispatch_at", 0) + min_interval - time.time()
                    if delay > 0:
                        time.sleep(delay)
                if send_deadline is not None and datetime.now(send_deadline.tzinfo) >= send_deadline:
                    session["outcome"] = "followup_window_closed"
                    save_state(manifest, state)
                    continue
                turns = session["turns"]
                conversation_id = turns[-1]["response"]["conversation_id"] if turns else None
                turn = {
                    "number": len(turns) + 1,
                    "question": session.pop("next_question"),
                    "request_id": f"synthetic-{state['run_id']}-{session['id']}-{len(turns) + 1}"[:64],
                    "status": "sending",
                }
                turns.append(turn)
                state["last_dispatch_at"] = time.time()
                sent += 1
                save_state(manifest, state)
                task = executor.submit(ask_rag, api_url, turn, conversation_id, timeout)
                tasks[task] = (session, turn)
            for task in as_completed(tasks):
                session, turn = tasks[task]
                turn.update(task.result())
                if turn["number"] == 1 and session.get("relevant") and "response" in turn:
                    turn["source_check"] = check_opening_sources(session, turn["response"])
                if turn["status"] == "error":
                    session["outcome"] = "rag_error"
                    if state.get("request_limit") is not None and (
                        turn.get("http_status") in {429, 503} and turn.get("retry_after_seconds", 0) > 60
                    ):
                        state["quota_blocked"] = True
                save_state(manifest, state)


def wait_until(target: datetime) -> None:
    while (delay := (target - datetime.now(target.tzinfo)).total_seconds()) > 0:
        time.sleep(min(delay, 60))


def recover_interrupted_turns(sessions) -> None:
    for session in sessions:
        if session["turns"] and session["turns"][-1]["status"] == "sending":
            session["turns"][-1].update(status="error", error="envoi interrompu, résultat inconnu")
            session["outcome"] = "rag_error"


def simulate(
    client, state, manifest, day, parallelism, timeout, *,
    api_url=None, send_not_before=None, send_deadline=None,
) -> int:
    api_url = api_url or state.get("api_url")
    sessions = state["sessions"]
    recover_interrupted_turns(sessions)
    save_state(manifest, state)

    for number in range(1, state["max_turns"] + 1):
        active = [session for session in sessions if "outcome" not in session]
        ready = [session for session in active if len(session["turns"]) < number and "next_question" in session]
        if ready:
            if number > 1 and send_not_before is not None:
                wait_until(send_not_before)
            print(f"Tour {number} · {len(ready)} conversations envoyées au RAG", flush=True)
            play_turns(state, manifest, ready, parallelism, timeout, api_url, send_deadline=send_deadline)
        waiting = [
            session for session in active
            if "outcome" not in session and len(session["turns"]) == number and "next_question" not in session
        ]
        if waiting and number < state["max_turns"] and request_quota_reached(
            state, sum(len(session["turns"]) for session in sessions)
        ):
            for session in waiting:
                session["outcome"] = "request_quota_reached"
            save_state(manifest, state)
            continue
        if waiting and number < state["max_turns"] and (
            send_deadline is None or datetime.now(send_deadline.tzinfo) < send_deadline
        ):
            generate_followups(client, state, manifest, day, number + 1, waiting, deadline=send_deadline)
        elif waiting:
            for session in waiting:
                session["outcome"] = "turn_limit" if number == state["max_turns"] else "followup_window_closed"
            save_state(manifest, state)

    report = manifest.parent / "conversations.jsonl"
    write_jsonl(report, sessions)
    errors = sum(session["outcome"] in {"rag_error", "followup_error"} for session in sessions)
    print(f"{len(sessions)} conversations · {errors} en erreur · {report}")
    return int(bool(errors))


@contextmanager
def ssh_rag_url(host: str, remote_port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        local_port = listener.getsockname()[1]

    command = [
        "ssh", "-NT", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
        "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", host,
    ]
    base_url = f"http://127.0.0.1:{local_port}"
    with subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as tunnel:
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and tunnel.poll() is None:
                try:
                    with requests.get(f"{base_url}/health", timeout=2, allow_redirects=False) as response:
                        if response.status_code == 200 and response.json().get("status") == "ok":
                            break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(0.3)
            else:
                raise RuntimeError(f"API RAG inaccessible via {host}:127.0.0.1:{remote_port}.")
            yield f"{base_url}/api/rag/stream"
        finally:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--api-url", help="URL complète de l’endpoint SSE.")
    target.add_argument("--ssh-host", help="Hôte ou alias OpenSSH du VPS, par exemple contabo.")
    parser.add_argument("--ssh-remote-port", type=int, default=8006)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="simulation")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.ssh_host:
        if not re.fullmatch(r"[\w][\w.@-]*", args.ssh_host) or not 1 <= args.ssh_remote_port <= 65535:
            parser.error("hôte ou port SSH invalide")
        api_target = f"ssh://{args.ssh_host}:{args.ssh_remote_port}/api/rag/stream"
    else:
        api_url = urlsplit(args.api_url)
        if api_url.scheme not in {"http", "https"} or not api_url.hostname or not api_url.path.endswith("/api/rag/stream"):
            parser.error("--api-url doit être une URL HTTP(S) finissant par /api/rag/stream")
        api_target = args.api_url
    if not 1 <= args.count < 10_000 or min(args.max_turns, args.parallelism, args.timeout) <= 0:
        parser.error("paramètres de volume ou de délai invalides")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("--run-name doit être un nom de dossier simple")
    day = args.date or datetime.now(ZoneInfo("Europe/Paris")).date()
    directory = args.output_dir / day.isoformat()
    manifest = directory / args.run_name / "simulation.json"
    load_project_env(PROJECT_DIR, override=False)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        parser.error("MISTRAL_API_KEY est requise")

    with Mistral(api_key=api_key) as client:
        if manifest.exists():
            state = json.loads(manifest.read_text(encoding="utf-8"))
            if (state["api_url"], state["max_turns"], len(state["sessions"])) != (
                api_target, args.max_turns, args.count
            ):
                parser.error("utilise les paramètres du run existant ou un autre --run-name")
        else:
            openings = directory / "questions.jsonl"
            if not openings.exists() and generate_questions(client, day, args.video_dir, args.count, directory):
                return 1
            questions = [json.loads(line) for line in openings.read_text(encoding="utf-8").splitlines()]
            if len(questions) < args.count:
                parser.error(f"{len(questions)} questions disponibles pour {args.count} sessions")
            sessions = [{**item, "next_question": item["question"], "turns": []} for item in questions[:args.count]]
            state = {
                "api_url": api_target, "max_turns": args.max_turns,
                "run_id": uuid4().hex[:12], "sessions": sessions,
            }
            manifest.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = ssh_rag_url(args.ssh_host, args.ssh_remote_port) if args.ssh_host else nullcontext(api_target)
            with connection as api_url:
                return simulate(client, state, manifest, day, args.parallelism, args.timeout, api_url=api_url)
        except (RuntimeError, OSError) as exc:
            parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
