from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from interface.backend.api import run_rag
from interface.backend.database import connect_database, ensure_chat_schema
from interface.backend.schemas import RagRequest
from interface.backend.telemetry import (
    configure_telemetry,
    current_trace_id,
    shutdown_telemetry,
    trace_operation,
)


@dataclass(frozen=True)
class ReplayMessage:
    message_id: int
    user_message: str
    answer_message: str
    trace_id: str | None


@dataclass(frozen=True)
class ReplayContext:
    original_trace_id: str
    original_conversation_id: int
    target_message_id: int
    target_question: str
    history: tuple[ReplayMessage, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clone le contexte exact précédant une trace Phoenix puis rejoue sa "
            "question dans une nouvelle conversation tracée."
        )
    )
    parser.add_argument("--trace-id", required=True, help="Trace Phoenix à rejouer.")
    parser.add_argument(
        "--mode",
        choices=("exact-context",),
        default="exact-context",
        help="Mode de rejeu. exact-context conserve les anciennes réponses.",
    )
    parser.add_argument(
        "--phoenix-base-url",
        default=os.getenv("PHOENIX_BASE_URL", "http://localhost:6006"),
        help="URL HTTP de Phoenix.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("PHOENIX_PROJECT_NAME", "rag-ionis"),
        help="Nom ou identifiant du projet Phoenix.",
    )
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le contexte qui serait cloné sans écrire ni appeler le RAG.",
    )
    return parser


def _json_object(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} n'est pas un objet JSON valide.") from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise RuntimeError(f"{label} est absent ou n'est pas un objet JSON.")


def fetch_trace_request(
    client: Any,
    *,
    project_identifier: str,
    trace_id: str,
    timeout: int,
) -> tuple[dict[str, Any], str | None]:
    spans = client.spans.get_spans(
        project_identifier=project_identifier,
        trace_ids=[trace_id],
        limit=1_000,
        timeout=timeout,
    )
    if not spans:
        raise RuntimeError(
            f"Trace Phoenix introuvable dans {project_identifier!r}: {trace_id}"
        )

    roots = [span for span in spans if span.get("parent_id") is None]
    if len(roots) != 1:
        raise RuntimeError(
            f"La trace {trace_id} doit avoir exactement un span racine; trouvé: {len(roots)}."
        )
    attributes = dict(roots[0].get("attributes") or {})
    traced_input = _json_object(
        attributes.get("input.value"),
        label=f"input.value de la trace {trace_id}",
    )
    full_request = traced_input.get("request")
    request = dict(full_request) if isinstance(full_request, Mapping) else traced_input
    session_id = attributes.get("session.id")
    return request, str(session_id) if session_id is not None else None


def load_replay_context(
    trace_id: str,
    *,
    expected_conversation_id: str | None = None,
) -> ReplayContext:
    ensure_chat_schema()
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, conversation_id, user_message
                FROM chat.messages
                WHERE trace_id = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (trace_id,),
            )
            target = cursor.fetchone()
            if target is None:
                raise RuntimeError(
                    f"Aucun message PostgreSQL n'est associé à la trace {trace_id}."
                )
            target_message_id = int(target[0])
            conversation_id = int(target[1])
            target_question = str(target[2] or "").strip()

            if (
                expected_conversation_id is not None
                and str(conversation_id) != expected_conversation_id
            ):
                raise RuntimeError(
                    "La session Phoenix et la conversation PostgreSQL ne correspondent "
                    f"pas: {expected_conversation_id} != {conversation_id}."
                )

            cursor.execute(
                """
                SELECT id, user_message, answer_message, trace_id
                FROM chat.messages
                WHERE conversation_id = %s
                  AND id < %s
                ORDER BY id ASC
                """,
                (conversation_id, target_message_id),
            )
            history = tuple(
                ReplayMessage(
                    message_id=int(row[0]),
                    user_message=str(row[1] or ""),
                    answer_message=str(row[2] or ""),
                    trace_id=str(row[3]) if row[3] is not None else None,
                )
                for row in cursor.fetchall()
            )

    return ReplayContext(
        original_trace_id=trace_id,
        original_conversation_id=conversation_id,
        target_message_id=target_message_id,
        target_question=target_question,
        history=history,
    )


def seed_replay_conversation(context: ReplayContext) -> int:
    ensure_chat_schema()
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO chat.conversations DEFAULT VALUES RETURNING id")
            replay_conversation_id = int(cursor.fetchone()[0])
            for message in context.history:
                cursor.execute(
                    """
                    INSERT INTO chat.messages (
                        conversation_id,
                        user_message,
                        answer_message,
                        trace_id
                    )
                    VALUES (%s, %s, %s, NULL)
                    """,
                    (
                        replay_conversation_id,
                        message.user_message,
                        message.answer_message,
                    ),
                )
        connection.commit()
    return replay_conversation_id


def build_replay_request(
    traced_request: Mapping[str, Any],
    context: ReplayContext,
    replay_conversation_id: int,
) -> RagRequest:
    defaults = RagRequest(question=context.target_question).model_dump()
    aliases = {
        "apiUrl": ("apiUrl", "api_url"),
        "reformulationModel": ("reformulationModel", "reformulation_model"),
        "plannerModel": ("plannerModel", "planner_model"),
        "answerModel": ("answerModel", "answer_model"),
        "reformulationPrompt": ("reformulationPrompt", "reformulation_prompt"),
        "plannerPrompt": ("plannerPrompt", "planner_prompt"),
        "answerPrompt": ("answerPrompt", "answer_prompt"),
        "embeddingModel": ("embeddingModel", "embedding_model"),
        "rerankModel": ("rerankModel", "rerank_model"),
        "useSql": ("useSql", "use_sql"),
        "useRerank": ("useRerank", "use_rerank"),
        "topK": ("topK", "top_k"),
        "finalK": ("finalK", "final_k"),
    }
    payload = dict(defaults)
    for field_name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in traced_request:
                payload[field_name] = traced_request[candidate]
                break
    payload["question"] = context.target_question
    payload["conversationId"] = replay_conversation_id
    return RagRequest.model_validate(payload)


def replay_context_summary(context: ReplayContext) -> dict[str, Any]:
    return {
        "original_trace_id": context.original_trace_id,
        "original_conversation_id": context.original_conversation_id,
        "target_message_id": context.target_message_id,
        "target_question": context.target_question,
        "history_message_count": len(context.history),
        "history": [
            {
                "message_id": message.message_id,
                "user_message": message.user_message,
                "answer_preview": message.answer_message[:240],
                "trace_id": message.trace_id,
            }
            for message in context.history
        ],
    }


def compact_response(response: Any) -> dict[str, Any]:
    return {
        "conversation_id": response.conversation_id,
        "message_id": response.message_id,
        "answer": response.answer,
        "action": response.action,
        "sources": [
            {
                "chunk_id": source.chunk_id,
                "video_title": source.video_title,
                "video_url": source.video_url,
            }
            for source in response.sources
        ],
    }


def run_replay(
    args: argparse.Namespace,
    *,
    shutdown_after: bool = True,
) -> dict[str, Any]:
    from phoenix.client import Client

    phoenix_base_url = args.phoenix_base_url.rstrip("/")
    client = Client(
        base_url=phoenix_base_url,
        api_key=os.getenv("PHOENIX_API_KEY"),
    )
    traced_request, session_id = fetch_trace_request(
        client,
        project_identifier=args.project,
        trace_id=args.trace_id,
        timeout=args.timeout,
    )
    context = load_replay_context(
        args.trace_id,
        expected_conversation_id=session_id,
    )
    summary = replay_context_summary(context)
    if args.dry_run:
        return {"status": "dry_run", "mode": args.mode, **summary}

    os.environ["PHOENIX_BASE_URL"] = phoenix_base_url
    os.environ["PHOENIX_PROJECT_NAME"] = args.project
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = f"{phoenix_base_url}/v1/traces"
    configure_telemetry()
    try:
        with trace_operation(
            "rag.replay",
            kind="CHAIN",
            input_value={"mode": args.mode, **summary},
            attributes={
                "rag.replay.mode": args.mode,
                "rag.replay.original_trace_id": args.trace_id,
                "rag.replay.original_conversation_id": context.original_conversation_id,
            },
        ) as replay_span:
            replay_trace_id = current_trace_id()
            with trace_operation(
                "rag.replay.seed_history",
                kind="TOOL",
                input_value={
                    "original_trace_id": args.trace_id,
                    "history_message_count": len(context.history),
                    "original_message_ids": [
                        message.message_id for message in context.history
                    ],
                },
            ) as seed_span:
                replay_conversation_id = seed_replay_conversation(context)
                seed_span.set_session_id(replay_conversation_id)
                seed_span.set_output(
                    {
                        "replay_conversation_id": replay_conversation_id,
                        "seeded_message_count": len(context.history),
                    }
                )

            replay_span.set_session_id(replay_conversation_id)
            request = build_replay_request(
                traced_request,
                context,
                replay_conversation_id,
            )
            response = run_rag(request)
            result = {
                "status": "replayed",
                "mode": args.mode,
                "original_trace_id": args.trace_id,
                "replay_trace_id": replay_trace_id,
                "original_conversation_id": context.original_conversation_id,
                "replay_conversation_id": replay_conversation_id,
                "seeded_message_count": len(context.history),
                "request": request.model_dump(),
                "response": compact_response(response),
                "phoenix_project": args.project,
                "phoenix_base_url": phoenix_base_url,
                "phoenix_url": f"{phoenix_base_url}/projects",
            }
            replay_span.set_output(result)
            return result
    finally:
        if shutdown_after:
            shutdown_telemetry()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_replay(args)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
