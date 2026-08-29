from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from interface.backend.database import (
    ConversationNotFoundError,
    connect_database,
    store_chat_message,
)
from interface.backend.config import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
)
from interface.backend.conversation_memory import remember_conversation_turn
from interface.backend.generation import (
    generate_final_answer,
    select_answer_sources,
)
from interface.backend.answer_judge import judge_final_answer
from interface.backend.analytics_sql import run_analytics_text_to_sql
from interface.backend.orchestration import orchestrate_request
from interface.backend.llm_providers import (
    LLM_MODEL_CATALOG,
    LLMProviderError,
    provider_for_model,
)
from interface.backend.schemas import ChunkSource, ExecutionPlan, RagRequest, RagResponse
from interface.backend.telemetry import current_trace_id, telemetry_status, trace_operation
from interface.backend.utilities import get_llm_client


router = APIRouter()


def has_empty_sql_result(retrieval: dict[str, Any]) -> bool:
    direct_lookup = retrieval.get("direct_lookup") or {}
    if direct_lookup and direct_lookup.get("result_count") == 0:
        return True
    return any(
        item.get("source") == "sql" and item.get("result_count") == 0
        for item in retrieval.get("multi_source_actions", [])
    )


def should_run_answer_judge(
    route: str | None,
    action: str,
    retrieval: dict[str, Any],
) -> bool:
    if route == "direct":
        return False
    if action in {"clarify", "abstain"}:
        return True
    return bool(
        retrieval.get("sql_sub_intent") == "analytics"
        or has_empty_sql_result(retrieval)
    )


def run_answer_judge(
    client: Any,
    model: str | None,
    question: str,
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
    answer: str,
    action: str,
    *,
    pass_name: str,
) -> dict[str, Any]:
    with trace_operation(
        "rag.answer_judge",
        kind="GUARDRAIL",
        input_value={
            "pass": pass_name,
            "question": question,
            "route": retrieval.get("route"),
            "sql_sub_intent": retrieval.get("sql_sub_intent"),
            "source_count": len(sources),
            "answer": answer,
            "action": action,
        },
    ) as judge_span:
        verdict = judge_final_answer(
            client,
            model,
            question,
            retrieval,
            sources,
            answer,
            action,
        )
        judge_span.set_output(verdict)
    return verdict


@router.get("/llm-models")
def llm_models() -> dict[str, Any]:
    provider = "mistral"
    provider_config = LLM_MODEL_CATALOG[provider]
    models = [
        {
            "provider": provider,
            "provider_label": provider_config["label"],
            "label": model_label,
            "id": model_id,
        }
        for model_label, model_id in provider_config["models"]
    ]
    return {
        "models": models,
        "defaults": {
            "reformulationModel": DEFAULT_REFORMULATION_MODEL,
            "plannerModel": DEFAULT_PLANNER_MODEL,
            "answerModel": DEFAULT_GENERATION_MODEL,
        },
    }


@router.get("/video-thumbnails", response_model=list[str])
def video_thumbnails() -> list[str]:
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT thumbnail_medium_url
                FROM videos
                WHERE thumbnail_medium_url IS NOT NULL
                  AND thumbnail_medium_url <> ''
                ORDER BY published_at DESC NULLS LAST, id DESC
                LIMIT 32
                """
            )
            return [row[0] for row in cursor.fetchall()]


def execute_rag(payload: RagRequest) -> RagResponse:
    if not payload.useSql:
        raise HTTPException(status_code=400, detail="Le backend actuel attend useSql=true pour interroger la base.")

    try:
        with trace_operation(
            "rag.orchestration",
            kind="AGENT",
            input_value=payload.model_dump(),
        ) as orchestration_span:
            answer, sources, retrieval = orchestrate_request(payload)
            orchestration_span.set_output(
                {
                    "answer_provided": bool(answer),
                    "sources": sources,
                    "retrieval": retrieval,
                }
            )

        answer_client = get_llm_client()
        answer_trace: dict[str, str] = {}
        if not answer:
            with trace_operation(
                "rag.generation",
                kind="CHAIN",
                input_value={
                    "question": retrieval.get("contextual_question", payload.question),
                    "model": retrieval["answer_model"],
                    "sources": sources,
                },
            ) as generation_span:
                answer = generate_final_answer(
                    answer_client,
                    retrieval.get("contextual_question", payload.question),
                    retrieval["answer_model"],
                    retrieval,
                    sources,
                    answer_trace,
                    payload.answerPrompt,
                )
                generation_span.set_output(
                    {
                        "answer": answer,
                        "trace": answer_trace,
                    }
                )
        else:
            answer_trace["action"] = "answer"

        answer_action = answer_trace.get("action", "abstain")
        route = retrieval.get("route") or retrieval.get("retrieval_mode")
        if should_run_answer_judge(route, answer_action, retrieval):
            initial_verdict = run_answer_judge(
                answer_client,
                retrieval.get("answer_model"),
                payload.question,
                retrieval,
                sources,
                answer,
                answer_action,
                pass_name="initial",
            )
            judge_trace: dict[str, Any] = {
                "initial": initial_verdict,
                "retry_attempted": False,
                "retry_performed": False,
            }
            if not initial_verdict.get("valid", True):
                retry_stage = initial_verdict.get("retry_stage")
                judge_feedback = (
                    str(initial_verdict.get("correction") or "").strip()
                    or str(initial_verdict.get("reason") or "").strip()
                )
                retry_performed = False
                if retry_stage == "sql" and retrieval.get("sql_sub_intent") == "analytics":
                    judge_trace["retry_attempted"] = True
                    execution_plan = ExecutionPlan.model_validate(
                        retrieval.get("execution_plan") or {}
                    )
                    with trace_operation(
                        "rag.answer_judge.sql_retry",
                        kind="CHAIN",
                        input_value={
                            "execution_plan": execution_plan.model_dump(),
                            "feedback": judge_feedback,
                        },
                    ) as retry_span:
                        retry_sources, retry_trace = run_analytics_text_to_sql(
                            execution_plan,
                            answer_client,
                            retrieval.get("planner_model"),
                            database_persons=retrieval.get("resolved_persons", []),
                            database_companies=retrieval.get("resolved_companies", []),
                            correction_feedback=judge_feedback,
                        )
                        retry_span.set_output({**retry_trace, "results": retry_sources})
                    judge_trace["sql_retry"] = retry_trace
                    if retry_trace.get("status") == "executed":
                        sources = retry_sources
                        retrieval["sql_query"] = retry_trace.get("sql")
                        retrieval["final_k"] = len(sources)
                        retrieval["judge_sql_retry"] = retry_trace
                        if isinstance(retrieval.get("multi_source_actions"), list):
                            retrieval["multi_source_actions"].append(
                                {
                                    "action": len(retrieval["multi_source_actions"]) + 1,
                                    "source": "sql",
                                    "operation": "judge_retry_analytics_sql",
                                    "status": "completed",
                                    "request": {
                                        "sql": retry_trace.get("sql"),
                                        "params": retry_trace.get("params", []),
                                    },
                                    "response": sources,
                                    "result_count": len(sources),
                                }
                            )
                        retry_performed = True
                elif retry_stage == "generation":
                    judge_trace["retry_attempted"] = True
                    retry_performed = True

                if retry_performed:
                    judge_trace["retry_performed"] = True
                    answer_trace = {}
                    with trace_operation(
                        "rag.generation.retry",
                        kind="CHAIN",
                        input_value={
                            "question": retrieval.get("contextual_question", payload.question),
                            "model": retrieval.get("answer_model"),
                            "sources": sources,
                            "feedback": judge_feedback,
                        },
                    ) as retry_generation_span:
                        answer = generate_final_answer(
                            answer_client,
                            retrieval.get("contextual_question", payload.question),
                            retrieval.get("answer_model"),
                            retrieval,
                            sources,
                            answer_trace,
                            payload.answerPrompt,
                            judge_feedback=judge_feedback,
                        )
                        retry_generation_span.set_output(
                            {"answer": answer, "trace": answer_trace}
                        )
                    answer_action = answer_trace.get("action", "abstain")
                    judge_trace["final"] = run_answer_judge(
                        answer_client,
                        retrieval.get("answer_model"),
                        payload.question,
                        retrieval,
                        sources,
                        answer,
                        answer_action,
                        pass_name="final",
                    )
            retrieval["answer_judge"] = judge_trace

        retrieval["answer_action"] = answer_action
        if answer_action in {"answer", "clarify"}:
            answer, carousel_sources = select_answer_sources(answer, sources)
        else:
            carousel_sources = []
        retrieval["answer_source_indexes"] = [
            index for index, source in enumerate(sources, start=1) if source in carousel_sources
        ]
        trace_id = current_trace_id()
        retrieval["telemetry"] = {
            "trace_id": trace_id,
            "project": telemetry_status().get("project"),
        }
        with trace_operation(
            "rag.store_message",
            kind="TOOL",
            input_value={
                "conversation_id": payload.conversationId,
                "trace_id": trace_id,
            },
        ) as storage_span:
            conversation_id, message_id = store_chat_message(
                conversation_id=payload.conversationId,
                user_message=payload.question,
                answer_message=answer,
                trace_id=trace_id,
            )
            memory_update = remember_conversation_turn(
                conversation_id,
                message_id,
                payload.question,
                answer,
            )
            storage_span.set_session_id(conversation_id)
            storage_span.set_output(
                {"conversation_id": conversation_id, "message_id": message_id, "memory": memory_update}
            )
        retrieval["conversation_memory"] = memory_update
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RagResponse(
        conversation_id=conversation_id,
        message_id=message_id,
        answer=answer,
        action=answer_action,
        sources=[ChunkSource(**source) for source in carousel_sources],
        retrieval=retrieval,
    )


def run_rag(payload: RagRequest) -> RagResponse:
    with trace_operation(
        "rag.request",
        kind="CHAIN",
        input_value={
            "question": payload.question,
            "conversation_id": payload.conversationId,
            "reformulation_model": payload.reformulationModel,
            "planner_model": payload.plannerModel,
            "answer_model": payload.answerModel,
            "embedding_model": payload.embeddingModel,
            "rerank_model": payload.rerankModel,
            "use_rerank": payload.useRerank,
            "request": payload.model_dump(),
        },
    ) as request_span:
        request_span.set_session_id(payload.conversationId)
        response = execute_rag(payload)
        request_span.set_session_id(response.conversation_id)
        request_span.set_attribute("rag.action", response.action)
        request_span.set_attribute("rag.source_count", len(response.sources))
        request_span.set_output(response.model_dump())
        return response


def ensure_interface_uses_mistral(payload: RagRequest) -> None:
    for field_name in (
        "reformulationModel",
        "plannerModel",
        "answerModel",
    ):
        model = getattr(payload, field_name)
        try:
            provider = provider_for_model(model)
        except LLMProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if provider != "mistral":
            raise HTTPException(
                status_code=400,
                detail=(
                    "L'interface RAG utilise uniquement Mistral. "
                    "Les autres fournisseurs sont reserves a "
                    "utils/run_phoenix_experiment.py."
                ),
            )


@router.post("/rag", response_model=RagResponse)
def rag(payload: RagRequest) -> RagResponse:
    ensure_interface_uses_mistral(payload)
    return run_rag(payload)
