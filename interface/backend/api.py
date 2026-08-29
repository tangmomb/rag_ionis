from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from interface.backend.database import (
    ConversationNotFoundError,
    connect_database,
    create_conversation,
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
from interface.backend.orchestration import orchestrate_request
from interface.backend.llm_providers import (
    LLM_MODEL_CATALOG,
    LLMProviderError,
    provider_for_model,
)
from interface.backend.schemas import ChunkSource, RagRequest, RagResponse
from interface.backend.telemetry import current_trace_id, telemetry_status, trace_operation
from interface.backend.utilities import get_llm_client, normalize_model_name


router = APIRouter()


@router.get("/llm-models")
def llm_models() -> dict[str, Any]:
    models = []
    for provider, provider_config in LLM_MODEL_CATALOG.items():
        models.extend(
            {
                "provider": provider,
                "provider_label": provider_config["label"],
                "label": model_label,
                "id": model_id,
            }
            for model_label, model_id in provider_config["models"]
        )
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
        if payload.conversationId is None:
            payload.conversationId = create_conversation()
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
        answer_trace: dict[str, Any] = {}
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
        retrieval["answer_action"] = answer_action
        if answer_action in {"answer", "clarify"}:
            answer, carousel_sources = select_answer_sources(
                answer,
                sources,
                answer_trace.get("source_indexes"),
            )
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
                topic_id=(retrieval.get("conversation_topic") or {}).get("topic_id"),
            )
            with trace_operation(
                "rag.conversation_memory.summary",
                kind="CHAIN",
                input_value={
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "question": payload.question,
                },
            ) as memory_span:
                memory_update = remember_conversation_turn(
                    conversation_id,
                    message_id,
                    payload.question,
                    answer,
                    summary_client=answer_client,
                    summary_model=retrieval.get("answer_model") or DEFAULT_GENERATION_MODEL,
                    topic_id=(retrieval.get("conversation_topic") or {}).get("topic_id"),
                )
                memory_span.set_output(memory_update)
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


def validate_step_models(payload: RagRequest) -> None:
    """Normalize and validate each independently configured RAG LLM."""
    defaults = {
        "reformulationModel": DEFAULT_REFORMULATION_MODEL,
        "plannerModel": DEFAULT_PLANNER_MODEL,
        "answerModel": DEFAULT_GENERATION_MODEL,
    }
    for field_name, default in defaults.items():
        model = normalize_model_name(getattr(payload, field_name), default)
        try:
            provider_for_model(model)
        except LLMProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        setattr(payload, field_name, model)


@router.post("/rag", response_model=RagResponse)
def rag(payload: RagRequest) -> RagResponse:
    validate_step_models(payload)
    return run_rag(payload)
