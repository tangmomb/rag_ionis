from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from fastapi import APIRouter, HTTPException
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from interface.backend.answer_evaluation import (
    correction_loop_enabled,
    evaluate_answer_shadow,
    shadow_evaluation_enabled,
    shadow_evaluation_model,
)
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
    MAX_FINAL_K,
    MAX_TOP_K,
)
from interface.backend.conversation_memory import (
    remember_conversation_json_turn,
)
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


class RagResponseState(TypedDict, total=False):
    """State passed between the existing response-pipeline steps."""

    payload: dict[str, Any]
    answer: str
    sources: list[dict[str, Any]]
    retrieval: dict[str, Any]
    answer_trace: dict[str, Any]
    shadow_evaluation: dict[str, Any]
    answer_action: str
    carousel_sources: list[dict[str, Any]]
    conversation_id: int
    message_id: int
    correction_count: int
    correction_requested: bool
    shadow_evaluation_history: list[dict[str, Any]]


@dataclass
class RagResponseContext:
    answer_client: Any = None
    shadow_evaluation_enabled_override: bool | None = None
    shadow_evaluation_model_override: str | None = None
    shadow_evaluation_sink: dict[str, Any] | None = None
    correction_loop_enabled_override: bool | None = None


def _runtime_context(runtime: Runtime[RagResponseContext]) -> RagResponseContext:
    return runtime.context or RagResponseContext()


def _response_payload(state: RagResponseState) -> RagRequest:
    return RagRequest.model_validate(state["payload"])


def _orchestrate_response(state: RagResponseState) -> dict[str, Any]:
    payload = _response_payload(state)
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
    return {
        "answer": answer,
        "sources": sources,
        "retrieval": retrieval,
    }


def _generate_response(
    state: RagResponseState,
    runtime: Runtime[RagResponseContext],
) -> dict[str, Any]:
    payload = _response_payload(state)
    retrieval = state["retrieval"]
    sources = state["sources"]
    answer = state["answer"]
    answer_client = get_llm_client()
    context = _runtime_context(runtime)
    context.answer_client = answer_client
    answer_trace: dict[str, Any] = {}
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
            retrieval.get("answer_prompt_override") or payload.answerPrompt,
        )
        generation_span.set_output(
            {
                "answer": answer,
                "trace": answer_trace,
            }
        )
    return {
        "answer": answer,
        "answer_trace": answer_trace,
    }


def _accept_precomputed_response(
    state: RagResponseState,
    runtime: Runtime[RagResponseContext],
) -> dict[str, Any]:
    _runtime_context(runtime).answer_client = get_llm_client()
    return {
        "answer_trace": {"action": "answer"},
    }


def _generation_route(state: RagResponseState) -> Literal["generate", "accept_precomputed"]:
    return "accept_precomputed" if state["answer"] else "generate"


def _evaluate_response(
    state: RagResponseState,
    runtime: Runtime[RagResponseContext],
) -> dict[str, Any]:
    payload = _response_payload(state)
    retrieval = state["retrieval"]
    context = _runtime_context(runtime)
    route = retrieval.get("route") or retrieval.get("retrieval_mode")
    evaluation_enabled = context.shadow_evaluation_enabled_override
    if evaluation_enabled is None:
        evaluation_enabled = shadow_evaluation_enabled()
    evaluation_model = (
        context.shadow_evaluation_model_override
        or shadow_evaluation_model(str(retrieval.get("answer_model") or ""))
    )
    evaluation_client = context.answer_client or get_llm_client()
    context.answer_client = evaluation_client
    input_value = {
        "mode": "shadow",
        "enabled": evaluation_enabled,
        "model": evaluation_model,
        "route": route,
        "question": retrieval.get("contextual_question", payload.question),
        "action": state["answer_trace"].get("action", "abstain"),
        "source_count": len(state["sources"]),
    }
    with trace_operation(
        "rag.shadow_evaluation",
        kind="EVALUATOR",
        input_value=input_value,
    ) as evaluation_span:
        if not input_value["enabled"]:
            evaluation = {
                "enabled": False,
                "mode": "shadow",
                "verdict": "not_run",
                "issue": "none",
                "status": "not_run",
                "reason": "disabled",
            }
        elif route == "direct":
            evaluation = {
                "enabled": True,
                "mode": "shadow",
                "verdict": "not_applicable",
                "issue": "none",
                "status": "not_applicable",
                "reason": "direct_answer",
            }
        elif evaluation_client is None or not evaluation_model:
            evaluation = {
                "enabled": True,
                "mode": "shadow",
                "verdict": "not_run",
                "issue": "none",
                "status": "not_run",
                "reason": "missing_client_or_model",
            }
        else:
            try:
                evaluation = evaluate_answer_shadow(
                    evaluation_client,
                    evaluation_model,
                    retrieval.get("contextual_question", payload.question),
                    state["answer"],
                    state["answer_trace"].get("action", "abstain"),
                    state["sources"],
                )
            except Exception as exc:  # Le mode shadow ne bloque jamais la réponse.
                evaluation = {
                    "enabled": True,
                    "mode": "shadow",
                    "verdict": "error",
                    "issue": "none",
                    "status": "error",
                    "reason": str(exc),
                }
        evaluation_span.set_output(evaluation)
    checkpoint_evaluation = {
        key: value
        for key, value in evaluation.items()
        if key not in {"prompt", "response_raw"}
    }
    if context.shadow_evaluation_sink is not None:
        context.shadow_evaluation_sink.clear()
        context.shadow_evaluation_sink.update(checkpoint_evaluation)
    correction_enabled = context.correction_loop_enabled_override
    if correction_enabled is None:
        correction_enabled = correction_loop_enabled()
    correction_requested = bool(
        correction_enabled
        and checkpoint_evaluation.get("verdict") == "needs_correction"
        and state.get("correction_count", 0) < 1
        and route != "direct"
    )
    return {
        "shadow_evaluation": checkpoint_evaluation,
        "correction_requested": correction_requested,
    }


def _correction_prompt(
    base_prompt: str | None,
    evaluation: dict[str, Any],
    previous_answer: str,
) -> str:
    correction_context = {
        "issue": evaluation.get("issue"),
        "reason": evaluation.get("reason"),
        "suggested_correction": evaluation.get("suggested_correction"),
        "previous_answer": previous_answer,
    }
    return (
        f"{base_prompt or ''}\n\n"
        "Correction interne obligatoire : produis une nouvelle réponse en "
        "corrigeant uniquement le problème décrit ci-dessous. Reste strictement "
        "fondé sur les mêmes sources et n'invente aucune information.\n"
        f"{correction_context}"
    ).strip()


def _correct_response(
    state: RagResponseState,
    runtime: Runtime[RagResponseContext],
) -> dict[str, Any]:
    payload = _response_payload(state)
    retrieval = dict(state["retrieval"])
    evaluation = dict(state["shadow_evaluation"])
    previous_answer = state["answer"]
    answer_trace: dict[str, Any] = {}
    correction_count = state.get("correction_count", 0) + 1
    correction_metadata: dict[str, Any] = {
        "attempted": True,
        "count": correction_count,
        "issue": evaluation.get("issue"),
        "strategy": "regenerate_answer",
    }
    with trace_operation(
        "rag.correction",
        kind="CHAIN",
        input_value={
            "question": retrieval.get("contextual_question", payload.question),
            "model": retrieval.get("answer_model"),
            "evaluation": evaluation,
            "attempt": correction_count,
        },
    ) as correction_span:
        try:
            context = _runtime_context(runtime)
            answer_client = context.answer_client or get_llm_client()
            context.answer_client = answer_client
            corrected_answer = generate_final_answer(
                answer_client,
                retrieval.get("contextual_question", payload.question),
                retrieval["answer_model"],
                retrieval,
                state["sources"],
                answer_trace,
                _correction_prompt(payload.answerPrompt, evaluation, previous_answer),
            )
            correction_metadata["succeeded"] = True
        except Exception as exc:  # Une correction ne doit jamais perdre la réponse initiale.
            corrected_answer = previous_answer
            answer_trace = dict(state["answer_trace"])
            correction_metadata.update({"succeeded": False, "error": str(exc)})
        correction_span.set_output(
            {
                "succeeded": correction_metadata["succeeded"],
                "answer": corrected_answer,
                "trace": answer_trace,
            }
        )
    retrieval["correction"] = correction_metadata
    return {
        "answer": corrected_answer,
        "answer_trace": answer_trace,
        "retrieval": retrieval,
        "correction_count": correction_count,
        "correction_requested": False,
        "shadow_evaluation_history": [
            *state.get("shadow_evaluation_history", []),
            evaluation,
        ],
    }


def _post_evaluation_route(
    state: RagResponseState,
) -> Literal["correct", "retry_retrieval", "expand_retrieval", "finalize"]:
    if not state.get("correction_requested", False):
        return "finalize"
    issue = state.get("shadow_evaluation", {}).get("issue")
    if issue == "bad_retrieval":
        return "retry_retrieval"
    if issue == "insufficient_sources":
        return "expand_retrieval"
    return "correct"


def _retrieval_correction_prompt(
    base_prompt: str | None,
    strategy: str,
    evaluation: dict[str, Any],
) -> str:
    if strategy == "retry_retrieval":
        instruction = (
            "Le retrieval précédent était hors sujet. Produis une requête de "
            "recherche différente, plus précise, tout en conservant exactement "
            "le sens de la question utilisateur."
        )
    else:
        instruction = (
            "Les sources précédentes étaient incomplètes. Produis une requête de "
            "recherche plus large couvrant toutes les facettes de la question, "
            "sans en modifier le sens."
        )
    return (
        f"{base_prompt or ''}\n\n{instruction}\n"
        f"Diagnostic interne : issue={evaluation.get('issue')}; "
        f"reason={evaluation.get('reason')}; "
        f"suggestion={evaluation.get('suggested_correction')}"
    ).strip()


def _retrieve_for_correction(
    state: RagResponseState,
    strategy: Literal["retry_retrieval", "expand_retrieval"],
) -> dict[str, Any]:
    payload = _response_payload(state)
    evaluation = dict(state["shadow_evaluation"])
    correction_count = state.get("correction_count", 0) + 1
    if strategy == "expand_retrieval":
        top_k = MAX_TOP_K
        final_k = MAX_FINAL_K
    else:
        top_k = min(MAX_TOP_K, max(payload.topK + 10, payload.topK * 2))
        final_k = min(MAX_FINAL_K, max(payload.finalK + 5, payload.finalK * 2))
    corrected_payload = payload.model_copy(
        update={
            "topK": top_k,
            "finalK": final_k,
            "plannerPrompt": _retrieval_correction_prompt(
                payload.plannerPrompt,
                strategy,
                evaluation,
            ),
        }
    )
    correction_metadata: dict[str, Any] = {
        "attempted": True,
        "count": correction_count,
        "issue": evaluation.get("issue"),
        "strategy": strategy,
        "top_k": top_k,
        "final_k": final_k,
    }
    with trace_operation(
        f"rag.correction.{strategy}",
        kind="RETRIEVER",
        input_value={
            "question": payload.question,
            "evaluation": evaluation,
            "top_k": top_k,
            "final_k": final_k,
        },
    ) as correction_span:
        try:
            answer, sources, retrieval = orchestrate_request(corrected_payload)
            retrieval = dict(retrieval)
            retrieval["answer_prompt_override"] = _correction_prompt(
                payload.answerPrompt,
                evaluation,
                state["answer"],
            )
            correction_metadata["succeeded"] = True
        except Exception as exc:  # Le retrieval correctif ne bloque jamais la réponse.
            answer = state["answer"]
            sources = state["sources"]
            retrieval = dict(state["retrieval"])
            correction_metadata.update({"succeeded": False, "error": str(exc)})
        retrieval["correction"] = correction_metadata
        correction_span.set_output(
            {
                "succeeded": correction_metadata["succeeded"],
                "answer_provided": bool(answer),
                "source_count": len(sources),
            }
        )
    return {
        "answer": answer,
        "sources": sources,
        "retrieval": retrieval,
        "correction_count": correction_count,
        "correction_requested": False,
        "shadow_evaluation_history": [
            *state.get("shadow_evaluation_history", []),
            evaluation,
        ],
    }


def _retry_retrieval(state: RagResponseState) -> dict[str, Any]:
    return _retrieve_for_correction(state, "retry_retrieval")


def _expand_retrieval(state: RagResponseState) -> dict[str, Any]:
    return _retrieve_for_correction(state, "expand_retrieval")


def _post_retrieval_route(
    state: RagResponseState,
) -> Literal["generate", "accept_precomputed", "evaluate"]:
    correction = state.get("retrieval", {}).get("correction", {})
    if not correction.get("succeeded", False):
        return "evaluate"
    return _generation_route(state)


def _finalize_response(state: RagResponseState) -> dict[str, Any]:
    answer = state["answer"].replace("\x00", "")
    sources = state["sources"]
    retrieval = state["retrieval"]
    answer_trace = state["answer_trace"]
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
    if state.get("shadow_evaluation_history"):
        retrieval["shadow_evaluation_history"] = state[
            "shadow_evaluation_history"
        ]
    return {
        "answer": answer,
        "answer_action": answer_action,
        "carousel_sources": carousel_sources,
    }


def _persist_response(
    state: RagResponseState,
    runtime: Runtime[RagResponseContext],
) -> dict[str, Any]:
    payload = _response_payload(state)
    answer = state["answer"]
    retrieval = state["retrieval"]
    answer_client = _runtime_context(runtime).answer_client or get_llm_client()
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
            topic_id=None,
        )
        with trace_operation(
            "rag.conversation_memory.update",
            kind="CHAIN",
            input_value={
                "conversation_id": conversation_id,
                "message_id": message_id,
                "question": payload.question,
            },
        ) as memory_span:
            memory_update = remember_conversation_json_turn(
                conversation_id,
                payload.question,
                answer,
                retrieval.get("question_reformulation") or {},
                summary_client=answer_client,
                summary_model=retrieval.get("answer_model") or DEFAULT_GENERATION_MODEL,
            )
            memory_span.set_output(memory_update)
        storage_span.set_session_id(conversation_id)
        storage_span.set_output(
            {"conversation_id": conversation_id, "message_id": message_id, "memory": memory_update}
        )
    retrieval["conversation_memory"] = memory_update
    return {
        "conversation_id": conversation_id,
        "message_id": message_id,
    }


def build_rag_response_graph():
    graph = StateGraph(
        RagResponseState,
        context_schema=RagResponseContext,
    )
    graph.add_node("orchestrate", _orchestrate_response)
    graph.add_node("generate", _generate_response)
    graph.add_node("accept_precomputed", _accept_precomputed_response)
    graph.add_node("evaluate", _evaluate_response)
    graph.add_node("correct", _correct_response)
    graph.add_node("retry_retrieval", _retry_retrieval)
    graph.add_node("expand_retrieval", _expand_retrieval)
    graph.add_node("finalize", _finalize_response)
    graph.add_node("persist", _persist_response)
    graph.add_edge(START, "orchestrate")
    graph.add_conditional_edges("orchestrate", _generation_route)
    graph.add_edge("generate", "evaluate")
    graph.add_edge("accept_precomputed", "evaluate")
    graph.add_conditional_edges("evaluate", _post_evaluation_route)
    graph.add_edge("correct", "evaluate")
    graph.add_conditional_edges("retry_retrieval", _post_retrieval_route)
    graph.add_conditional_edges("expand_retrieval", _post_retrieval_route)
    graph.add_edge("finalize", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


RAG_RESPONSE_GRAPH = build_rag_response_graph()


@router.get("/llm-models")
def llm_models() -> dict[str, Any]:
    provider_config = LLM_MODEL_CATALOG["openai"]
    models = [
        {
            "provider": "openai",
            "provider_label": provider_config["label"],
            "label": "Luna",
            "id": DEFAULT_GENERATION_MODEL,
        }
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


def execute_rag(
    payload: RagRequest,
    *,
    shadow_evaluation_enabled_override: bool | None = None,
    shadow_evaluation_model_override: str | None = None,
    shadow_evaluation_sink: dict[str, Any] | None = None,
    correction_loop_enabled_override: bool | None = None,
) -> RagResponse:
    if not payload.useSql:
        raise HTTPException(status_code=400, detail="Le backend actuel attend useSql=true pour interroger la base.")

    try:
        if payload.conversationId is None:
            payload.conversationId = create_conversation()
        result = RAG_RESPONSE_GRAPH.invoke(
            {"payload": payload.model_dump()},
            context=RagResponseContext(
                shadow_evaluation_enabled_override=(
                    shadow_evaluation_enabled_override
                ),
                shadow_evaluation_model_override=(
                    shadow_evaluation_model_override
                ),
                shadow_evaluation_sink=shadow_evaluation_sink,
                correction_loop_enabled_override=(
                    correction_loop_enabled_override
                ),
            ),
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RagResponse(
        conversation_id=result["conversation_id"],
        message_id=result["message_id"],
        answer=result["answer"],
        action=result["answer_action"],
        sources=[ChunkSource(**source) for source in result["carousel_sources"]],
        retrieval=result["retrieval"],
    )


def run_rag(
    payload: RagRequest,
    *,
    shadow_evaluation_enabled_override: bool | None = None,
    shadow_evaluation_model_override: str | None = None,
    shadow_evaluation_sink: dict[str, Any] | None = None,
    correction_loop_enabled_override: bool | None = None,
) -> RagResponse:
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
        response = execute_rag(
            payload,
            shadow_evaluation_enabled_override=(
                shadow_evaluation_enabled_override
            ),
            shadow_evaluation_model_override=(
                shadow_evaluation_model_override
            ),
            shadow_evaluation_sink=shadow_evaluation_sink,
            correction_loop_enabled_override=(
                correction_loop_enabled_override
            ),
        )
        request_span.set_session_id(response.conversation_id)
        request_span.set_attribute("rag.action", response.action)
        request_span.set_attribute("rag.source_count", len(response.sources))
        request_span.set_output(response.model_dump())
        return response


def validate_step_models(payload: RagRequest) -> None:
    """Validate the single OpenAI model used by every RAG inference stage."""
    defaults = {
        "reformulationModel": DEFAULT_REFORMULATION_MODEL,
        "plannerModel": DEFAULT_PLANNER_MODEL,
        "answerModel": DEFAULT_GENERATION_MODEL,
    }
    for field_name, default in defaults.items():
        model = normalize_model_name(getattr(payload, field_name), default)
        if model != default:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Le RAG utilise uniquement gpt-5.6-luna."
                ),
            )
        try:
            provider_for_model(model)
        except LLMProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        setattr(payload, field_name, model)


@router.post("/rag", response_model=RagResponse)
def rag(payload: RagRequest) -> RagResponse:
    validate_step_models(payload)
    return run_rag(payload)
