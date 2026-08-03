from __future__ import annotations

from typing import Any

from interface.backend.analytics_sql import run_analytics_text_to_sql
from interface.backend.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
    DEFAULT_RERANK_MODEL,
)
from interface.backend.database import fetch_conversation_memory
from interface.backend.planner import (
    apply_deterministic_sql_policy,
    build_execution_plan,
    build_social_answer,
    extract_video_title_hint,
    has_structured_sql_filters,
    reformulate_question,
    resolve_company_filters,
    resolve_person_filters,
    run_planner,
    sanitize_video_title_hint,
)
from interface.backend.retrieval import lookup_video_document, retrieve_chunks, trace_formatted_sql
from interface.backend.schemas import RagRequest
from interface.backend.telemetry import trace_operation
from interface.backend.utilities import get_llm_client, normalize_model_name


def build_direct_retrieval(base_retrieval: dict[str, Any], answer: str, route_name: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    retrieval = {
        **base_retrieval,
        "direct_answer": answer,
        "answer_model": None,
        "embedding_model": None,
        "rerank_model": None,
        "retrieval_mode": route_name,
        "sql_main_source": False,
        "sql_prefilters": False,
        "bm25_top_k": 0,
        "vector_top_k": 0,
        "rrf_top_n": 0,
        "final_k": 0,
        "used_rerank": False,
        "general_question_only": True,
        "sql_query": None,
        "prefilter": {},
        "sql_prefilters_trace": {},
        "bm25": {},
        "vector": {},
        "rrf": {},
        "rerank": {},
        "direct_lookup": {},
    }
    return answer, [], retrieval


def orchestrate_request(payload: RagRequest) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    client = get_llm_client()
    reformulation_model = normalize_model_name(
        payload.reformulationModel,
        DEFAULT_REFORMULATION_MODEL,
    )
    planner_model = normalize_model_name(
        payload.plannerModel,
        DEFAULT_PLANNER_MODEL,
    )
    with trace_operation(
        "rag.reformulation",
        kind="CHAIN",
        input_value={
            "question": payload.question,
            "conversation_id": payload.conversationId,
            "model": reformulation_model,
        },
    ) as reformulation_span:
        contextual_question, reformulation_trace = reformulate_question(
            payload.question,
            payload.conversationId,
            client,
            reformulation_model,
            payload.reformulationPrompt,
        )
        reformulation_span.set_output(reformulation_trace)

    with trace_operation(
        "rag.planner",
        kind="AGENT",
        input_value={"question": contextual_question, "model": planner_model},
    ) as planner_span:
        planner_plan, planner_prompt, planner_raw, pydantic_verification = run_planner(
            contextual_question,
            client,
            planner_model,
            payload.plannerPrompt,
        )
        planner_plan.title_hint = sanitize_video_title_hint(
            contextual_question,
            extract_video_title_hint(contextual_question) or planner_plan.title_hint,
        )
        policy_correction = apply_deterministic_sql_policy(
            contextual_question,
            planner_plan,
        )
        planner_span.set_output(
            {
                "prompt": planner_prompt,
                "response_raw": planner_raw,
                "validated": pydantic_verification,
                "policy_correction": policy_correction,
                "plan": planner_plan.model_dump(),
            }
        )

    planned_persons = [
        str(value).strip()
        for value in planner_plan.persons
        if str(value).strip()
    ]
    database_persons: list[str] = []
    person_resolution: dict[str, Any] = {
        "applied": False,
        "ambiguous": False,
        "requested": [],
        "suggestions": [],
        "suggestion_scores": [],
        "auto_resolved": False,
        "reason": "no_planned_persons",
    }
    if planned_persons:
        with trace_operation(
            "rag.person_resolution",
            kind="CHAIN",
            input_value={"planned_persons": planned_persons},
        ) as person_span:
            database_persons, person_resolution = resolve_person_filters(
                planned_persons,
            )
            person_span.set_output(person_resolution)

    planned_companies = [
        str(value).strip()
        for value in planner_plan.companies
        if str(value).strip()
    ]
    database_company: list[str] = []
    company_resolution: dict[str, Any] = {
        "applied": False,
        "requested": [],
        "resolved": [],
        "matches": [],
        "unresolved": [],
        "reason": "no_planned_company",
    }
    if planned_companies:
        with trace_operation(
            "rag.company_resolution",
            kind="CHAIN",
            input_value={"planned_company": planned_companies},
        ) as company_span:
            database_company, company_resolution = resolve_company_filters(
                planned_companies,
            )
            company_span.set_output(company_resolution)
    execution_plan = build_execution_plan(
        payload,
        planner_plan,
    )
    # Les routes sans SQL structuré n'ont pas de sous-opération longue à
    # envelopper ; on conserve leur plan dans un span dédié.
    if not (
        execution_plan.sql_main_source
        and execution_plan.route in {"rag", "multi_source"}
    ):
        with trace_operation(
            "rag.execution_plan",
            kind="CHAIN",
            input_value={
                "planner_plan": planner_plan.model_dump(),
            },
        ) as execution_plan_span:
            execution_plan_span.set_output(execution_plan.model_dump())

    base_retrieval = {
        "route": execution_plan.route,
        "sql_sub_intent": execution_plan.sql_sub_intent,
        "planner_prompt": planner_prompt,
        "planner_response_raw": planner_raw,
        "pydantic_verification": pydantic_verification,
        "question_reformulation": reformulation_trace,
        "contextual_question": contextual_question,
        "reformulation_model": reformulation_model,
        "planner_model": planner_model,
        "planner_plan": planner_plan.model_dump(),
        "execution_plan": execution_plan.model_dump(),
        "validated_query": execution_plan.model_dump(),
        "person_resolution": person_resolution,
        "company_resolution": company_resolution,
    }

    if person_resolution.get("ambiguous"):
        clarification_retrieval = {
            **base_retrieval,
            "answer_model": normalize_model_name(payload.answerModel, DEFAULT_GENERATION_MODEL),
            "embedding_model": normalize_model_name(payload.embeddingModel, DEFAULT_EMBEDDING_MODEL),
            "rerank_model": normalize_model_name(payload.rerankModel or "", DEFAULT_RERANK_MODEL),
            "retrieval_mode": "person_clarification",
            "sql_main_source": execution_plan.sql_main_source,
            "sql_prefilters": False,
            "bm25_top_k": 0,
            "vector_top_k": 0,
            "rrf_top_n": 0,
            "final_k": 0,
            "used_rerank": False,
            "general_question_only": False,
            "sql_query": None,
            "prefilter": {},
            "sql_prefilters_trace": {},
            "bm25": {},
            "vector": {},
            "rrf": {},
            "rerank": {},
            "direct_lookup": {},
            "memory": {},
        }
        return "", [], clarification_retrieval

    if execution_plan.route == "direct":
        return build_direct_retrieval(base_retrieval, build_social_answer(contextual_question), "direct")

    if execution_plan.route == "memory":
        with trace_operation(
            "rag.memory",
            kind="CHAIN",
            input_value={"conversation_id": payload.conversationId},
        ) as memory_span:
            memory_items, memory_trace = fetch_conversation_memory(payload.conversationId)
            memory_span.set_output({**memory_trace, "results": memory_items})
        answer_model = normalize_model_name(payload.answerModel, DEFAULT_GENERATION_MODEL)
        retrieval = {
            **base_retrieval,
            "answer_model": answer_model,
            "embedding_model": None,
            "rerank_model": None,
            "retrieval_mode": "memory",
            "sql_main_source": False,
            "sql_prefilters": False,
            "bm25_top_k": 0,
            "vector_top_k": 0,
            "rrf_top_n": 0,
            "final_k": 0,
            "used_rerank": False,
            "general_question_only": True,
            "sql_query": None,
            "prefilter": {},
            "sql_prefilters_trace": {},
            "bm25": {},
            "vector": {},
            "rrf": {},
            "rerank": {},
            "memory": memory_trace,
            "memory_items": memory_items,
        }
        return "", [], retrieval

    if execution_plan.route == "rag" and execution_plan.sql_main_source:
        sql_sub_intent = execution_plan.sql_sub_intent or "specific_persons"
        # Ce span couvre l'application réelle du plan : Phoenix l'affiche
        # ainsi avant le SQL qu'il pilote, au lieu d'un span de construction
        # instantané trié arbitrairement parmi ses frères.
        with trace_operation(
            "rag.execution_plan",
            kind="CHAIN",
            input_value={
                "execution_plan": execution_plan.model_dump(),
            },
        ) as execution_plan_span:
            with trace_operation(
                "rag.structured_sql",
                kind="CHAIN",
                input_value={
                    "execution_plan": execution_plan.model_dump(),
                    "sql_sub_intent": sql_sub_intent,
                },
            ) as sql_span:
                if sql_sub_intent == "analytics":
                    sources, direct_trace = run_analytics_text_to_sql(
                        execution_plan,
                        client,
                        planner_model,
                        database_persons=database_persons,
                        database_companies=database_company,
                    )
                else:
                    sources, direct_trace = lookup_video_document(
                        execution_plan,
                        sql_sub_intent,
                        database_persons=database_persons,
                        database_company=database_company,
                        transcript_persons=person_resolution.get("matched_in_transcripts", []),
                    )
                sql_span.set_output({**direct_trace, "results": sources})
                trace_formatted_sql("rag.structured_sql", direct_trace)
            execution_plan_span.set_output(
                {
                    "execution_plan": execution_plan.model_dump(),
                    "source_count": len(sources),
                }
            )
        fallback_trace: dict[str, Any] = {}
        retrieval_mode = "rag+structured_sql"
        if not sources:
            try:
                fallback_sources, fallback_trace = retrieve_chunks(payload, execution_plan)
            except Exception as exc:  # pragma: no cover
                fallback_trace = {"mode": "rag_fallback", "error": str(exc), "result_count": 0}
                fallback_sources = []
            if fallback_sources:
                sources = fallback_sources
                retrieval_mode = "rag+structured_sql_fallback"
        answer_model = normalize_model_name(payload.answerModel, DEFAULT_GENERATION_MODEL)
        retrieval = {
            **base_retrieval,
            "answer_model": answer_model,
            "embedding_model": fallback_trace.get("embedding_model"),
            "rerank_model": fallback_trace.get("rerank_model"),
            "retrieval_mode": retrieval_mode,
            "sql_main_source": True,
            "sql_prefilters": has_structured_sql_filters(execution_plan),
            "bm25_top_k": fallback_trace.get("bm25_top_k", 0),
            "vector_top_k": fallback_trace.get("vector_top_k", 0),
            "rrf_top_n": fallback_trace.get("rrf_top_n", 0),
            "final_k": len(sources),
            "used_rerank": fallback_trace.get("used_rerank", False),
            "general_question_only": not has_structured_sql_filters(execution_plan),
            "sql_query": direct_trace["sql"],
            "prefilter": fallback_trace.get("prefilter", {}),
            "sql_prefilters_trace": fallback_trace.get("sql_prefilters_trace", {}),
            "bm25": fallback_trace.get("bm25", {}),
            "vector": fallback_trace.get("vector", {}),
            "rrf": fallback_trace.get("rrf", {}),
            "rerank": fallback_trace.get("rerank", {}),
            "direct_lookup": {**direct_trace, "rag_fallback": fallback_trace},
            "memory": {},
            "sql_sub_intent": sql_sub_intent,
        }
        return "", sources, retrieval

    if execution_plan.route == "multi_source":
        memory_items: list[dict[str, str]] = []
        memory_trace: dict[str, Any] = {}
        doc_sources: list[dict[str, Any]] = []
        doc_trace: dict[str, Any] = {}
        multi_source_actions: list[dict[str, Any]] = []

        if execution_plan.use_memory:
            with trace_operation(
                "rag.memory",
                kind="CHAIN",
                input_value={"conversation_id": payload.conversationId},
            ) as memory_span:
                memory_items, memory_trace = fetch_conversation_memory(payload.conversationId)
                memory_span.set_output({**memory_trace, "results": memory_items})
            multi_source_actions.append(
                {
                    "action": len(multi_source_actions) + 1,
                    "source": "memory",
                    "operation": "fetch_conversation_memory",
                    "status": "completed",
                    "request": {
                        "sql": memory_trace.get("sql"),
                        "params": memory_trace.get("params", []),
                    },
                    "response": memory_items,
                    "result_count": len(memory_items),
                }
            )

        if execution_plan.sql_main_source:
            sql_sub_intent = execution_plan.sql_sub_intent or "specific_persons"
            with trace_operation(
                "rag.execution_plan",
                kind="CHAIN",
                input_value={
                    "execution_plan": execution_plan.model_dump(),
                },
            ) as execution_plan_span:
                with trace_operation(
                    "rag.structured_sql",
                    kind="CHAIN",
                    input_value={
                        "execution_plan": execution_plan.model_dump(),
                        "sql_sub_intent": sql_sub_intent,
                    },
                ) as sql_span:
                    if sql_sub_intent == "analytics":
                        doc_sources, doc_trace = run_analytics_text_to_sql(
                            execution_plan,
                            client,
                            planner_model,
                            database_persons=database_persons,
                            database_companies=database_company,
                        )
                    else:
                        doc_sources, doc_trace = lookup_video_document(
                            execution_plan,
                            sql_sub_intent,
                            database_persons=database_persons,
                            database_company=database_company,
                            transcript_persons=person_resolution.get("matched_in_transcripts", []),
                        )
                    sql_span.set_output({**doc_trace, "results": doc_sources})
                    trace_formatted_sql("rag.structured_sql", doc_trace)
                execution_plan_span.set_output(
                    {
                        "execution_plan": execution_plan.model_dump(),
                        "source_count": len(doc_sources),
                    }
                )
            multi_source_actions.append(
                {
                    "action": len(multi_source_actions) + 1,
                    "source": "sql",
                    "operation": "lookup_video_document",
                    "sub_intent": sql_sub_intent,
                    "status": "completed",
                    "request": {
                        "sql": doc_trace.get("sql"),
                        "params": doc_trace.get("params", []),
                    },
                    "response": doc_sources,
                    "result_count": len(doc_sources),
                }
            )
        elif execution_plan.use_rag:
            doc_sources, doc_trace = retrieve_chunks(payload, execution_plan)
            multi_source_actions.append(
                {
                    "action": len(multi_source_actions) + 1,
                    "source": "rag",
                    "operation": "retrieve_chunks",
                    "status": "completed",
                    "request": doc_trace,
                    "response": doc_sources,
                    "result_count": len(doc_sources),
                }
            )

        answer_model = normalize_model_name(payload.answerModel, DEFAULT_GENERATION_MODEL)
        retrieval = {
            **base_retrieval,
            "answer_model": answer_model,
            "embedding_model": doc_trace.get("embedding_model"),
            "rerank_model": doc_trace.get("rerank_model"),
            "retrieval_mode": execution_plan.route,
            "sql_main_source": execution_plan.sql_main_source,
            "bm25_top_k": doc_trace.get("bm25_top_k", 0),
            "vector_top_k": doc_trace.get("vector_top_k", 0),
            "rrf_top_n": doc_trace.get("rrf_top_n", 0),
            "final_k": doc_trace.get("final_k", len(doc_sources)),
            "used_rerank": doc_trace.get("used_rerank", False),
            "sql_prefilters": doc_trace.get("sql_prefilters", False),
            "general_question_only": doc_trace.get("general_question_only", True),
            "sql_query": doc_trace.get("sql_query"),
            "prefilter": doc_trace.get("prefilter", {}) if doc_trace.get("retrieval_mode") == "prefilter+bm25+vector+rrf" else {},
            "sql_prefilters_trace": doc_trace.get("sql_prefilters_trace", {}) if doc_trace.get("retrieval_mode") == "prefilter+bm25+vector+rrf" else {},
            "bm25": doc_trace.get("bm25", {}),
            "vector": doc_trace.get("vector", {}),
            "rrf": doc_trace.get("rrf", {}),
            "rerank": doc_trace.get("rerank", {}),
            "direct_lookup": doc_trace.get("direct_lookup", {}),
            "memory": memory_trace,
            "memory_items": memory_items,
            "multi_source_actions": multi_source_actions,
        }
        return "", doc_sources, retrieval

    sources, retrieval = retrieve_chunks(payload, execution_plan)
    retrieval["route"] = "rag"
    retrieval.update(base_retrieval)
    return "", sources, retrieval
