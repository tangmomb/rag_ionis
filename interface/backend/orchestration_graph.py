from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from interface.backend import orchestration as services
from interface.backend.config import (
    DEFAULT_ANALYTICS_SQL_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
    DEFAULT_RERANK_MODEL,
)
from interface.backend.conversation_memory import TOPIC_MATCH_MAX_COSINE_DISTANCE
from interface.backend.schemas import ExecutionPlan, PlannerPlan, RagRequest


class RagOrchestrationState(TypedDict, total=False):
    payload: RagRequest
    client: Any
    reformulation_model: str
    planner_model: str
    analytics_sql_model: str
    contextual_question: str
    reformulation_trace: dict[str, Any]
    topic_assignment: dict[str, Any]
    planner_plan: PlannerPlan
    planner_prompt: str
    planner_raw: str
    pydantic_verification: bool
    database_persons: list[str]
    person_resolution: dict[str, Any]
    database_company: list[str]
    company_resolution: dict[str, Any]
    execution_plan: ExecutionPlan
    base_retrieval: dict[str, Any]
    answer: str
    sources: list[dict[str, Any]]
    retrieval: dict[str, Any]


def initialize(state: RagOrchestrationState) -> dict[str, Any]:
    payload = state["payload"]
    return {
        "client": services.get_llm_client(),
        "reformulation_model": services.normalize_model_name(
            payload.reformulationModel,
            DEFAULT_REFORMULATION_MODEL,
        ),
        "planner_model": services.normalize_model_name(
            payload.plannerModel,
            DEFAULT_PLANNER_MODEL,
        ),
        "analytics_sql_model": DEFAULT_ANALYTICS_SQL_MODEL,
    }


def reformulate(state: RagOrchestrationState) -> dict[str, Any]:
    payload = state["payload"]
    client = state["client"]
    reformulation_model = state["reformulation_model"]
    with services.trace_operation(
        "rag.reformulation",
        kind="CHAIN",
        input_value={
            "question": payload.question,
            "conversation_id": payload.conversationId,
            "model": reformulation_model,
        },
    ) as reformulation_span:
        memory = services.load_reformulation_memory(
            payload.conversationId,
            payload.question,
            include_episodes=False,
        )
        if memory.get("available"):
            light_question, light_trace = services.reformulate_question(
                payload.question,
                payload.conversationId,
                client,
                reformulation_model,
                payload.reformulationPrompt,
                history_override=memory.get("immediate_history", []),
                phase="light",
            )
            follows_active_topic = bool(light_trace.get("follow_up"))
            with services.trace_operation(
                "rag.conversation_memory.topic_assignment",
                kind="TOOL",
                input_value={
                    "conversation_id": payload.conversationId,
                    "follow_up": follows_active_topic,
                },
            ) as topic_span:
                topic_assignment = services.assign_topic_id(
                    payload.conversationId,
                    follows_active_topic,
                )
                topic_span.set_output(topic_assignment)
            if follows_active_topic:
                long_memory = {
                    "available": True,
                    "strategy": "final_rewrite_skipped_for_follow_up",
                    "episodes": [],
                }
                contextual_question = light_question
                final_trace = {
                    "phase": "final",
                    "skipped": True,
                    "reason": "follow_up_uses_light_rewrite",
                }
            else:
                with services.trace_operation(
                    "rag.conversation_memory.topic_match",
                    kind="RETRIEVER",
                    input_value={
                        "conversation_id": payload.conversationId,
                        "question": light_question,
                        "excluded_topic_id": topic_assignment.get("topic_id"),
                        "match_max_cosine_distance": TOPIC_MATCH_MAX_COSINE_DISTANCE,
                    },
                ) as topic_match_span:
                    long_memory = services.load_reformulation_memory(
                        payload.conversationId,
                        light_question,
                        include_episodes=False,
                        embed_question=True,
                        exclude_topic_id=topic_assignment.get("topic_id"),
                    )
                    topic_match_span.set_output(
                        {
                            "available": long_memory.get("available"),
                            "related_topics": long_memory.get("related_topics", []),
                            "retrieval": long_memory.get("retrieval", {}),
                        }
                    )
                contextual_question, final_trace = services.reformulate_question(
                    light_question,
                    payload.conversationId,
                    client,
                    reformulation_model,
                    payload.reformulationPrompt,
                    history_override=[],
                    memory_context={
                        "related_topics": long_memory.get("related_topics", []),
                    },
                    phase="final",
                )
            reformulation_trace = {
                "strategy": (
                    "light_rewrite_only"
                    if follows_active_topic
                    else "light_rewrite+topic_match+final_rewrite"
                ),
                "follow_up": follows_active_topic,
                "topic_assignment": topic_assignment,
                "light": light_trace,
                "final": final_trace,
                "memory": {
                    "light": {
                        key: value
                        for key, value in memory.items()
                        if key != "immediate_history"
                    },
                    "long": {
                        key: value
                        for key, value in long_memory.items()
                        if key != "immediate_history"
                    },
                },
            }
        else:
            contextual_question, reformulation_trace = services.reformulate_question(
                payload.question,
                payload.conversationId,
                client,
                reformulation_model,
                payload.reformulationPrompt,
            )
            topic_assignment = {"reason": "memory_unavailable"}
        reformulation_span.set_output(reformulation_trace)
    return {
        "contextual_question": contextual_question,
        "reformulation_trace": reformulation_trace,
        "topic_assignment": topic_assignment,
    }


def plan(state: RagOrchestrationState) -> dict[str, Any]:
    payload = state["payload"]
    contextual_question = state["contextual_question"]
    planner_model = state["planner_model"]
    with services.trace_operation(
        "rag.planner",
        kind="AGENT",
        input_value={"question": contextual_question, "model": planner_model},
    ) as planner_span:
        planner_plan, planner_prompt, planner_raw, pydantic_verification = (
            services.run_planner(
                contextual_question,
                state["client"],
                planner_model,
                payload.plannerPrompt,
            )
        )
        planner_plan.title_hint = services.sanitize_video_title_hint(
            contextual_question,
            services.extract_video_title_hint(contextual_question)
            or planner_plan.title_hint,
        )
        policy_correction = services.apply_deterministic_sql_policy(
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
    return {
        "planner_plan": planner_plan,
        "planner_prompt": planner_prompt,
        "planner_raw": planner_raw,
        "pydantic_verification": pydantic_verification,
    }


def resolve_entities(state: RagOrchestrationState) -> dict[str, Any]:
    planner_plan = state["planner_plan"]
    planned_persons = [
        str(value).strip() for value in planner_plan.persons if str(value).strip()
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
        with services.trace_operation(
            "rag.person_resolution",
            kind="CHAIN",
            input_value={"planned_persons": planned_persons},
        ) as person_span:
            database_persons, person_resolution = services.resolve_person_filters(
                planned_persons
            )
            person_span.set_output(person_resolution)

    planned_companies = [
        str(value).strip() for value in planner_plan.companies if str(value).strip()
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
        with services.trace_operation(
            "rag.company_resolution",
            kind="CHAIN",
            input_value={"planned_company": planned_companies},
        ) as company_span:
            database_company, company_resolution = services.resolve_company_filters(
                planned_companies
            )
            company_span.set_output(company_resolution)
    return {
        "database_persons": database_persons,
        "person_resolution": person_resolution,
        "database_company": database_company,
        "company_resolution": company_resolution,
    }


def build_execution_plan(state: RagOrchestrationState) -> dict[str, Any]:
    payload = state["payload"]
    planner_plan = state["planner_plan"]
    execution_plan = services.build_execution_plan(payload, planner_plan)
    if not (
        execution_plan.sql_main_source
        and execution_plan.route in {"rag", "multi_source"}
    ):
        with services.trace_operation(
            "rag.execution_plan",
            kind="CHAIN",
            input_value={"planner_plan": planner_plan.model_dump()},
        ) as execution_plan_span:
            execution_plan_span.set_output(execution_plan.model_dump())

    base_retrieval = {
        "route": execution_plan.route,
        "sql_sub_intent": execution_plan.sql_sub_intent,
        "planner_prompt": state["planner_prompt"],
        "planner_response_raw": state["planner_raw"],
        "pydantic_verification": state["pydantic_verification"],
        "question_reformulation": state["reformulation_trace"],
        "contextual_question": state["contextual_question"],
        "reformulation_model": state["reformulation_model"],
        "planner_model": state["planner_model"],
        "analytics_sql_model": state["analytics_sql_model"],
        "planner_plan": planner_plan.model_dump(),
        "execution_plan": execution_plan.model_dump(),
        "validated_query": execution_plan.model_dump(),
        "person_resolution": state["person_resolution"],
        "company_resolution": state["company_resolution"],
        "resolved_persons": state["database_persons"],
        "resolved_companies": state["database_company"],
        "conversation_topic": state["topic_assignment"],
    }
    return {"execution_plan": execution_plan, "base_retrieval": base_retrieval}


def select_route(
    state: RagOrchestrationState,
) -> Literal[
    "person_clarification",
    "direct",
    "structured_sql",
    "multi_source",
    "rag",
]:
    if state["person_resolution"].get("ambiguous"):
        return "person_clarification"
    execution_plan = state["execution_plan"]
    if execution_plan.route == "direct":
        return "direct"
    if execution_plan.route == "rag" and execution_plan.sql_main_source:
        return "structured_sql"
    if execution_plan.route == "multi_source":
        return "multi_source"
    return "rag"


def person_clarification(state: RagOrchestrationState) -> dict[str, Any]:
    payload = state["payload"]
    execution_plan = state["execution_plan"]
    retrieval = {
        **state["base_retrieval"],
        "answer_model": services.normalize_model_name(
            payload.answerModel, DEFAULT_GENERATION_MODEL
        ),
        "embedding_model": services.normalize_model_name(
            payload.embeddingModel, DEFAULT_EMBEDDING_MODEL
        ),
        "rerank_model": services.normalize_model_name(
            payload.rerankModel or "", DEFAULT_RERANK_MODEL
        ),
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
    }
    return {"answer": "", "sources": [], "retrieval": retrieval}


def direct(state: RagOrchestrationState) -> dict[str, Any]:
    answer, sources, retrieval = services.build_direct_retrieval(
        state["base_retrieval"],
        services.build_social_answer(state["contextual_question"]),
        "direct",
    )
    return {"answer": answer, "sources": sources, "retrieval": retrieval}


def run_structured_lookup(
    state: RagOrchestrationState,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    execution_plan = state["execution_plan"]
    sql_sub_intent = execution_plan.sql_sub_intent or "specific_persons"
    with services.trace_operation(
        "rag.execution_plan",
        kind="CHAIN",
        input_value={"execution_plan": execution_plan.model_dump()},
    ) as execution_plan_span:
        with services.trace_operation(
            "rag.structured_sql",
            kind="CHAIN",
            input_value={
                "execution_plan": execution_plan.model_dump(),
                "sql_sub_intent": sql_sub_intent,
            },
        ) as sql_span:
            if sql_sub_intent == "analytics":
                sources, direct_trace = services.run_analytics_text_to_sql(
                    execution_plan,
                    state["client"],
                    state["analytics_sql_model"],
                    database_persons=state["database_persons"],
                    database_companies=state["database_company"],
                )
            else:
                sources, direct_trace = services.lookup_video_document(
                    execution_plan,
                    sql_sub_intent,
                    database_persons=state["database_persons"],
                    database_company=state["database_company"],
                    transcript_persons=state["person_resolution"].get(
                        "matched_in_transcripts", []
                    ),
                )
            sql_span.set_output({**direct_trace, "results": sources})
            services.trace_formatted_sql("rag.structured_sql", direct_trace)
        execution_plan_span.set_output(
            {
                "execution_plan": execution_plan.model_dump(),
                "source_count": len(sources),
            }
        )
    return sources, direct_trace, sql_sub_intent


def structured_sql(state: RagOrchestrationState) -> dict[str, Any]:
    payload = state["payload"]
    execution_plan = state["execution_plan"]
    sources, direct_trace, sql_sub_intent = run_structured_lookup(state)
    retrieval = {
        **state["base_retrieval"],
        "answer_model": services.normalize_model_name(
            payload.answerModel, DEFAULT_GENERATION_MODEL
        ),
        "embedding_model": None,
        "rerank_model": None,
        "retrieval_mode": "rag+structured_sql",
        "sql_main_source": True,
        "sql_prefilters": services.has_structured_sql_filters(execution_plan),
        "bm25_top_k": 0,
        "vector_top_k": 0,
        "rrf_top_n": 0,
        "final_k": len(sources),
        "used_rerank": False,
        "general_question_only": not services.has_structured_sql_filters(
            execution_plan
        ),
        "sql_query": direct_trace["sql"],
        "prefilter": {},
        "sql_prefilters_trace": {},
        "bm25": {},
        "vector": {},
        "rrf": {},
        "rerank": {},
        "direct_lookup": direct_trace,
        "sql_sub_intent": sql_sub_intent,
    }
    return {"answer": "", "sources": sources, "retrieval": retrieval}


def multi_source(state: RagOrchestrationState) -> dict[str, Any]:
    payload = state["payload"]
    execution_plan = state["execution_plan"]
    doc_sources: list[dict[str, Any]] = []
    doc_trace: dict[str, Any] = {}
    actions: list[dict[str, Any]] = []
    if execution_plan.sql_main_source:
        doc_sources, doc_trace, sql_sub_intent = run_structured_lookup(state)
        actions.append(
            {
                "action": 1,
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
        doc_sources, doc_trace = services.retrieve_chunks(payload, execution_plan)
        actions.append(
            {
                "action": 1,
                "source": "rag",
                "operation": "retrieve_chunks",
                "status": "completed",
                "request": doc_trace,
                "response": doc_sources,
                "result_count": len(doc_sources),
            }
        )

    retrieval = {
        **state["base_retrieval"],
        "answer_model": services.normalize_model_name(
            payload.answerModel, DEFAULT_GENERATION_MODEL
        ),
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
        "prefilter": (
            doc_trace.get("prefilter", {})
            if doc_trace.get("retrieval_mode") == "prefilter+bm25+vector+rrf"
            else {}
        ),
        "sql_prefilters_trace": (
            doc_trace.get("sql_prefilters_trace", {})
            if doc_trace.get("retrieval_mode") == "prefilter+bm25+vector+rrf"
            else {}
        ),
        "bm25": doc_trace.get("bm25", {}),
        "vector": doc_trace.get("vector", {}),
        "rrf": doc_trace.get("rrf", {}),
        "rerank": doc_trace.get("rerank", {}),
        "direct_lookup": doc_trace.get("direct_lookup", {}),
        "multi_source_actions": actions,
    }
    return {"answer": "", "sources": doc_sources, "retrieval": retrieval}


def rag(state: RagOrchestrationState) -> dict[str, Any]:
    sources, retrieval = services.retrieve_chunks(
        state["payload"], state["execution_plan"]
    )
    retrieval["route"] = "rag"
    retrieval.update(state["base_retrieval"])
    return {"answer": "", "sources": sources, "retrieval": retrieval}


def build_graph():
    graph = StateGraph(RagOrchestrationState)
    graph.add_node("initialize", initialize)
    graph.add_node("reformulate", reformulate)
    graph.add_node("plan", plan)
    graph.add_node("resolve_entities", resolve_entities)
    graph.add_node("build_execution_plan", build_execution_plan)
    graph.add_node("person_clarification", person_clarification)
    graph.add_node("direct", direct)
    graph.add_node("structured_sql", structured_sql)
    graph.add_node("multi_source", multi_source)
    graph.add_node("rag", rag)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "reformulate")
    graph.add_edge("reformulate", "plan")
    graph.add_edge("plan", "resolve_entities")
    graph.add_edge("resolve_entities", "build_execution_plan")
    graph.add_conditional_edges("build_execution_plan", select_route)
    for route in (
        "person_clarification",
        "direct",
        "structured_sql",
        "multi_source",
        "rag",
    ):
        graph.add_edge(route, END)
    return graph.compile()


RAG_ORCHESTRATION_GRAPH = build_graph()


def invoke(payload: RagRequest) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    result = RAG_ORCHESTRATION_GRAPH.invoke({"payload": payload})
    return result["answer"], result["sources"], result["retrieval"]
