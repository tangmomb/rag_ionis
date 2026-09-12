from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from interface.backend import orchestration as services
from interface.backend.config import (
    DEFAULT_ANALYTICS_SQL_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
    DEFAULT_RERANK_MODEL,
)
from interface.backend.planner import PERSON_NAME_PART_SIMILARITY_THRESHOLD
from interface.backend.schemas import ExecutionPlan, ExecutionRoute, PlannerPlan, RagRequest


class RagOrchestrationState(TypedDict, total=False):
    payload: dict[str, Any]
    reformulation_model: str
    planner_model: str
    analytics_sql_model: str
    contextual_question: str
    reformulation_trace: dict[str, Any]
    topic_assignment: dict[str, Any]
    planner_plan: dict[str, Any]
    planner_prompt: str
    planner_raw: str
    pydantic_verification: bool
    database_persons: list[str]
    person_resolution: dict[str, Any]
    database_company: list[str]
    company_resolution: dict[str, Any]
    resolved_title_hints: list[str]
    title_resolution: dict[str, Any]
    execution_plan: dict[str, Any]
    base_retrieval: dict[str, Any]
    answer: str
    sources: list[dict[str, Any]]
    retrieval: dict[str, Any]


@dataclass(frozen=True)
class RagOrchestrationContext:
    client: Any = None


def _runtime_client(runtime: Runtime[RagOrchestrationContext]) -> Any:
    context = runtime.context
    return context.client if context is not None and context.client is not None else services.get_llm_client()


def _payload(state: RagOrchestrationState) -> RagRequest:
    return RagRequest.model_validate(state["payload"])


def _planner_plan(state: RagOrchestrationState) -> PlannerPlan:
    return PlannerPlan.model_validate(state["planner_plan"])


def _execution_plan(state: RagOrchestrationState) -> ExecutionPlan:
    return ExecutionPlan.model_validate(state["execution_plan"])


def _execution_plan_trace(
    execution_plan: ExecutionPlan,
) -> dict[str, Any]:
    """Serialize the executable plan, whose persons are already canonical."""
    return execution_plan.model_dump()


def _set_execution_route(
    execution_plan: ExecutionPlan,
    route: ExecutionRoute,
) -> None:
    """Make the executable plan and its effective retrieval source agree."""
    execution_plan.route = route
    if route == "vector_search":
        return

    execution_plan.top_k = None
    execution_plan.final_k = None


def _resolved_plan_persons(state: RagOrchestrationState) -> list[str]:
    """Merge confident speaker and transcript matches into one SQL input list."""
    persons: list[str] = []
    for person in [
        *state.get("database_persons", []),
        *[
            str(item["person"])
            for item in state.get("person_resolution", {}).get(
                "suggestion_transcripts", []
            )
            if float(item.get("score", 0)) > PERSON_NAME_PART_SIMILARITY_THRESHOLD
        ],
    ]:
        if person not in persons:
            persons.append(person)
    return persons


def _resolved_plan_companies(state: RagOrchestrationState) -> list[str]:
    """Prefer canonical companies without dropping an explicit planner entity."""
    companies = [
        str(item["company"])
        for item in state.get("company_resolution", {}).get(
            "suggestion_companies", [])
        if float(item.get("score", 0)) >= 0.85
    ]
    return companies or _planner_plan(state).companies


def _resolved_plan_title_hints(state: RagOrchestrationState) -> list[str]:
    """Use canonical titles when available, otherwise retain explicit title hints."""
    titles = state.get("resolved_title_hints")
    if isinstance(titles, list) and titles:
        return titles
    legacy_title = state.get("resolved_title_hint")
    if isinstance(legacy_title, str) and legacy_title:
        return [legacy_title]
    return _planner_plan(state).title_hints


def _resolved_transcript_persons(state: RagOrchestrationState) -> list[str]:
    """Keep transcript matches distinct from speaker-table matches for SQL."""
    return [
        str(item["person"])
        for item in state.get("person_resolution", {}).get(
            "suggestion_transcripts", []
        )
        if float(item.get("score", 0)) > PERSON_NAME_PART_SIMILARITY_THRESHOLD
    ]


def _resolved_plan_title_hint(state: RagOrchestrationState) -> str | None:
    """Compatibility helper for callers that only support one canonical title."""
    titles = _resolved_plan_title_hints(state)
    return titles[0] if titles else None


def initialize(state: RagOrchestrationState) -> dict[str, Any]:
    payload = _payload(state)
    return {
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


def reformulate(
    state: RagOrchestrationState,
    runtime: Runtime[RagOrchestrationContext],
) -> dict[str, Any]:
    payload = _payload(state)
    client = _runtime_client(runtime)
    reformulation_model = state["reformulation_model"]
    with services.trace_operation(
        "reformulation",
        kind="CHAIN",
        input_value={
            "question": payload.question,
            "conversation_id": payload.conversationId,
            "model": reformulation_model,
        },
    ) as reformulation_span:
        memory_result = services.load_conversation_memory(payload.conversationId)
        conversation_memory = memory_result["memory"]
        contextual_question, reformulation_trace = services.reformulate_question(
            payload.question,
            payload.conversationId,
            client,
            reformulation_model,
            payload.reformulationPrompt,
            history_override=[],
            memory_context={"conversation_memory": conversation_memory},
        )
        reformulation_trace["strategy"] = "conversation_json_single_rewrite"
        reformulation_trace["memory_available"] = memory_result["available"]
        topic_assignment = {
            "follow_up": bool(reformulation_trace.get("follow_up", False)),
            "topic": reformulation_trace.get("topic", ""),
        }
        reformulation_span.set_output(reformulation_trace)
    return {
        "contextual_question": contextual_question,
        "reformulation_trace": reformulation_trace,
        "topic_assignment": topic_assignment,
    }


def plan(
    state: RagOrchestrationState,
    runtime: Runtime[RagOrchestrationContext],
) -> dict[str, Any]:
    payload = _payload(state)
    contextual_question = state["contextual_question"]
    planner_model = state["planner_model"]
    with services.trace_operation(
        "planner",
        kind="AGENT",
        input_value={"question": contextual_question, "model": planner_model},
    ) as planner_span:
        planner_plan, planner_prompt, planner_raw, pydantic_verification = (
            services.run_planner(
                contextual_question,
                _runtime_client(runtime),
                planner_model,
                payload.plannerPrompt,
            )
        )
        explicit_title_hint = services.extract_video_title_hint(contextual_question)
        planner_plan.title_hints = services.sanitize_video_title_hints(
            contextual_question,
            [*([explicit_title_hint] if explicit_title_hint else []), *planner_plan.title_hints],
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
                "planner_output_rejected": planner_plan.output_rejection_reason,
                "policy_correction": policy_correction,
                "plan": planner_plan.model_dump(),
            }
        )
    return {
        "planner_plan": planner_plan.model_dump(),
        "planner_prompt": planner_prompt,
        "planner_raw": planner_raw,
        "pydantic_verification": pydantic_verification,
    }


def resolve_entities(state: RagOrchestrationState) -> dict[str, Any]:
    planner_plan = _planner_plan(state)
    planned_persons = [
        str(value).strip() for value in planner_plan.persons if str(value).strip()
    ]
    database_persons: list[str] = []
    person_resolution: dict[str, Any] = {
        "applied": False,
        "ambiguous": False,
        "requested": [],
        "suggestion_speakers": [],
        "suggestion_transcripts": [],
        "reason": "no_planned_persons",
    }
    if planned_persons:
        with services.trace_operation(
            "person_resolution",
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
        "requested": [],
        "suggestion_companies": [],
    }
    if planned_companies:
        with services.trace_operation(
            "company_resolution",
            kind="CHAIN",
            input_value={"planned_companies": planned_companies},
        ) as company_span:
            database_company, company_resolution = services.resolve_company_filters(
                planned_companies
            )
            company_span.set_output(company_resolution)

    planned_title_hints = planner_plan.title_hints
    resolved_title_hints: list[str] = []
    title_resolution: dict[str, Any] = {
        "requested": None,
        "suggestion_titles": [],
    }
    if planned_title_hints:
        with services.trace_operation(
            "title_resolution",
            kind="CHAIN",
            input_value={"planned_title_hints": planned_title_hints},
        ) as title_span:
            resolved_title_hints, title_resolution = services.resolve_title_hints(
                planned_title_hints
            )
            title_span.set_output(title_resolution)
    return {
        "database_persons": database_persons,
        "person_resolution": person_resolution,
        "database_company": database_company,
        "company_resolution": company_resolution,
        "resolved_title_hints": resolved_title_hints,
        "title_resolution": title_resolution,
    }


def build_execution_plan(state: RagOrchestrationState) -> dict[str, Any]:
    payload = _payload(state)
    planner_plan = _planner_plan(state)
    execution_plan = services.build_execution_plan(payload, planner_plan)
    execution_plan.persons = _resolved_plan_persons(state)
    execution_plan.companies = _resolved_plan_companies(state)
    execution_plan.title_hints = _resolved_plan_title_hints(state)
    # The LLM supplies only the coarse direct/search intent. Once entities have
    # been resolved, expose the exact graph branch on the final execution plan.
    if state["person_resolution"].get("ambiguous"):
        _set_execution_route(execution_plan, "person_clarification")
    elif execution_plan.route == "direct":
        _set_execution_route(execution_plan, "direct")
    elif (
        execution_plan.route == "sql_search"
        or (
            execution_plan.sql_sub_intent is None
            and not execution_plan.description_requested
            and (
                execution_plan.persons
                or execution_plan.companies
                or execution_plan.title_hints
            )
        )
    ):
        _set_execution_route(execution_plan, "sql_search")
    else:
        _set_execution_route(execution_plan, "vector_search")
    execution_plan_trace = _execution_plan_trace(execution_plan)
    if not (
        execution_plan.route == "sql_search"
    ):
        with services.trace_operation(
            "execution_plan",
            kind="CHAIN",
            input_value={"planner_plan": planner_plan.model_dump()},
        ) as execution_plan_span:
            execution_plan_span.set_output(execution_plan_trace)

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
        "execution_plan": execution_plan_trace,
        "validated_query": execution_plan.model_dump(),
        "person_resolution": state["person_resolution"],
        "company_resolution": state["company_resolution"],
        "title_resolution": state["title_resolution"],
        "resolved_companies": state["database_company"],
        "conversation_topic": state["topic_assignment"],
    }
    return {
        "execution_plan": execution_plan.model_dump(),
        "base_retrieval": base_retrieval,
    }


def select_route(
    state: RagOrchestrationState,
) -> ExecutionRoute:
    return _execution_plan(state).route


def person_clarification(state: RagOrchestrationState) -> dict[str, Any]:
    payload = _payload(state)
    execution_plan = _execution_plan(state)
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


def _sql_parent_output(trace: dict[str, Any]) -> dict[str, Any]:
    """Keep detailed SQL result rows, including transcripts, on child spans only."""
    output = {
        key: value
        for key, value in trace.items()
        if key not in {"query_results", "persons_table"}
    }
    persons_table = trace.get("persons_table")
    if isinstance(persons_table, dict):
        output["persons_table"] = {
            key: value
            for key, value in persons_table.items()
            if key != "query_results"
        }
    return output


def run_structured_lookup(
    state: RagOrchestrationState,
    client: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    execution_plan = _execution_plan(state)
    execution_plan_trace = _execution_plan_trace(execution_plan)
    sql_sub_intent = execution_plan.sql_sub_intent
    lookup_intent = (
        "analytics"
        if sql_sub_intent == "analytics"
        else "description"
        if execution_plan.description_requested
        else "specific_persons"
    )
    with services.trace_operation(
        "execution_plan",
        kind="CHAIN",
        input_value={"execution_plan": execution_plan_trace},
    ) as execution_plan_span:
        with services.trace_operation(
            "sql",
            kind="CHAIN",
            input_value={
                "execution_plan": execution_plan_trace,
                "sql_sub_intent": sql_sub_intent,
                "lookup_intent": lookup_intent,
            },
        ) as sql_span:
            if lookup_intent == "analytics":
                sources, direct_trace = services.run_deterministic_analytics(
                    execution_plan,
                    database_persons=state["database_persons"],
                    database_companies=state["database_company"],
                )
            elif lookup_intent == "description":
                sources, direct_trace = services.lookup_video_document(
                    execution_plan,
                    lookup_intent,
                    database_persons=execution_plan.persons,
                    database_company=state["database_company"],
                    transcript_persons=execution_plan.persons,
                )
            else:
                entity_queries: list[tuple[str, str, ExecutionPlan, dict[str, Any]]] = []
                # A speaker-table hit and a transcript hit are separate sources,
                # even when they resolve to the same canonical display name.
                for person in state["database_persons"]:
                    entity_queries.append(
                        (
                            "speaker",
                            person,
                            execution_plan.model_copy(
                                update={"persons": [], "companies": [], "title_hints": []}
                            ),
                            {"database_persons": [person], "database_company": [], "transcript_persons": []},
                        )
                    )
                for person in _resolved_transcript_persons(state):
                    entity_queries.append(
                        (
                            "transcript",
                            person,
                            execution_plan.model_copy(
                                update={"persons": [person], "companies": [], "title_hints": []}
                            ),
                            {"database_persons": [], "database_company": [], "transcript_persons": [person]},
                        )
                    )
                for company in execution_plan.companies:
                    entity_queries.append(
                        (
                            "company",
                            company,
                            execution_plan.model_copy(
                                update={"persons": [], "companies": [company], "title_hints": []}
                            ),
                            {"database_persons": [], "database_company": [company], "transcript_persons": []},
                        )
                    )
                for title in execution_plan.title_hints:
                    entity_queries.append(
                        (
                            "title",
                            title,
                            execution_plan.model_copy(
                                update={"persons": [], "companies": [], "title_hints": [title]}
                            ),
                            {"database_persons": [], "database_company": [], "transcript_persons": []},
                        )
                    )

                sources_by_video: dict[int, dict[str, Any]] = {}
                entity_traces: list[dict[str, Any]] = []
                input_count = 0
                lookup_span_names = {
                    "speaker": "person_in_speaker_lookup",
                    "transcript": "person_in_transcript_lookup",
                    "title": "title_lookup",
                    "company": "company_lookup",
                }
                for entity_type, entity, entity_plan, filters in entity_queries:
                    with services.trace_operation(
                        lookup_span_names[entity_type],
                        kind="RETRIEVER",
                        input_value={"entity_type": entity_type, "entity": entity},
                    ) as entity_span:
                        entity_sources, entity_trace = services.lookup_video_document(
                            entity_plan,
                            "specific_persons",
                            **filters,
                        )
                        entity_span.set_output(
                            {
                                "result_count": len(entity_sources),
                                "lookup_strategy": entity_trace.get("lookup_strategy"),
                            }
                        )
                    input_count += len(entity_sources)
                    for source in entity_sources:
                        sources_by_video.setdefault(int(source["chunk_id"]), source)
                    entity_traces.append(
                        {
                            "entity_type": entity_type,
                            "entity": entity,
                            "result_count": len(entity_sources),
                            "lookup_strategy": entity_trace.get("lookup_strategy"),
                        }
                    )
                sources = list(sources_by_video.values())
                direct_trace = {
                    "mode": "specific_persons",
                    "entity_queries": entity_traces,
                    "result_count": len(sources),
                    "deduplication": {
                        "input_count": input_count,
                        "duplicate_count": input_count - len(sources),
                        "output_count": len(sources),
                        "key": "video_id",
                    },
                }
            sql_span.set_output(_sql_parent_output(direct_trace))
            services.trace_formatted_sql("sql", direct_trace)
        execution_plan_span.set_output(
            {
                "execution_plan": execution_plan_trace,
                "source_count": len(sources),
            }
        )
    return sources, direct_trace, sql_sub_intent


def sql_search(
    state: RagOrchestrationState,
    runtime: Runtime[RagOrchestrationContext],
) -> dict[str, Any]:
    payload = _payload(state)
    execution_plan = _execution_plan(state)
    sources, direct_trace, sql_sub_intent = run_structured_lookup(
        state, _runtime_client(runtime)
    )
    retrieval = {
        **state["base_retrieval"],
        "answer_model": services.normalize_model_name(
            payload.answerModel, DEFAULT_GENERATION_MODEL
        ),
        "embedding_model": None,
        "rerank_model": None,
        "retrieval_mode": "search+sql",
        "sql_prefilters": services.has_sql_filters(execution_plan),
        "bm25_top_k": 0,
        "vector_top_k": 0,
        "rrf_top_n": 0,
        "final_k": len(sources),
        "used_rerank": False,
        "general_question_only": not services.has_sql_filters(
            execution_plan
        ),
        "sql_query": direct_trace.get("sql")
        or direct_trace.get("stats", {}).get("sql"),
        "prefilter": {},
        "sql_prefilters_trace": {},
        "bm25": {},
        "vector": {},
        "rrf": {},
        "rerank": {},
        "direct_lookup": direct_trace,
        "sql_sub_intent": sql_sub_intent,
        "description_requested": execution_plan.description_requested,
    }
    return {"answer": "", "sources": sources, "retrieval": retrieval}


def vector_search(state: RagOrchestrationState) -> dict[str, Any]:
    sources, retrieval = services.retrieve_chunks(
        _payload(state), _execution_plan(state)
    )
    retrieval.update(state["base_retrieval"])
    return {"answer": "", "sources": sources, "retrieval": retrieval}


def build_graph():
    graph = StateGraph(
        RagOrchestrationState,
        context_schema=RagOrchestrationContext,
    )
    graph.add_node("initialize", initialize)
    graph.add_node("reformulate", reformulate)
    graph.add_node("plan", plan)
    graph.add_node("resolve_entities", resolve_entities)
    graph.add_node("build_execution_plan", build_execution_plan)
    graph.add_node("person_clarification", person_clarification)
    graph.add_node("direct", direct)
    graph.add_node("sql_search", sql_search)
    graph.add_node("vector_search", vector_search)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "reformulate")
    graph.add_edge("reformulate", "plan")
    graph.add_edge("plan", "resolve_entities")
    graph.add_edge("resolve_entities", "build_execution_plan")
    graph.add_conditional_edges("build_execution_plan", select_route)
    for route in (
        "person_clarification",
        "direct",
        "sql_search",
        "vector_search",
    ):
        graph.add_edge(route, END)
    return graph.compile()


RAG_ORCHESTRATION_GRAPH = build_graph()


def invoke(payload: RagRequest) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    result = RAG_ORCHESTRATION_GRAPH.invoke(
        {"payload": payload.model_dump()},
        context=RagOrchestrationContext(client=services.get_llm_client()),
    )
    return result["answer"], result["sources"], result["retrieval"]
