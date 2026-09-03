from __future__ import annotations

from typing import Any

from interface.backend.analytics_sql import run_deterministic_analytics
from interface.backend.config import (
    DEFAULT_ANALYTICS_SQL_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
    DEFAULT_RERANK_MODEL,
)
from interface.backend.conversation_memory import (
    TOPIC_MATCH_MAX_COSINE_DISTANCE,
    assign_topic_id,
    load_conversation_memory,
    load_reformulation_memory,
)
from interface.backend.planner import (
    apply_deterministic_sql_policy,
    build_execution_plan,
    build_social_answer,
    extract_video_title_hint,
    has_sql_filters,
    reformulate_question,
    resolve_company_filters,
    resolve_person_filters,
    resolve_title_hints,
    run_planner,
    sanitize_video_title_hints,
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
    from interface.backend.orchestration_graph import invoke

    return invoke(payload)
