from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from interface.backend.database import connect_analytics_database
from interface.backend.llm_providers import LLMClientProtocol
from interface.backend.schemas import ExecutionPlan
from interface.backend.telemetry import trace_operation
from interface.backend.utilities import (
    format_sql_for_trace,
    safe_json_loads,
    serialize_openai_response,
)


MAX_ANALYTICS_ROWS = 100
MAX_ANALYTICS_PARAMS = 50
MAX_ANALYTICS_SQL_LENGTH = 20_000
ANALYTICS_STATEMENT_TIMEOUT_MS = 5_000
DEFAULT_MAX_ANALYTICS_TOTAL_COST = 100_000.0

ANALYTICS_SQL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sql": {"type": "string"},
        "params": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                ],
            },
        },
    },
    "required": ["sql", "params"],
    "additionalProperties": False,
}

ALLOWED_ANALYTICS_TABLES = {
    "videos",
    "stats",
    "speakers",
    "video_speakers",
    "comments",
}
DISALLOWED_SQL_KEYWORDS = re.compile(
    r"\b(?:insert|update|delete|drop|alter|truncate|create|grant|revoke|"
    r"copy|call|do|vacuum|analyze|refresh|merge|execute|prepare|deallocate|"
    r"set|reset|show)\b",
    re.IGNORECASE,
)
DISALLOWED_SQL_OBJECTS = re.compile(
    r"\b(?:pg_catalog|information_schema|pg_[a-z0-9_]*|dblink|"
    r"current_setting|set_config|query_to_xml|database_to_xml|lo_[a-z0-9_]*)\b",
    re.IGNORECASE,
)


ANALYTICS_SCHEMA_PROMPT = """Schéma autorisé (search_path=data,public) :
videos(id, title, description, url, thumbnail_medium_url, video_type, published_at, ...)
stats(id, video_id, view_count, like_count, comment_count, snapshot_date, data_collected_date)
speakers(id, name, title), video_speakers(video_id, speaker_id)
comments(id, video_id, parent_comment_id, author_name, text, like_count, published_at, is_deleted)
video_type : interview, video_recording, long_video, motion_design.
"""


def build_analytics_sql_prompt(
    question: str,
    query: ExecutionPlan,
    database_persons: list[str] | None = None,
    database_companies: list[str] | None = None,
    correction_feedback: str | None = None,
) -> tuple[str, str]:
    system_prompt = f"""Tu es un spécialiste Text-to-SQL PostgreSQL. Produis la requête analytique
qui répond exactement à la question, sans répondre toi-même.

{ANALYTICS_SCHEMA_PROMPT}

Règles :
- Une seule requête SELECT/CTE, tables ci-dessus seulement, sans commentaire ni ;.
- Toute valeur utilisateur va dans params via %s, dans le même ordre. Pas de SELECT *.
- Pour les stats actuelles, prends le dernier snapshot de chaque vidéo avec un LIMIT 1
  corrélé (`WHERE stats.video_id = v.id`), jamais un LIMIT 1 global.
- `title_hints` vide signifie aucun filtre `v.title`; sinon filtre avec l'un des titres fournis.
  N'invente jamais un titre depuis une description. Une interview impose
  `v.video_type = %s` avec `interview`.
- Toute liste de vidéos retourne `video_id`, `video_title`, `video_url` et
  `thumbnail_medium_url`; maximum {MAX_ANALYTICS_ROWS} lignes non agrégées.
- Plusieurs speakers = OR/IN, jamais AND, sauf coapparition explicitement demandée.
- Une comparaison retourne les stats de tous les éléments concernés : jamais de LIMIT
  1 final/global; trie seulement si utile. Toute limite finale doit être au moins 6.
  Le seul LIMIT 1 autorisé est celui, corrélé, qui sélectionne le dernier snapshot.

Exemple unique — comparaison de speakers :
{{"sql":"SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, v.thumbnail_medium_url AS thumbnail_medium_url, s.view_count FROM videos v JOIN video_speakers vs ON vs.video_id = v.id JOIN speakers sp ON sp.id = vs.speaker_id JOIN LATERAL (SELECT view_count FROM stats WHERE video_id = v.id ORDER BY snapshot_date DESC, data_collected_date DESC, id DESC LIMIT 1) s ON TRUE WHERE sp.name ILIKE %s OR sp.name ILIKE %s ORDER BY s.view_count DESC NULLS LAST","params":["%Déborah Rolland%","%Simon Payen%"]}}
"""
    context = {
        "question": question,
        "title_hints": query.title_hints,
        "persons": database_persons or query.persons,
        "companies_requested": query.companies,
        "companies_resolved": database_companies or [],
        "published_after": query.published_after,
        "published_before": query.published_before,
        "correction_feedback": (correction_feedback or "").strip() or None,
    }
    return system_prompt, json.dumps(context, ensure_ascii=False)


def _strip_sql_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def _relation_names(sql: str) -> set[str]:
    names = set()
    for match in re.finditer(
        r"\b(?:from|join)\s+((?:[a-z_][a-z0-9_]*\.)?[a-z_][a-z0-9_]*)",
        sql,
        flags=re.IGNORECASE,
    ):
        name = match.group(1).lower().split(".")[-1]
        if name != "lateral":
            names.add(name)
    return names


def _cte_names(sql: str) -> set[str]:
    return {
        match.group(1).lower()
        for match in re.finditer(
            r"(?:\bwith|,)\s*([a-z_][a-z0-9_]*)\s+as\s*\(",
            sql,
            flags=re.IGNORECASE,
        )
    }


def _valid_param(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_valid_param(item) for item in value)
    return False


def _uses_global_stats_limit_one(sql: str) -> bool:
    """Détecte un LIMIT 1 sur stats qui n'est pas borné à une vidéo."""
    parentheses: list[tuple[int, int]] = []
    stack: list[int] = []
    for index, character in enumerate(sql):
        if character == "(":
            stack.append(index)
        elif character == ")" and stack:
            parentheses.append((stack.pop(), index))

    for match in re.finditer(
        r"\bfrom\s+(?:(?:data|public)\.)?stats\b",
        sql,
        flags=re.IGNORECASE,
    ):
        containing_scopes = [
            (start, end)
            for start, end in parentheses
            if start < match.start() < end
        ]
        if containing_scopes:
            start, end = min(containing_scopes, key=lambda scope: scope[1] - scope[0])
            scope_sql = sql[start + 1:end]
        else:
            scope_sql = sql

        if not re.search(r"\blimit\s+1\b", scope_sql, flags=re.IGNORECASE):
            continue
        if re.search(
            r"\bdistinct\s+on\s*\([^)]*\bvideo_id\b|"
            r"\bpartition\s+by\b[^)]*\bvideo_id\b",
            scope_sql,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(
            r"(?:\b[a-z_][a-z0-9_]*\.)?video_id\s*=\s*"
            r"(?:%s|[a-z_][a-z0-9_]*\.id\b)|"
            r"\b[a-z_][a-z0-9_]*\.id\s*=\s*"
            r"(?:[a-z_][a-z0-9_]*\.)?video_id\b",
            scope_sql,
            flags=re.IGNORECASE,
        ):
            continue
        return True
    return False


def validate_analytics_sql(sql: str, params: list[Any]) -> dict[str, Any]:
    normalized = str(sql or "").strip()
    errors: list[str] = []
    if not normalized:
        errors.append("empty_sql")
    if len(normalized) > MAX_ANALYTICS_SQL_LENGTH:
        errors.append("sql_too_long")
    if "--" in normalized or "/*" in normalized or "*/" in normalized:
        errors.append("comments_forbidden")
    if ";" in normalized:
        errors.append("multiple_or_terminated_statements_forbidden")
    if re.search(r"'(?:''|[^'])*'", normalized):
        errors.append("literal_values_forbidden")

    sql_without_literals = _strip_sql_literals(normalized)
    if normalized and not re.match(r"^(?:select|with)\b", normalized, re.IGNORECASE):
        errors.append("select_only")
    if DISALLOWED_SQL_KEYWORDS.search(sql_without_literals):
        errors.append("disallowed_keyword")
    if DISALLOWED_SQL_OBJECTS.search(sql_without_literals):
        errors.append("disallowed_object")
    if _uses_global_stats_limit_one(sql_without_literals):
        errors.append("stats_latest_snapshot_not_per_video")

    relations = _relation_names(sql_without_literals)
    ctes = _cte_names(sql_without_literals)
    unknown_relations = sorted(relations - ALLOWED_ANALYTICS_TABLES - ctes)
    if not relations:
        errors.append("missing_allowed_relation")
    if unknown_relations:
        errors.append("unknown_relations:" + ",".join(unknown_relations))

    if not isinstance(params, list):
        errors.append("params_must_be_list")
        params = []
    if len(params) > MAX_ANALYTICS_PARAMS:
        errors.append("too_many_params")
    if not all(_valid_param(value) for value in params):
        errors.append("invalid_param_type")
    if normalized.count("%s") != len(params):
        errors.append("placeholder_count_mismatch")

    return {
        "valid": not errors,
        "errors": errors,
        "relations": sorted(relations),
        "ctes": sorted(ctes),
        "placeholder_count": normalized.count("%s"),
        "param_count": len(params),
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def analytics_rows_to_sources(
    columns: list[str],
    rows: list[Any],
) -> list[dict[str, Any]]:
    sources = []
    for index, row in enumerate(rows):
        record = {
            column: _json_compatible(value)
            for column, value in zip(columns, row)
        }
        video_id = record.get("video_id")
        chunk_id = int(video_id) if isinstance(video_id, int) else -(index + 1)
        video_title = str(
            record.get("video_title")
            or record.get("title")
            or "Résultat analytique"
        )
        video_url = str(record.get("video_url") or record.get("url") or "")
        thumbnail_medium_url = str(
            record.get("thumbnail_medium_url") or ""
        ).strip() or None
        text = "\n".join(
            f"{column}: {json.dumps(value, ensure_ascii=False, default=str)}"
            for column, value in record.items()
        )
        sources.append(
            {
                "chunk_id": chunk_id,
                "video_title": video_title,
                "video_url": video_url,
                "thumbnail_medium_url": thumbnail_medium_url,
                "chunk_index": 0,
                "text": text,
                "persons": [],
                "bm25_score": None,
            }
        )
    return sources


def bounded_analytics_sql(sql: str) -> str:
    return (
        "SELECT * FROM (\n"
        + sql
        + f"\n) AS generated_analytics_query LIMIT {MAX_ANALYTICS_ROWS}"
    )


def max_analytics_total_cost() -> float:
    raw_value = os.getenv("ANALYTICS_MAX_TOTAL_COST", "").strip()
    if not raw_value:
        return DEFAULT_MAX_ANALYTICS_TOTAL_COST
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError("ANALYTICS_MAX_TOTAL_COST doit être un nombre positif.") from exc
    if value <= 0:
        raise RuntimeError("ANALYTICS_MAX_TOTAL_COST doit être un nombre positif.")
    return value


def explain_total_cost(explain_plan: Any) -> float | None:
    current = explain_plan
    if isinstance(current, list) and current:
        current = current[0]
    if not isinstance(current, dict):
        return None
    plan = current.get("Plan", current)
    if not isinstance(plan, dict):
        return None
    total_cost = plan.get("Total Cost")
    try:
        return float(total_cost)
    except (TypeError, ValueError):
        return None


def summarize_cost_validation(cost_validation: dict[str, Any]) -> dict[str, Any]:
    """Keep only the small, actionable part of an EXPLAIN validation result."""
    summary_keys = (
        "valid",
        "status",
        "total_cost",
        "max_total_cost",
        "error",
    )
    return {
        key: cost_validation[key]
        for key in summary_keys
        if key in cost_validation
    }


def explain_analytics_sql(sql: str, params: list[Any]) -> dict[str, Any]:
    bounded_sql = bounded_analytics_sql(sql)
    with connect_analytics_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                f"SET LOCAL statement_timeout = '{ANALYTICS_STATEMENT_TIMEOUT_MS}ms'"
            )
            cursor.execute("EXPLAIN (FORMAT JSON) " + bounded_sql, params)
            explain_row = cursor.fetchone()
            explain_plan = explain_row[0] if explain_row else None
    total_cost = explain_total_cost(explain_plan)
    max_total_cost = max_analytics_total_cost()
    return {
        "valid": total_cost is not None and total_cost <= max_total_cost,
        "total_cost": total_cost,
        "max_total_cost": max_total_cost,
    }


def execute_analytics_sql(
    sql: str,
    params: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bounded_sql = bounded_analytics_sql(sql)
    with connect_analytics_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                f"SET LOCAL statement_timeout = '{ANALYTICS_STATEMENT_TIMEOUT_MS}ms'"
            )
            cursor.execute(bounded_sql, params)
            rows = cursor.fetchall()
            columns = []
            for column in cursor.description or []:
                column_name = getattr(column, "name", None)
                if column_name is None:
                    column_name = column[0]
                columns.append(str(column_name))
    sources = analytics_rows_to_sources(columns, rows)
    return sources, {
        "bounded_sql": format_sql_for_trace(bounded_sql),
        "columns": columns,
        "row_count": len(rows),
    }


def run_analytics_text_to_sql(
    query: ExecutionPlan,
    client: LLMClientProtocol | None,
    model: str,
    *,
    database_persons: list[str] | None = None,
    database_companies: list[str] | None = None,
    correction_feedback: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace: dict[str, Any] = {
        "mode": "analytics",
        "strategy": "llm_text_to_sql",
        "model": model,
        "sql": None,
        "params": [],
        "result_count": 0,
    }
    if client is None:
        trace["status"] = "no_llm_client"
        return [], trace

    system_prompt, user_prompt = build_analytics_sql_prompt(
        query.query_text or query.raw_question,
        query,
        database_persons,
        database_companies,
        correction_feedback,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    with trace_operation(
            "analytics.sql_generation",
        kind="AGENT",
        input_value={"question": query.raw_question, "model": model},
    ) as generation_span:
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                max_output_tokens=2_000,
                response_schema=ANALYTICS_SQL_RESPONSE_SCHEMA,
            )
            raw_response = serialize_openai_response(response)
            payload = safe_json_loads(
                str(getattr(response, "output_text", "") or "").strip()
            )
            sql = str(payload.get("sql") or "").strip()
            params = payload.get("params", [])
            generation_trace = {
                "status": "generated",
                "model": model,
                "prompt": messages,
                "response_raw": raw_response,
                "sql": format_sql_for_trace(sql),
                "params": params,
            }
        except Exception as exc:
            generation_trace = {
                "status": "generation_error",
                "model": model,
                "error": str(exc),
            }
            generation_span.set_output(generation_trace)
            trace.update(generation_trace)
            return [], trace
        generation_span.set_output(generation_trace)

    with trace_operation(
            "analytics.sql_validation",
        kind="GUARDRAIL",
        input_value={"sql": format_sql_for_trace(sql), "params": params},
    ) as validation_span:
        validation = validate_analytics_sql(sql, params)
        validation_span.set_output(validation)
    trace.update(
        {
            "prompt": messages,
            "response_raw": generation_trace["response_raw"],
            "sql": format_sql_for_trace(sql),
            "params": params if isinstance(params, list) else [],
            "validation": validation,
        }
    )
    if not validation["valid"]:
        trace["status"] = "validation_rejected"
        return [], trace

    with trace_operation(
            "analytics.sql_cost_validation",
        kind="GUARDRAIL",
        input_value={"sql": format_sql_for_trace(sql), "params": params},
    ) as cost_span:
        try:
            cost_validation = explain_analytics_sql(sql, params)
        except Exception as exc:
            cost_validation = {
                "valid": False,
                "status": "explain_error",
                "error": str(exc),
            }
        cost_validation_summary = summarize_cost_validation(cost_validation)
        cost_span.set_output(cost_validation_summary)
    trace["cost_validation"] = cost_validation_summary
    if not cost_validation["valid"]:
        trace["status"] = "cost_rejected"
        return [], trace

    with trace_operation(
            "analytics.sql_execution",
        kind="TOOL",
        input_value={"sql": format_sql_for_trace(sql), "params": params},
    ) as execution_span:
        try:
            sources, execution = execute_analytics_sql(sql, params)
            execution_span.set_output({**execution, "results": sources})
        except Exception as exc:
            execution = {"status": "execution_error", "error": str(exc)}
            execution_span.set_output(execution)
            trace.update(execution)
            return [], trace

    trace.update(
        {
            "status": "executed",
            "execution": execution,
            "result_count": len(sources),
        }
    )
    return sources, trace


def _analytics_text(value_sql: str) -> str:
    return (
        "btrim(regexp_replace("
        f"unaccent(lower(coalesce({value_sql}, ''))), "
        "'[^[:alnum:]]+', ' ', 'g'))"
    )


def _analytics_video_date_filters(query: ExecutionPlan) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if query.published_after:
        clauses.append("v.published_at >= %s::timestamptz")
        params.append(query.published_after)
    if query.published_before:
        clauses.append("v.published_at <= %s::timestamptz")
        params.append(query.published_before)
    return clauses, params


def _resolved_analytics_entities(
    query: ExecutionPlan,
    database_persons: list[str] | None,
    database_companies: list[str] | None,
) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    for kind, values in (
        ("person", database_persons or query.persons),
        ("company", database_companies or query.companies),
        ("title", query.title_hints),
    ):
        for value in values:
            cleaned = str(value or "").strip()
            entity = {"kind": kind, "value": cleaned}
            if cleaned and entity not in entities:
                entities.append(entity)
    return entities


def _lookup_analytics_entity_videos(
    entity: dict[str, str],
    query: ExecutionPlan,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind, value = entity["kind"], entity["value"]
    date_clauses, date_params = _analytics_video_date_filters(query)
    if kind == "person":
        entity_clause = (
            "(EXISTS (SELECT 1 FROM video_speakers vs "
            "JOIN speakers sp ON sp.id = vs.speaker_id "
            "WHERE vs.video_id = v.id "
            f"AND {_analytics_text('sp.name')} = {_analytics_text('%s')}) "
            "OR EXISTS (SELECT 1 FROM transcripts transcript_row "
            "WHERE transcript_row.video_id = v.id "
            "AND transcript_row.transcript_enriched IS NOT NULL "
            "AND concat(' ', "
            f"{_analytics_text('transcript_row.transcript_enriched')}, ' ') "
            "LIKE concat(chr(37), ' ', "
            f"{_analytics_text('%s')}, ' ', chr(37))))"
        )
    elif kind == "company":
        entity_clause = (
            "EXISTS (SELECT 1 FROM video_speakers vs "
            "JOIN speakers sp ON sp.id = vs.speaker_id "
            "WHERE vs.video_id = v.id AND sp.title IS NOT NULL "
            f"AND {_analytics_text('sp.title')} LIKE "
            f"concat(chr(37), {_analytics_text('%s')}, chr(37)))"
        )
    else:
        entity_clause = f"{_analytics_text('v.title')} = {_analytics_text('%s')}"

    where_clauses = [entity_clause, *date_clauses]
    sql = f"""
        SELECT DISTINCT v.id, v.title, v.url, v.thumbnail_medium_url
        FROM videos v
        WHERE {' AND '.join(where_clauses)}
        ORDER BY v.id ASC
        LIMIT {MAX_ANALYTICS_ROWS}
    """
    params = [value, value, *date_params] if kind == "person" else [value, *date_params]
    with connect_analytics_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                f"SET LOCAL statement_timeout = '{ANALYTICS_STATEMENT_TIMEOUT_MS}ms'"
            )
            cursor.execute(sql, params)
            rows = cursor.fetchall()
    videos = [
        {
            "video_id": int(row[0]),
            "video_title": str(row[1] or ""),
            "video_url": str(row[2] or ""),
            "thumbnail_medium_url": str(row[3] or "").strip() or None,
        }
        for row in rows
    ]
    return videos, {"sql": format_sql_for_trace(sql), "params": params}


def _global_analytics_ranking_sources(
    query: ExecutionPlan,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the planner-requested global ranking window from one SQL query."""
    date_clauses, params = _analytics_video_date_filters(query)
    where_clause = f"WHERE {' AND '.join(date_clauses)}" if date_clauses else ""
    ranking_columns = {
        ("views", "desc"): ("view_count", "views_top_rank", "views", "top"),
        ("views", "asc"): ("view_count", "views_bottom_rank", "views", "bottom"),
        ("likes", "desc"): ("like_count", "likes_top_rank", "likes", "top"),
        ("likes", "asc"): ("like_count", "likes_bottom_rank", "likes", "bottom"),
        ("comments", "desc"): ("comment_count", "comments_top_rank", "comments", "top"),
        ("comments", "asc"): ("comment_count", "comments_bottom_rank", "comments", "bottom"),
    }
    metric = query.analytics_metric or "all"
    rank_start, rank_end = query.analytics_rank_start or 1, query.analytics_rank_end or 3
    sql = f"""
        WITH ranked AS (
            SELECT
                v.id AS video_id,
                v.title AS video_title,
                v.url AS video_url,
                v.thumbnail_medium_url,
                v.video_type,
                v.published_at,
                latest.snapshot_date,
                latest.view_count,
                latest.like_count,
                latest.comment_count,
                COUNT(*) OVER () AS population_video_count,
                MIN(v.published_at) OVER () AS first_published_at,
                MAX(v.published_at) OVER () AS last_published_at,
                row_number() OVER (ORDER BY latest.view_count DESC NULLS LAST, v.id ASC) AS views_top_rank,
                row_number() OVER (ORDER BY latest.view_count ASC NULLS LAST, v.id ASC) AS views_bottom_rank,
                row_number() OVER (ORDER BY latest.like_count DESC NULLS LAST, v.id ASC) AS likes_top_rank,
                row_number() OVER (ORDER BY latest.like_count ASC NULLS LAST, v.id ASC) AS likes_bottom_rank,
                row_number() OVER (ORDER BY latest.comment_count DESC NULLS LAST, v.id ASC) AS comments_top_rank,
                row_number() OVER (ORDER BY latest.comment_count ASC NULLS LAST, v.id ASC) AS comments_bottom_rank
            FROM videos v
            JOIN LATERAL (
                SELECT snapshot_date, view_count, like_count, comment_count
                FROM stats
                WHERE video_id = v.id
                ORDER BY snapshot_date DESC, data_collected_date DESC, id DESC
                LIMIT 1
            ) latest ON TRUE
            {where_clause}
        )
        {{ranking_query}}
        ORDER BY metric, direction, ranking
    """
    ranking_queries: list[str] = []
    ranking_params: list[Any] = []
    selected_rankings = (
        list(ranking_columns.values())
        if metric == "all"
        else [ranking_columns[(metric, query.analytics_order or "desc")]]
    )
    for value_column, rank_column, metric_label, direction in selected_rankings:
        ranking_queries.append(
            f"SELECT *, '{metric_label}' AS metric, '{direction}' AS direction, {rank_column} AS ranking "
            f"FROM ranked WHERE {value_column} IS NOT NULL AND {rank_column} BETWEEN %s AND %s"
        )
        ranking_params.extend([rank_start, rank_end])
    sql = sql.replace("{ranking_query}", "\n        UNION ALL\n        ".join(ranking_queries))
    with connect_analytics_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                f"SET LOCAL statement_timeout = '{ANALYTICS_STATEMENT_TIMEOUT_MS}ms'"
            )
            cursor.execute(sql, [*params, *ranking_params])
            rows = cursor.fetchall()
    sources: list[dict[str, Any]] = []
    population_video_count = int(rows[0][10]) if rows else 0
    first_published_at = _json_compatible(rows[0][11]) if rows else None
    last_published_at = _json_compatible(rows[0][12]) if rows else None
    for row in rows:
        metric, direction, ranking = str(row[19]), str(row[20]), int(row[21])
        value_index = {"views": 7, "likes": 8, "comments": 9}[metric]
        value = _json_compatible(row[value_index])
        snapshot = {
            "snapshot_date": _json_compatible(row[6]),
            "view_count": _json_compatible(row[7]),
            "like_count": _json_compatible(row[8]),
            "comment_count": _json_compatible(row[9]),
        }
        label = {"views": "vues", "likes": "likes", "comments": "commentaires"}[metric]
        direction_label = "plus élevées" if direction == "top" else "plus faibles"
        sources.append(
            {
                "chunk_id": int(row[0]),
                "video_title": str(row[1] or ""),
                "video_url": str(row[2] or ""),
                "thumbnail_medium_url": str(row[3] or "").strip() or None,
                "chunk_index": 0,
                "persons": [],
                "bm25_score": None,
                "video_type": str(row[4] or "") or None,
                "published_at": _json_compatible(row[5]),
                "stats": [snapshot],
                "global_ranking": {"metric": metric, "direction": direction, "rank": ranking, "value": value},
                "text": (
                    f"Population analysée : {population_video_count} vidéos. "
                    f"Première publication : {first_published_at or 'inconnue'}. "
                    f"Dernière publication : {last_published_at or 'inconnue'}. "
                    f"Classement global — {label} {direction_label}, rang {ranking}: "
                    f"{value} ({snapshot['snapshot_date']})."
                ),
            }
        )
    return sources, {
        "sql": format_sql_for_trace(sql),
        "params": [*params, *ranking_params],
        "population_video_count": population_video_count,
        "first_published_at": first_published_at,
        "last_published_at": last_published_at,
        "ranking_result_count": len(sources),
        "analytics_metric": metric,
        "analytics_order": query.analytics_order,
        "analytics_rank_start": rank_start,
        "analytics_rank_end": rank_end,
    }


def _analytics_stats_sources(video_ids: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not video_ids:
        return [], {"sql": None, "params": [], "video_count": 0, "snapshot_count": 0}
    sql = """
        SELECT
            v.id AS video_id,
            v.title AS video_title,
            v.url AS video_url,
            v.thumbnail_medium_url,
            v.video_type,
            v.published_at,
            s.snapshot_date,
            s.view_count,
            s.like_count,
            s.comment_count
        FROM videos v
        LEFT JOIN stats s ON s.video_id = v.id
        WHERE v.id = ANY(%s)
        ORDER BY v.id ASC, s.snapshot_date ASC NULLS LAST
    """
    with connect_analytics_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                f"SET LOCAL statement_timeout = '{ANALYTICS_STATEMENT_TIMEOUT_MS}ms'"
            )
            cursor.execute(sql, [video_ids])
            rows = cursor.fetchall()

    videos: dict[int, dict[str, Any]] = {}
    for row in rows:
        video_id = int(row[0])
        video = videos.setdefault(
            video_id,
            {
                "chunk_id": video_id,
                "video_title": str(row[1] or ""),
                "video_url": str(row[2] or ""),
                "thumbnail_medium_url": str(row[3] or "").strip() or None,
                "chunk_index": 0,
                "persons": [],
                "bm25_score": None,
                "video_type": str(row[4] or "") or None,
                "published_at": _json_compatible(row[5]),
                "stats": [],
            },
        )
        if row[6] is not None:
            video["stats"].append(
                {
                    "snapshot_date": _json_compatible(row[6]),
                    "view_count": _json_compatible(row[7]),
                    "like_count": _json_compatible(row[8]),
                    "comment_count": _json_compatible(row[9]),
                }
            )

    sources = []
    for video_id in video_ids:
        video = videos.get(video_id)
        if video is None:
            continue
        snapshots = video["stats"]
        stats_lines = [
            "- " + ", ".join(
                f"{key}: {value}" for key, value in snapshot.items() if value is not None
            )
            for snapshot in snapshots
        ]
        video["text"] = "Historique complet des statistiques:\n" + (
            "\n".join(stats_lines) or "- Aucune statistique disponible"
        )
        sources.append(video)
    return sources, {
        "sql": format_sql_for_trace(sql),
        "params": [video_ids],
        "video_count": len(sources),
        "snapshot_count": sum(len(source["stats"]) for source in sources),
    }


def run_deterministic_analytics(
    query: ExecutionPlan,
    *,
    database_persons: list[str] | None = None,
    database_companies: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load stats for either an entity-filtered or complete analytics population."""
    entities = _resolved_analytics_entities(query, database_persons, database_companies)
    analytics_scope = query.analytics_scope or "specific"
    trace: dict[str, Any] = {
        "mode": "analytics",
        "strategy": f"deterministic_{analytics_scope}_stats",
        "analytics_scope": analytics_scope,
        "entities": entities,
        "entity_lookups": [],
        "candidate_video_count": 0,
        "result_count": 0,
    }
    videos_by_id: dict[int, dict[str, Any]] = {}
    if analytics_scope == "global":
        with trace_operation(
            "analytics_global_rankings",
            kind="RETRIEVER",
            input_value={"published_after": query.published_after, "published_before": query.published_before},
        ) as global_span:
            sources, rankings_trace = _global_analytics_ranking_sources(query)
            global_span.set_output({**rankings_trace, "results": sources})
        trace["global_rankings"] = rankings_trace
        trace["candidate_video_count"] = rankings_trace["population_video_count"]
        trace["result_count"] = len(sources)
        return sources, trace
    else:
        for entity in entities:
            with trace_operation(
                "analytics_entity_lookup",
                kind="RETRIEVER",
                input_value=entity,
            ) as entity_span:
                videos, lookup_trace = _lookup_analytics_entity_videos(entity, query)
                entity_output = {**entity, "result_count": len(videos), "results": videos}
                entity_span.set_output(entity_output)
            trace["entity_lookups"].append({**entity, **lookup_trace, "result_count": len(videos), "results": videos})
            for video in videos:
                videos_by_id.setdefault(int(video["video_id"]), video)

    video_ids = list(videos_by_id)
    trace["candidate_video_count"] = len(video_ids)
    with trace_operation(
        "analytics_total_videos",
        kind="RETRIEVER",
        input_value={"entity_count": len(entities)},
    ) as total_videos_span:
        total_videos_span.set_output(
            {
                "candidate_video_count": len(video_ids),
                "video_ids": video_ids,
            }
        )
    with trace_operation(
        "analytics_all_video_stats",
        kind="RETRIEVER",
        input_value={"video_ids": video_ids},
    ) as stats_span:
        sources, stats_trace = _analytics_stats_sources(video_ids)
        stats_span.set_output({**stats_trace, "results": sources})
    trace["stats"] = stats_trace
    trace["result_count"] = len(sources)
    return sources, trace
