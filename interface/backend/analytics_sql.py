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


ANALYTICS_SCHEMA_PROMPT = """Schéma PostgreSQL autorisé (search_path=data,public) :

videos(
  id bigint primary key,
  youtube_video_id text,
  title text,
  description text,
  url text,
  duration_seconds integer,
  is_long_video boolean,
  thumbnail_medium_url text,
  has_subtitles boolean,
  video_type text,
  published_at timestamptz,
  data_collected_date timestamptz
)
Valeurs connues de videos.video_type : interview, video_recording, long_video, motion_design.

stats(
  id bigint primary key,
  video_id bigint references videos(id),
  view_count bigint,
  like_count bigint,
  comment_count bigint,
  snapshot_date date,
  data_collected_date timestamptz
)
Il existe plusieurs snapshots par vidéo. Pour les statistiques actuelles, sélectionner
exactement le snapshot le plus récent de chaque vidéo avec :
ORDER BY snapshot_date DESC, data_collected_date DESC, id DESC LIMIT 1.

speakers(id bigint primary key, name text, title text, data_collected_date timestamptz)
video_speakers(video_id bigint, speaker_id bigint, data_collected_date timestamptz)
comments(id bigint primary key, video_id bigint, parent_comment_id bigint,
         author_name text, text text, like_count bigint, published_at timestamptz,
         updated_at timestamptz, is_deleted boolean)
"""


def build_analytics_sql_prompt(
    question: str,
    query: ExecutionPlan,
    database_persons: list[str] | None = None,
    database_companies: list[str] | None = None,
) -> tuple[str, str]:
    system_prompt = f"""Tu es un spécialiste Text-to-SQL PostgreSQL. Produis la requête analytique
qui répond exactement à la question, sans répondre toi-même.

{ANALYTICS_SCHEMA_PROMPT}

Règles obligatoires :
- Retourne uniquement un objet JSON avec exactement les clés sql et params.
- sql doit être une unique requête SELECT, éventuellement précédée de CTE WITH.
- N'utilise que les tables du schéma fourni.
- N'utilise jamais SELECT *, sauf dans un COUNT(*).
- Utilise des placeholders psycopg %s pour toute valeur issue de la question et place
  ces valeurs, dans le même ordre, dans params.
- N'ajoute ni commentaire SQL ni point-virgule.
- Pour un classement ou un extremum, trie sur la métrique demandée et applique la
  limite utile. Ne trie pas par date de publication sauf demande explicite.
- Pour compter des vidéos, interroge videos et utilise COUNT(DISTINCT v.id).
- Pour une interview, filtre v.video_type = %s avec la valeur interview dans params.
- Traite title_hint et les titres mentionnés comme des fragments : utilise v.title ILIKE %s
  avec une valeur entourée de %, sauf si un identifiant vidéo exact est fourni.
- Pour une métadonnée portant sur une vidéo, retourne toujours aussi v.id AS video_id,
  v.title AS video_title et v.url AS video_url.
- Pour une statistique YouTube actuelle, utilise uniquement le dernier snapshot de
  chaque vidéo selon la règle du schéma.
- Quand une ligne correspond à une vidéo, expose si possible les alias video_id,
  video_title et video_url afin de rendre le résultat lisible.
- Les requêtes non agrégées doivent retourner au maximum {MAX_ANALYTICS_ROWS} lignes.

Exemple « quelle vidéo a le plus de vues ? » :
{{"sql":"SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, s.view_count FROM videos v JOIN LATERAL (SELECT view_count FROM stats WHERE video_id = v.id ORDER BY snapshot_date DESC, data_collected_date DESC, id DESC LIMIT 1) s ON TRUE ORDER BY s.view_count DESC NULLS LAST LIMIT %s","params":[1]}}

Exemple « combien de vidéos interview sur la chaîne ? » :
{{"sql":"SELECT COUNT(DISTINCT v.id) AS video_count FROM videos v WHERE v.video_type = %s","params":["interview"]}}
"""
    context = {
        "question": question,
        "title_hint": query.title_hint,
        "persons_requested": query.persons,
        "persons_resolved": database_persons or [],
        "companies_requested": query.companies,
        "companies_resolved": database_companies or [],
        "published_after": query.published_after,
        "published_before": query.published_before,
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
        text = "\n".join(
            f"{column}: {json.dumps(value, ensure_ascii=False, default=str)}"
            for column, value in record.items()
        )
        sources.append(
            {
                "chunk_id": chunk_id,
                "video_title": video_title,
                "video_url": video_url,
                "thumbnail_medium_url": None,
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
        "explain": explain_plan,
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
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    with trace_operation(
        "rag.analytics.sql_generation",
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
        "rag.analytics.sql_validation",
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
        "rag.analytics.sql_cost_validation",
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
        cost_span.set_output(cost_validation)
    trace["cost_validation"] = cost_validation
    if not cost_validation["valid"]:
        trace["status"] = "cost_rejected"
        return [], trace

    with trace_operation(
        "rag.analytics.sql_execution",
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
