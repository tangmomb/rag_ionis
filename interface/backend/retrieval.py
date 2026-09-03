from __future__ import annotations

from typing import Any

from interface.backend.config import (
    DEFAULT_BM25_LIMIT,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FINAL_K,
    DEFAULT_FUSION_K,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PREFILTER_LIMIT,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RRF_TOP_N,
    DEFAULT_VECTOR_LIMIT,
)
from interface.backend.database import connect_database
from interface.backend.schemas import ExecutionPlan, RagRequest
from interface.backend.telemetry import trace_operation
from interface.backend.utilities import (
    format_sql_for_trace,
    format_sql_pretty,
    get_cohere_client,
    get_openai_client,
    normalize_model_name,
    resolve_cohere_rerank_model,
)

VIDEO_PERSON_NAMES_SQL = """
ARRAY(
    SELECT person_row.name
    FROM video_speakers video_person
    JOIN speakers person_row ON person_row.id = video_person.speaker_id
    WHERE video_person.video_id = v.id
    ORDER BY person_row.id
)
"""

VIDEO_PERSON_DETAILS_SQL = """
COALESCE(
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'name', person_row.name,
                'title', person_row.title
            )
            ORDER BY person_row.id
        )
        FROM video_speakers video_person
        JOIN speakers person_row ON person_row.id = video_person.speaker_id
        WHERE video_person.video_id = v.id
    ),
    '[]'::jsonb
)
"""

VIDEO_TRANSCRIPT_DOCUMENT_SQL = """
(
    SELECT COALESCE(
        transcript_row.transcript_enriched,
        transcript_row.transcript
    )
    FROM transcripts transcript_row
    WHERE transcript_row.video_id = v.id
    ORDER BY
        CASE WHEN transcript_row.language_code = 'fr' THEN 0 ELSE 1 END,
        transcript_row.id DESC
    LIMIT 1
)
"""


def normalized_sql_text(value_sql: str) -> str:
    """Normalise accents, casse, ponctuation et espaces dans une expression SQL."""
    return (
        "btrim(regexp_replace("
        f"unaccent(lower(coalesce({value_sql}, ''))), "
        "'[^[:alnum:]]+', ' ', 'g'))"
    )


TITLE_CONTAINS_SQL = (
    f"{normalized_sql_text('v.title')} LIKE "
    f"concat(chr(37), {normalized_sql_text('%s')}, chr(37))"
)
TITLE_HINTS_CONTAINS_SQL = (
    f"{normalized_sql_text('v.title')} LIKE ANY(ARRAY("
    f"SELECT concat(chr(37), {normalized_sql_text('title_hint')}, chr(37)) "
    "FROM unnest(%s::text[]) AS title_hint))"
)


def trace_formatted_sql(span_name: str, trace: dict[str, Any]) -> None:
    persons_table = trace.get("persons_table")
    sql_entries: list[tuple[str | None, dict[str, Any]]] = []
    if isinstance(persons_table, dict) and persons_table.get("sql"):
        sql_entries.append(("persons_in_speakers", persons_table))
    if trace.get("sql"):
        sql_entries.append(
            (
                "persons_in_transcripts" if sql_entries else None,
                trace,
            )
        )

    for label, sql_trace in sql_entries:
        formatted_sql = format_sql_pretty(sql_trace.get("sql"))
        if formatted_sql is None:
            continue
        formatted_span_name = (
            label
            if label
            else f"{span_name}.sql_formatted"
        )
        with trace_operation(
            formatted_span_name,
            kind="CHAIN",
            input_value={"params": sql_trace.get("params", [])},
        ) as sql_span:
            query_results = sql_trace.get("query_results")
            if isinstance(query_results, list):
                sql_span.set_output(
                    {
                        "sql": formatted_sql,
                        "result_count": sql_trace.get(
                            "query_result_count", len(query_results)
                        ),
                        "results": summarize_sql_results(query_results),
                    }
                )
            else:
                sql_span.set_output_text(formatted_sql)

    deduplication = trace.get("deduplication")
    if isinstance(deduplication, dict):
        with trace_operation(
            "deduplicate_videos",
            kind="TOOL",
            input_value={
                "sources": ["persons_in_speakers", "persons_in_transcripts"],
                "input_count": deduplication.get("input_count", 0),
            },
        ) as deduplication_span:
            deduplication_span.set_output(deduplication)


def summarize_sql_results(results: list[Any]) -> list[dict[str, Any]]:
    """Serialize each SQL query result for its Phoenix span."""
    fields = (
        "chunk_id",
        "video_title",
        "video_url",
        "thumbnail_medium_url",
        "persons",
        "person_details",
        "video_type",
        "text",
    )
    return [
        {field: result[field] for field in fields if field in result}
        for result in results
        if isinstance(result, dict)
    ]


def append_person_filter_clauses(clauses: list[str], params: list[Any], persons: list[str]) -> None:
    cleaned_persons = [person.strip() for person in persons if person.strip()]
    if not cleaned_persons:
        return
    name_conditions = " OR ".join(
        f"{normalized_sql_text('person_row.name')} = {normalized_sql_text('%s')}"
        for _ in cleaned_persons
    )
    clauses.append(
        f"""
        EXISTS (
            SELECT 1
            FROM video_speakers video_person
            JOIN speakers person_row ON person_row.id = video_person.speaker_id
            WHERE video_person.video_id = v.id
              AND ({name_conditions})
        )
        """
    )
    params.extend(cleaned_persons)


def append_company_filter_clauses(
    clauses: list[str],
    params: list[Any],
    companies: list[str],
) -> None:
    cleaned_companies = [
        company.strip() for company in companies if company.strip()
    ]
    if not cleaned_companies:
        return
    title_conditions = " OR ".join(
        f"{normalized_sql_text('person_row.title')} LIKE "
        f"concat(chr(37), {normalized_sql_text('%s')}, chr(37))"
        for _ in cleaned_companies
    )
    clauses.append(
        f"""
        EXISTS (
            SELECT 1
            FROM video_speakers video_person
            JOIN speakers person_row ON person_row.id = video_person.speaker_id
            WHERE video_person.video_id = v.id
              AND person_row.title IS NOT NULL
              AND ({title_conditions})
        )
        """
    )
    params.extend(cleaned_companies)


def build_prefilter_conditions(query: ExecutionPlan) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if query.title_hints:
        clauses.append(TITLE_HINTS_CONTAINS_SQL)
        params.append(query.title_hints)
    if query.published_after:
        clauses.append("v.published_at >= %s::timestamptz")
        params.append(query.published_after)
    if query.published_before:
        clauses.append("v.published_at <= %s::timestamptz")
        params.append(query.published_before)
    return clauses, params


def prefilter_candidate_chunk_ids(query: ExecutionPlan) -> tuple[list[int] | None, dict[str, Any]]:
    clauses, params = build_prefilter_conditions(query)
    if not clauses:
        return None, {
            "applied": False,
            "general_question_only": True,
            "candidate_chunk_ids": None,
            "candidate_count": None,
            "sql": None,
            "params": [],
            "sql_prefilters": False,
        }

    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT c.id
        FROM chunks c
        JOIN videos v ON v.id = c.video_id
        WHERE c.chunk_level = 'detail'
          AND {where_sql}
        ORDER BY c.id ASC
        LIMIT %s
    """
    sql_params = [*params, DEFAULT_PREFILTER_LIMIT]
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, sql_params)
            rows = cursor.fetchall()

    candidate_ids = [int(row[0]) for row in rows]
    return candidate_ids, {
        "applied": True,
        "general_question_only": False,
        "candidate_chunk_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "sql": format_sql_for_trace(sql),
        "params": sql_params,
        "sql_prefilters": True,
    }


def query_terms(query: ExecutionPlan) -> str:
    return query.query_text.strip() or query.raw_question


def candidate_sql_clause(candidate_chunk_ids: list[int] | None) -> tuple[str, list[Any]]:
    if candidate_chunk_ids is None:
        return "", []
    if not candidate_chunk_ids:
        return " AND 1 = 0", []
    return " AND c.id = ANY(%s)", [candidate_chunk_ids]


def build_video_lookup_conditions(
    query: ExecutionPlan,
    database_persons: list[str] | None = None,
    database_company: list[str] | None = None,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    title_hints = query.title_hints
    if title_hints:
        clauses.append(TITLE_HINTS_CONTAINS_SQL)
        params.append(title_hints)
    if database_persons:
        append_person_filter_clauses(clauses, params, database_persons)
    if database_company:
        append_company_filter_clauses(clauses, params, database_company)
    elif database_company is not None and query.companies:
        clauses.append("1 = 0")
    if query.published_after:
        clauses.append("v.published_at >= %s::timestamptz")
        params.append(query.published_after)
    if query.published_before:
        clauses.append("v.published_at <= %s::timestamptz")
        params.append(query.published_before)
    return clauses, params


def lookup_video_document(
    query: ExecutionPlan,
    intent: str,
    *,
    database_persons: list[str] | None = None,
    database_company: list[str] | None = None,
    transcript_persons: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if intent == "specific_persons":
        def format_lookup_rows(
            rows: list[Any],
            *,
            transcript_persons: list[str] | None = None,
        ) -> list[dict[str, Any]]:
            results = []
            for row in rows:
                person_details = row[4] or []
                transcript = row[5]
                person_lines = [
                    f"- {item.get('name')}: {item.get('title') or 'poste non disponible'}"
                    for item in person_details
                    if isinstance(item, dict) and item.get("name")
                ]
                transcript_line = (
                    "\nPersonnes trouvées dans le transcript: "
                    + ", ".join(transcript_persons)
                    if transcript_persons
                    else ""
                )
                results.append(
                    {
                        "chunk_id": int(row[0]),
                        "video_title": row[1],
                        "video_url": row[2],
                        "thumbnail_medium_url": row[3],
                        "chunk_index": 0,
                        "text": (
                            f"Titre: {row[1]}\nURL: {row[2]}\n"
                            f"Type: {row[6] or 'non disponible'}\n"
                            "Intervenants et fonctions:\n"
                            + ("\n".join(person_lines) or "- Aucun intervenant renseigne")
                            + transcript_line
                            + f"\nTranscript:\n{transcript or 'non disponible'}"
                        ),
                        "persons": [
                            str(item["name"])
                            for item in person_details
                            if isinstance(item, dict) and item.get("name")
                        ],
                        "person_details": person_details,
                        "transcript": transcript,
                        "video_type": row[6],
                        "bm25_score": None,
                    }
                )
            return results

        person_sql: str | None = None
        person_params: list[Any] = []
        person_rows: list[Any] = []
        # Avec des personnes identifiées, la première recherche est strictement
        # bornée aux noms résolus dans la table relationnelle. Sans personne, lookup
        # conserve son comportement générique.
        if (
            database_persons
            or database_company
            or not (query.persons or query.companies)
        ):
            clauses, person_params = build_video_lookup_conditions(
                query,
                database_persons,
                database_company,
            )
            where_sql = " AND ".join(clauses) if clauses else "TRUE"
            person_sql = f"""
            SELECT
                v.id,
                v.title,
                v.url,
                v.thumbnail_medium_url,
                {VIDEO_PERSON_DETAILS_SQL},
                {VIDEO_TRANSCRIPT_DOCUMENT_SQL} AS transcript,
                v.video_type
            FROM videos v
            WHERE {where_sql}
            ORDER BY v.published_at DESC NULLS LAST, v.id DESC
            LIMIT 10
            """
            with connect_database() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(person_sql, person_params)
                    person_rows = cursor.fetchall()

        explicit_transcript_persons = [
            person.strip()
            for person in (transcript_persons or [])
            if person.strip()
        ]
        if not query.persons:
            results = format_lookup_rows(person_rows)
            return results, {
                "mode": intent,
                "lookup_strategy": (
                    "company_title"
                    if query.companies
                    else "persons_table"
                    if query.persons
                    else "generic"
                ),
                "sql": format_sql_for_trace(person_sql),
                "params": person_params,
                "result_count": len(results),
            }

        transcript_clauses, transcript_params = build_video_lookup_conditions(query)
        transcript_document = (
            "coalesce(t.transcript_enriched, t.transcript, '')"
        )
        transcript_search_persons: list[str] = []
        for person in [
            *(database_persons or []),
            *explicit_transcript_persons,
        ]:
            if person not in transcript_search_persons:
                transcript_search_persons.append(person)
        if not transcript_search_persons:
            transcript_search_persons = [
                person.strip() for person in query.persons if person.strip()
            ]
        if transcript_search_persons:
            transcript_person_conditions = " OR ".join(
                f"concat(' ', {normalized_sql_text(transcript_document)}, ' ') LIKE "
                f"concat(chr(37), ' ', {normalized_sql_text('%s')}, ' ', chr(37))"
                for _ in transcript_search_persons
            )
            transcript_clauses.append(f"({transcript_person_conditions})")
            transcript_params.extend(transcript_search_persons)
        transcript_where = (
            " AND ".join(transcript_clauses) if transcript_clauses else "TRUE"
        )
        transcript_sql = f"""
            SELECT
                v.id,
                v.title,
                v.url,
                v.thumbnail_medium_url,
                {VIDEO_PERSON_DETAILS_SQL},
                {transcript_document} AS transcript,
                v.video_type
            FROM videos v
            JOIN transcripts t ON t.video_id = v.id
            WHERE {transcript_where}
            ORDER BY v.published_at DESC NULLS LAST, v.id DESC
            LIMIT 10
        """
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(transcript_sql, transcript_params)
                transcript_rows = cursor.fetchall()

        transcript_results = format_lookup_rows(
            transcript_rows,
            transcript_persons=transcript_search_persons,
        )
        person_results = format_lookup_rows(person_rows)
        results_by_video = {
            result["chunk_id"]: result
            for result in person_results
        }
        for result in transcript_results:
            results_by_video.setdefault(result["chunk_id"], result)
        results = list(results_by_video.values())
        input_count = len(person_results) + len(transcript_results)
        return results, {
            "mode": intent,
            "lookup_strategy": (
                "persons_table+transcript_enriched"
                if person_rows and explicit_transcript_persons
                else "transcript_fallback"
            ),
            "sql": format_sql_for_trace(transcript_sql),
            "params": transcript_params,
            "result_count": len(results),
            "deduplication": {
                "input_count": input_count,
                "persons_in_speakers_count": len(person_results),
                "persons_in_transcripts_count": len(transcript_results),
                "duplicate_count": input_count - len(results),
                "output_count": len(results),
                "key": "video_id",
            },
            "query_result_count": len(transcript_results),
            "query_results": transcript_results,
            "persons_table": {
                "sql": format_sql_for_trace(person_sql),
                "params": person_params,
                "result_count": len(person_rows),
                "query_results": person_results,
            },
        }

    if intent == "description":
        clauses, params = build_video_lookup_conditions(query)
        where_sql = " AND ".join(clauses) if clauses else "TRUE"
        sql = f"""
            SELECT
                v.id,
                v.title,
                v.url,
                v.description
            FROM videos v
            WHERE {where_sql}
            ORDER BY v.published_at DESC NULLS LAST, v.id DESC
            LIMIT 10
        """
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()

        results = [
            {
                "chunk_id": int(row[0]),
                "video_title": row[1],
                "video_url": row[2],
                "thumbnail_medium_url": None,
                "chunk_index": 0,
                "text": f"Titre: {row[1]}\nURL: {row[2]}\nDescription: {row[3] or ''}",
                "persons": [],
                "video_description": row[3],
                "bm25_score": None,
            }
            for row in rows
        ]
        return results, {
            "mode": intent,
            "sql": format_sql_for_trace(sql),
            "params": params,
            "result_count": len(results),
        }

    if intent == "transcript_verbatim":
        document_expr = "t.transcript"
    elif intent == "transcript_qa":
        document_expr = "t.transcript_enriched"
    else:
        raise RuntimeError(f"Intent direct non supporte: {intent}")

    clauses, params = build_video_lookup_conditions(query)
    terms = query.query_text_bm25.strip() or query.query_text.strip() or query.raw_question
    if not query.title_hints:
        clauses.append(
            f"""
            to_tsvector('french', coalesce(v.title, '') || ' ' || coalesce({document_expr}, ''))
            @@ websearch_to_tsquery('french', %s)
            """
        )
        params.append(terms)

    where_sql = " AND ".join(clauses) if clauses else "TRUE"
    sql = f"""
        SELECT
            v.id,
            v.title,
            v.url,
            v.thumbnail_medium_url,
            {document_expr} AS document_text,
            {VIDEO_PERSON_NAMES_SQL},
            ts_rank_cd(
                to_tsvector('french', coalesce(v.title, '') || ' ' || coalesce({document_expr}, '')),
                websearch_to_tsquery('french', %s)
            ) AS score
        FROM videos v
        JOIN transcripts t ON t.video_id = v.id
        WHERE {document_expr} IS NOT NULL
          AND {where_sql}
        ORDER BY score DESC, v.published_at DESC NULLS LAST, v.id DESC
        LIMIT 1
    """
    sql_params = [terms, *params]
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, sql_params)
            row = cursor.fetchone()

    if row is None:
        return [], {
            "mode": intent,
            "sql": format_sql_for_trace(sql),
            "params": sql_params,
            "result_count": 0,
        }

    result = {
        "chunk_id": int(row[0]),
        "video_title": row[1],
        "video_url": row[2],
        "thumbnail_medium_url": row[3],
        "chunk_index": 0,
        "text": row[4],
        "persons": row[5] or [],
        "bm25_score": float(row[6]) if row[6] is not None else None,
    }
    return [result], {
        "mode": intent,
        "sql": format_sql_for_trace(sql),
        "params": sql_params,
        "result_count": 1,
    }


def fetch_bm25_chunks(query: ExecutionPlan, candidate_chunk_ids: list[int] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terms = query.query_text_bm25.strip() or query_terms(query)
    candidate_sql, candidate_params = candidate_sql_clause(candidate_chunk_ids)
    sql = f"""
        SELECT
            c.id,
            v.title,
            v.url,
            v.thumbnail_medium_url,
            c.chunk_index,
            c.chunk_level,
            c.chunk_parent_id,
            c.content,
            {VIDEO_PERSON_NAMES_SQL},
            ts_rank_cd(
                to_tsvector('french', coalesce(c.content, '')),
                websearch_to_tsquery('french', %s)
            ) AS score
        FROM chunks c
        JOIN videos v ON v.id = c.video_id
        WHERE to_tsvector('french', coalesce(c.content, '')) @@ websearch_to_tsquery('french', %s)
          AND c.chunk_level = 'detail'
        {candidate_sql}
        ORDER BY score DESC, c.id ASC
        LIMIT %s
    """
    params = [terms, terms, *candidate_params, DEFAULT_BM25_LIMIT]
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    chunks = [
        {
            "chunk_id": int(row[0]),
            "video_title": row[1],
            "video_url": row[2],
            "thumbnail_medium_url": row[3],
            "chunk_index": row[4],
            "chunk_level": row[5],
            "chunk_parent_id": int(row[6]) if row[6] is not None else None,
            "text": row[7],
            "persons": row[8] or [],
            "bm25_score": float(row[9]) if row[9] is not None else None,
        }
        for row in rows
    ]
    return chunks, {
        "mode": "bm25",
        "query_text_bm25": terms,
        "sql": format_sql_for_trace(sql),
        "params": params,
        "result_count": len(chunks),
    }


def fetch_vector_chunks(
    query: ExecutionPlan,
    question_embedding: list[float] | None,
    candidate_chunk_ids: list[int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if question_embedding is None:
        return [], {"mode": "vector", "sql": None, "params": [], "result_count": 0, "skipped": True}

    candidate_sql, candidate_params = candidate_sql_clause(candidate_chunk_ids)
    vector_literal = "[" + ",".join(str(value) for value in question_embedding) + "]"
    sql = f"""
        SELECT
            c.id,
            v.title,
            v.url,
            v.thumbnail_medium_url,
            c.chunk_index,
            c.chunk_level,
            c.chunk_parent_id,
            c.content,
            {VIDEO_PERSON_NAMES_SQL},
            1 - (c.embedding <=> %s::vector(2000)) AS score
        FROM chunks c
        JOIN videos v ON v.id = c.video_id
        WHERE c.embedding IS NOT NULL
          AND c.embedding_model = %s
          AND c.embedding_dimensions = %s
          AND c.chunk_level = 'detail'
        {candidate_sql}
        ORDER BY c.embedding <=> %s::vector(2000) ASC
        LIMIT %s
    """
    params = [
        vector_literal,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_EMBEDDING_DIMENSIONS,
        *candidate_params,
        vector_literal,
        DEFAULT_VECTOR_LIMIT,
    ]
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    chunks = [
        {
            "chunk_id": int(row[0]),
            "video_title": row[1],
            "video_url": row[2],
            "thumbnail_medium_url": row[3],
            "chunk_index": row[4],
            "chunk_level": row[5],
            "chunk_parent_id": int(row[6]) if row[6] is not None else None,
            "text": row[7],
            "persons": row[8] or [],
            "vector_score": float(row[9]) if row[9] is not None else None,
        }
        for row in rows
    ]
    return chunks, {
        "mode": "vector",
        "sql": format_sql_for_trace(sql),
        "params": params,
        "result_count": len(chunks),
        "skipped": False,
    }


def reciprocal_rank_fusion(
    bm25_chunks: list[dict[str, Any]],
    vector_chunks: list[dict[str, Any]],
    limit: int,
    rank_constant: int = DEFAULT_FUSION_K,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: dict[int, dict[str, Any]] = {}

    def apply_ranking(chunks: list[dict[str, Any]], source_name: str) -> None:
        for rank, chunk in enumerate(chunks, start=1):
            chunk_id = int(chunk["chunk_id"])
            if chunk_id not in scored:
                scored[chunk_id] = {**chunk, "rrf_score": 0.0, "rank_sources": {}}
            scored[chunk_id][f"{source_name}_score"] = chunk.get(f"{source_name}_score")
            scored[chunk_id]["rrf_score"] += 1.0 / (rank_constant + rank)
            scored[chunk_id]["rank_sources"][source_name] = rank

    apply_ranking(bm25_chunks, "bm25")
    apply_ranking(vector_chunks, "vector")

    fused = sorted(scored.values(), key=lambda item: (-item["rrf_score"], item.get("chunk_id", 0)))[:limit]
    return fused, {
        "rank_constant": rank_constant,
        "bm25_count": len(bm25_chunks),
        "vector_count": len(vector_chunks),
        "fused_count": len(fused),
    }


def rerank_chunks(question: str, chunks: list[dict[str, Any]], limit: int, rerank_model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not chunks:
        return [], {"applied": False, "reason": "no_chunks", "input_count": 0, "output_count": 0}
    cohere_client = get_cohere_client()
    if cohere_client is None:
        raise RuntimeError("COHERE_API_KEY manquante ou SDK Cohere indisponible pour le rerank.")

    documents = [chunk["text"] for chunk in chunks]
    resolved_model = resolve_cohere_rerank_model(rerank_model)

    try:
        response = cohere_client.rerank(
            model=resolved_model,
            query=question,
            documents=documents,
            top_n=min(limit, len(documents)),
        )
    except Exception as exc:
        raise RuntimeError(f"Echec du rerank Cohere ({resolved_model}): {exc}") from exc

    results = list(getattr(response, "results", []) or [])
    ordered: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    relevance_scores: list[float | None] = []

    for item in results:
        index = getattr(item, "index", None)
        if not isinstance(index, int):
            continue
        if 0 <= index < len(chunks):
            selected_chunk = {**chunks[index]}
            relevance_score = getattr(item, "relevance_score", None)
            selected_chunk["cohere_relevance_score"] = (
                float(relevance_score) if relevance_score is not None else None
            )
            ordered.append(selected_chunk)
            selected_indices.append(index + 1)
            relevance_scores.append(relevance_score)

    if ordered:
        output = ordered[:limit]
        return output, {
            "applied": True,
            "provider": "cohere",
            "input_count": len(chunks),
            "output_count": len(output),
            "selected_indices": selected_indices,
            "selected_chunk_ids": [item["chunk_id"] for item in output],
            "relevance_scores": relevance_scores[:limit],
            "requested_model": rerank_model,
            "resolved_model": resolved_model,
        }

    raise RuntimeError(f"Le rerank Cohere ({resolved_model}) n'a renvoye aucun resultat exploitable.")


def expand_detail_context(
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detail_ids = [
        int(chunk["chunk_id"])
        for chunk in chunks
        if chunk.get("chunk_level") == "detail"
    ]
    if not detail_ids:
        return chunks, {
            "applied": False,
            "detail_count": 0,
            "expanded_count": 0,
            "sql": None,
            "params": [],
        }

    sql = """
        SELECT
            detail.id,
            section.id,
            section.chunk_index,
            section.content,
            global_chunk.id,
            global_chunk.chunk_index,
            global_chunk.content
        FROM chunks detail
        LEFT JOIN chunks section
          ON section.id = detail.chunk_parent_id
         AND section.video_id = detail.video_id
         AND section.chunk_level = 'section'
        LEFT JOIN chunks global_chunk
          ON global_chunk.id = section.chunk_parent_id
         AND global_chunk.video_id = section.video_id
         AND global_chunk.chunk_level = 'global'
        WHERE detail.id = ANY(%s)
          AND detail.chunk_level = 'detail'
    """
    params = [detail_ids]
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    context_by_detail_id = {
        int(row[0]): {
            "section_context": (
                {
                    "chunk_id": int(row[1]),
                    "chunk_index": row[2],
                    "text": row[3],
                }
                if row[1] is not None
                else None
            ),
            "global_context": (
                {
                    "chunk_id": int(row[4]),
                    "chunk_index": row[5],
                    "text": row[6],
                }
                if row[4] is not None
                else None
            ),
        }
        for row in rows
    }
    expanded = [
        {
            **chunk,
            **context_by_detail_id.get(
                int(chunk["chunk_id"]),
                {"section_context": None, "global_context": None},
            ),
        }
        for chunk in chunks
    ]
    return expanded, {
        "applied": True,
        "detail_count": len(detail_ids),
        "expanded_count": sum(
            1
            for chunk in expanded
            if chunk.get("section_context") or chunk.get("global_context")
        ),
        "sql": format_sql_for_trace(sql),
        "params": params,
    }


def retrieve_chunks(payload: RagRequest, execution_plan: ExecutionPlan) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = get_openai_client()
    final_k = execution_plan.final_k or DEFAULT_FINAL_K
    answer_model = normalize_model_name(payload.answerModel, DEFAULT_GENERATION_MODEL)
    embedding_model = normalize_model_name(payload.embeddingModel, DEFAULT_EMBEDDING_MODEL)
    if embedding_model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Le modele d'embedding doit correspondre a la base: {DEFAULT_EMBEDDING_MODEL}."
        )
    rerank_model = normalize_model_name(payload.rerankModel or "", DEFAULT_RERANK_MODEL)

    with trace_operation(
        "rag.retrieval.prefilter",
        kind="CHAIN",
        input_value=execution_plan.model_dump(),
    ) as prefilter_span:
        prefilter_candidate_ids, prefilter_debug = prefilter_candidate_chunk_ids(execution_plan)
        prefilter_output = {
            "sql": prefilter_debug.get("sql"),
            "params": prefilter_debug.get("params", []),
            "applied": prefilter_debug.get("applied", False),
            "sql_prefilters": prefilter_debug.get("sql_prefilters", False),
            "candidate_count": prefilter_debug.get("candidate_count"),
            "candidate_chunk_ids": prefilter_candidate_ids,
            "general_question_only": prefilter_debug.get("general_question_only", True),
        }
        prefilter_span.set_output(prefilter_output)
        trace_formatted_sql("rag.retrieval.prefilter", prefilter_debug)

    question_embedding: list[float] | None = None
    if client is not None and payload.useSql:
        with trace_operation(
            "rag.retrieval.embedding",
            kind="CHAIN",
            input_value={
                "model": embedding_model,
                "dimensions": DEFAULT_EMBEDDING_DIMENSIONS,
                "text": query_terms(execution_plan),
            },
        ) as embedding_span:
            embedding_response = client.embeddings.create(
                model=embedding_model,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
                input=query_terms(execution_plan),
            )
            question_embedding = embedding_response.data[0].embedding
            embedding_span.set_output(
                {"model": embedding_model, "dimensions": len(question_embedding)}
            )

    with trace_operation(
        "rag.retrieval.bm25",
        kind="CHAIN",
        input_value={
            "query": execution_plan.query_text_bm25,
            "candidate_chunk_ids": prefilter_candidate_ids,
        },
    ) as bm25_span:
        bm25_chunks, bm25_debug = fetch_bm25_chunks(execution_plan, prefilter_candidate_ids)
        bm25_span.set_output({**bm25_debug, "results": bm25_chunks})
        trace_formatted_sql("rag.retrieval.bm25", bm25_debug)

    with trace_operation(
        "rag.retrieval.vector",
        kind="CHAIN",
        input_value={
            "query": execution_plan.query_text,
            "embedding_model": embedding_model,
            "candidate_chunk_ids": prefilter_candidate_ids,
        },
    ) as vector_span:
        vector_chunks, vector_debug = fetch_vector_chunks(
            execution_plan,
            question_embedding,
            prefilter_candidate_ids,
        )
        vector_span.set_output({**vector_debug, "results": vector_chunks})
        trace_formatted_sql("rag.retrieval.vector", vector_debug)

    with trace_operation(
        "rag.retrieval.rrf",
        kind="CHAIN",
        input_value={
            "bm25_chunk_ids": [item["chunk_id"] for item in bm25_chunks],
            "vector_chunk_ids": [item["chunk_id"] for item in vector_chunks],
            "limit": DEFAULT_RRF_TOP_N,
        },
    ) as rrf_span:
        fused_chunks, fusion_debug = reciprocal_rank_fusion(
            bm25_chunks,
            vector_chunks,
            DEFAULT_RRF_TOP_N,
        )
        rrf_span.set_output({"trace": fusion_debug, "results": fused_chunks})

    if payload.useRerank:
        with trace_operation(
            "rag.retrieval.rerank",
            kind="RERANKER",
            input_value={
                "query": execution_plan.query_text,
                "model": rerank_model,
                "documents": fused_chunks,
                "limit": final_k,
            },
        ) as rerank_span:
            rerank_span.set_attribute("reranker.query", execution_plan.query_text)
            rerank_span.set_attribute("reranker.model_name", rerank_model)
            rerank_span.set_attribute("reranker.top_k", final_k)
            rerank_span.set_documents("reranker.input_documents", fused_chunks)
            final_chunks, rerank_debug = rerank_chunks(
                execution_plan.query_text,
                fused_chunks,
                final_k,
                rerank_model,
            )
            rerank_span.set_documents("reranker.output_documents", final_chunks)
            rerank_span.set_output({"trace": rerank_debug, "results": final_chunks})
    else:
        final_chunks = fused_chunks[:final_k]
        rerank_debug = {
            "applied": False,
            "reason": "disabled",
            "input_count": len(fused_chunks),
            "output_count": len(final_chunks),
            "selected_chunk_ids": [item["chunk_id"] for item in final_chunks],
        }

    with trace_operation(
        "rag.retrieval.hierarchy",
        kind="CHAIN",
        input_value={
            "strategy": "detail_then_parents",
            "detail_chunk_ids": [item["chunk_id"] for item in final_chunks],
        },
    ) as hierarchy_span:
        final_chunks, hierarchy_debug = expand_detail_context(final_chunks)
        hierarchy_span.set_output(
            {"trace": hierarchy_debug, "results": final_chunks}
        )
        trace_formatted_sql("rag.retrieval.hierarchy", hierarchy_debug)

    return final_chunks, {
        "answer_model": answer_model,
        "embedding_model": embedding_model,
        "rerank_model": rerank_model,
        "retrieval_mode": "prefilter+bm25+vector+rrf",
        "bm25_top_k": DEFAULT_BM25_LIMIT,
        "vector_top_k": DEFAULT_VECTOR_LIMIT,
        "rrf_top_n": DEFAULT_RRF_TOP_N,
        "final_k": final_k,
        "used_rerank": payload.useRerank and bool(final_chunks),
        "sql_main_source": execution_plan.sql_main_source,
        "sql_prefilters": prefilter_debug["applied"],
        "general_question_only": prefilter_debug["general_question_only"],
        "sql_query": None,
        "prefilter": prefilter_debug,
        "sql_prefilters_trace": prefilter_debug,
        "bm25": {**bm25_debug, "results": bm25_chunks},
        "vector": {**vector_debug, "results": vector_chunks},
        "rrf": {**fusion_debug, "results": fused_chunks},
        "rerank": rerank_debug,
        "hierarchy_strategy": "detail_then_parents",
        "hierarchy": hierarchy_debug,
    }
