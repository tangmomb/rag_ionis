from __future__ import annotations

from typing import Any

import psycopg

from interface.backend.config import (
    DEFAULT_BM25_LIMIT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FINAL_K,
    DEFAULT_FUSION_K,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PREFILTER_LIMIT,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RRF_TOP_N,
    DEFAULT_VECTOR_LIMIT,
)
from interface.backend.database import get_database_url
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


def trace_formatted_sql(span_name: str, trace: dict[str, Any]) -> None:
    formatted_sql = format_sql_pretty(trace.get("sql"))
    if formatted_sql is None:
        return
    with trace_operation(
        f"{span_name}.sql_formatted",
        kind="CHAIN",
        input_value={"params": trace.get("params", [])},
    ) as sql_span:
        sql_span.set_output_text(formatted_sql)


def append_speaker_filter_clauses(clauses: list[str], params: list[Any], speakers: list[str]) -> None:
    for speaker in speakers:
        cleaned = speaker.strip()
        if not cleaned:
            continue
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM unnest(coalesce(v.speakers, ARRAY[]::text[])) AS speaker_name
                WHERE unaccent(lower(speaker_name)) LIKE unaccent(lower(%s))
            )
            """
        )
        params.append(f"%{cleaned}%")


def build_prefilter_conditions(query: ExecutionPlan) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if query.speakers:
        append_speaker_filter_clauses(clauses, params, query.speakers)
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
        WHERE {where_sql}
        ORDER BY c.id ASC
        LIMIT %s
    """
    sql_params = [*params, DEFAULT_PREFILTER_LIMIT]
    with psycopg.connect(get_database_url()) as connection:
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


def build_video_lookup_conditions(query: ExecutionPlan) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    title_hint = query.title_hint
    if title_hint:
        clauses.append("unaccent(lower(v.title)) LIKE unaccent(lower(%s))")
        params.append(f"%{title_hint}%")
    if query.speakers:
        append_speaker_filter_clauses(clauses, params, query.speakers)
    if query.published_after:
        clauses.append("v.published_at >= %s::timestamptz")
        params.append(query.published_after)
    if query.published_before:
        clauses.append("v.published_at <= %s::timestamptz")
        params.append(query.published_before)
    return clauses, params


def lookup_video_document(query: ExecutionPlan, intent: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if intent == "video_lookup":
        clauses, params = build_video_lookup_conditions(query)
        where_sql = " AND ".join(clauses) if clauses else "TRUE"
        sql = f"""
            SELECT
                v.id,
                v.title,
                v.url,
                v.thumbnail_medium_url,
                coalesce(v.speakers, ARRAY[]::text[]),
                v.published_at,
                v.video_type
            FROM videos v
            WHERE {where_sql}
            ORDER BY v.published_at DESC NULLS LAST, v.id DESC
            LIMIT 10
        """
        sql_params = params
        with psycopg.connect(get_database_url()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, sql_params)
                rows = cursor.fetchall()

        results = [
            {
                "chunk_id": int(row[0]),
                "video_title": row[1],
                "video_url": row[2],
                "thumbnail_medium_url": row[3],
                "chunk_index": 0,
                "text": (
                    f"Titre: {row[1]}\nURL: {row[2]}\n"
                    f"Type: {row[6] or 'non disponible'}\n"
                    f"Speakers: {', '.join(row[4] or [])}"
                ),
                "speakers": row[4] or [],
                "video_type": row[6],
                "bm25_score": None,
            }
            for row in rows
        ]
        return results, {
            "mode": intent,
            "sql": format_sql_for_trace(sql),
            "params": sql_params,
            "result_count": len(results),
        }

    if intent == "video_stats":
        clauses, params = build_video_lookup_conditions(query)
        if query.video_ids:
            clauses.append("v.id = ANY(%s)")
            params.append(query.video_ids)
        clauses.append("s.id IS NOT NULL")
        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT
                v.id,
                v.title,
                v.url,
                s.view_count,
                s.like_count,
                s.comment_count,
                s.snapshot_date
            FROM videos v
            JOIN LATERAL (
                SELECT id, view_count, like_count, comment_count, snapshot_date
                FROM stats
                WHERE video_id = v.id
                ORDER BY snapshot_date DESC, data_collected_date DESC, id DESC
                LIMIT 1
            ) s ON TRUE
            WHERE {where_sql}
            ORDER BY v.published_at DESC NULLS LAST, v.id DESC
            LIMIT 10
        """
        with psycopg.connect(get_database_url()) as connection:
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
                "text": (
                    f"Titre: {row[1]}\nURL: {row[2]}\n"
                    f"Vues: {row[3] if row[3] is not None else 'non disponible'}\n"
                    f"Likes: {row[4] if row[4] is not None else 'non disponible'}\n"
                    f"Commentaires: {row[5] if row[5] is not None else 'non disponible'}\n"
                    f"Date du snapshot: {row[6]}"
                ),
                "speakers": [],
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

    if intent == "video_description":
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
        with psycopg.connect(get_database_url()) as connection:
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
                "speakers": [],
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

    if intent == "video_transcript":
        document_expr = "coalesce(t.transcript_timecodes, t.transcript)"
    else:
        raise RuntimeError(f"Intent direct non supporte: {intent}")

    clauses, params = build_video_lookup_conditions(query)
    terms = query.query_text_bm25.strip() or query.query_text.strip() or query.raw_question
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
            coalesce(v.speakers, ARRAY[]::text[]),
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
    with psycopg.connect(get_database_url()) as connection:
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
        "speakers": row[5] or [],
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
            c.content,
            c.speakers,
            ts_rank_cd(
                to_tsvector('french', coalesce(c.content, '')),
                websearch_to_tsquery('french', %s)
            ) AS score
        FROM chunks c
        JOIN videos v ON v.id = c.video_id
        WHERE to_tsvector('french', coalesce(c.content, '')) @@ websearch_to_tsquery('french', %s)
        {candidate_sql}
        ORDER BY score DESC, c.id ASC
        LIMIT %s
    """
    params = [terms, terms, *candidate_params, DEFAULT_BM25_LIMIT]
    with psycopg.connect(get_database_url()) as connection:
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
            "text": row[5],
            "speakers": row[6] or [],
            "bm25_score": float(row[7]) if row[7] is not None else None,
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
            c.content,
            c.speakers,
            1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c
        JOIN videos v ON v.id = c.video_id
        WHERE c.embedding IS NOT NULL
        {candidate_sql}
        ORDER BY c.embedding <=> %s::vector ASC
        LIMIT %s
    """
    params = [vector_literal, *candidate_params, vector_literal, DEFAULT_VECTOR_LIMIT]
    with psycopg.connect(get_database_url()) as connection:
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
            "text": row[5],
            "speakers": row[6] or [],
            "vector_score": float(row[7]) if row[7] is not None else None,
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


def retrieve_chunks(payload: RagRequest, execution_plan: ExecutionPlan) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = get_openai_client()
    answer_model = normalize_model_name(payload.answerModel, DEFAULT_GENERATION_MODEL)
    embedding_model = normalize_model_name(payload.embeddingModel, DEFAULT_EMBEDDING_MODEL)
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
            input_value={"model": embedding_model, "text": query_terms(execution_plan)},
        ) as embedding_span:
            embedding_response = client.embeddings.create(model=embedding_model, input=query_terms(execution_plan))
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
                "limit": execution_plan.final_k,
            },
        ) as rerank_span:
            rerank_span.set_attribute("reranker.query", execution_plan.query_text)
            rerank_span.set_attribute("reranker.model_name", rerank_model)
            rerank_span.set_attribute("reranker.top_k", execution_plan.final_k)
            rerank_span.set_documents("reranker.input_documents", fused_chunks)
            final_chunks, rerank_debug = rerank_chunks(
                execution_plan.query_text,
                fused_chunks,
                execution_plan.final_k,
                rerank_model,
            )
            rerank_span.set_documents("reranker.output_documents", final_chunks)
            rerank_span.set_output({"trace": rerank_debug, "results": final_chunks})
    else:
        final_chunks = fused_chunks[: execution_plan.final_k]
        rerank_debug = {
            "applied": False,
            "reason": "disabled",
            "input_count": len(fused_chunks),
            "output_count": len(final_chunks),
            "selected_chunk_ids": [item["chunk_id"] for item in final_chunks],
        }

    return final_chunks, {
        "answer_model": answer_model,
        "embedding_model": embedding_model,
        "rerank_model": rerank_model,
        "retrieval_mode": "prefilter+bm25+vector+rrf",
        "bm25_top_k": DEFAULT_BM25_LIMIT,
        "vector_top_k": DEFAULT_VECTOR_LIMIT,
        "rrf_top_n": DEFAULT_RRF_TOP_N,
        "final_k": DEFAULT_FINAL_K,
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
    }
