from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from psycopg.types.json import Jsonb


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERFACE_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

DEFAULT_EXTRACTION_MODEL = "gpt-5.2"
DEFAULT_ANSWER_MODEL = "gpt-5.4-nano"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_TOP_K = 40
DEFAULT_FINAL_K = 5
MAX_TOP_K = 50
MAX_FINAL_K = 20
DEFAULT_PREFILTER_LIMIT = 1000
DEFAULT_FUSION_K = 60
DEFAULT_BM25_LIMIT = 40
DEFAULT_VECTOR_LIMIT = 40
DEFAULT_RRF_TOP_N = 30


class RagRequest(BaseModel):
    question: str = Field(min_length=1)
    conversationId: int | None = None
    apiUrl: str | None = None
    answerModel: str = DEFAULT_ANSWER_MODEL
    embeddingModel: str = DEFAULT_EMBEDDING_MODEL
    rerankModel: str | None = None
    useSql: bool = True
    useRerank: bool = True
    topK: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    finalK: int = Field(default=DEFAULT_FINAL_K, ge=1, le=MAX_FINAL_K)


class ExtractedFilters(BaseModel):
    intent: str = "rag_chunks"
    sql_sub_intent: str | None = None
    query_text: str
    query_text_bm25: str | None = None
    speakers: list[str] = Field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None


class ValidatedQuery(BaseModel):
    intent: str = "rag_chunks"
    sql_sub_intent: str | None = None
    raw_question: str
    query_text: str
    query_text_bm25: str
    speakers: list[str] = Field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    final_k: int = Field(default=DEFAULT_FINAL_K, ge=1, le=MAX_FINAL_K)


class ChunkSource(BaseModel):
    chunk_id: int
    video_title: str
    video_url: str
    chunk_index: int
    score: float | None = None
    text: str
    speakers: list[str] = Field(default_factory=list)


class RagResponse(BaseModel):
    conversation_id: int
    message_id: int
    answer: str
    sources: list[ChunkSource]
    retrieval: dict[str, Any]


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL manquant dans l'environnement.")
    return database_url


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def normalize_model_name(value: str, default: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return default
    lower = cleaned.lower()
    if lower in {"meme modele que la base", "même modèle que la base"}:
        return DEFAULT_EMBEDDING_MODEL
    if lower == "cohere rerank":
        return "cohere-rerank"
    if lower == "gpt5.4nano":
        return DEFAULT_ANSWER_MODEL
    return cleaned


def safe_json_loads(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(value[start : end + 1])
        raise


def build_extraction_prompt(question: str) -> tuple[str, str]:
    system_prompt = (
        "Tu extrais des filtres de recherche pour un backend RAG video. "
        "Retourne uniquement un JSON valide avec les cles exactes: "
        "intent, sql_sub_intent, query_text, query_text_bm25, speakers, published_after, published_before. "
        "intent doit etre l'une de ces valeurs exactes: rag_chunks, sql_request, social. "
        "sql_sub_intent peut etre null ou l'une de ces valeurs exactes: video_lookup, video_transcript, video_summary. "
        "rag_chunks = question documentaire a repondre par retrieval sur chunks. "
        "sql_request = la reponse attend surtout une requete SQL ciblee sur les tables metadata/videos/transcripts. "
        "video_lookup = l'utilisateur cherche une ou plusieurs videos correspondant a une personne ou a des filtres metadata. "
        "video_transcript = l'utilisateur demande une transcription, un verbatim ou le transcript complet d'une video. "
        "video_summary = l'utilisateur demande un resume ou une synthese de video. "
        "social = salutation, politesse, small talk ou message conversationnel qui ne demande pas d'information metier issue de la base. "
        "Tu dois toujours choisir l'intent le plus adapte a partir du message utilisateur, y compris pour les messages tres courts. "
        "Exemples: 'bonjour' => social, 'salut ca va' => social, 'merci' => social, "
        "'trouve une video avec Andy Leveque' => intent=sql_request et sql_sub_intent=video_lookup, "
        "'donne le transcript complet de la video sur Parcoursup' => intent=sql_request et sql_sub_intent=video_transcript, "
        "'resume cette video sur l'alternance' => intent=sql_request et sql_sub_intent=video_summary. "
        "Si intent n'est pas sql_request, sql_sub_intent doit etre null. "
        "query_text doit contenir la reformulation utile pour la recherche semantique/vectorielle. "
        "query_text_bm25 doit etre une version tres courte orientee mots-cles, compatible recherche plein texte BM25. "
        "query_text_bm25 ne doit contenir que des noms propres, acronymes, entites nommees, termes metier ou mots-cles concrets. "
        "Evite les verbes, les questions naturelles, les reformulations longues, les mots vides et les termes generiques comme "
        "'trouver', 'identifier', 'expliquer', 'parler', 'video', 'contenu', 'personne', 'role'. "
        "Si la question porte sur une personne nommee Andy Leveque, query_text_bm25 doit ressembler a 'Andy Leveque' et pas a une phrase. "
        "speakers est un tableau. "
        "Les dates peuvent etre null."
    )
    return system_prompt, question


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def build_social_answer(question: str) -> str:
    lower = normalize_text(question).strip()
    if any(token in lower for token in ("merci",)):
        return "Avec plaisir. Si tu veux, je peux aussi t'aider a chercher une video ou repondre a une question sur la base."
    if any(token in lower for token in ("ca va", "ca roule", "comment ca va")):
        return "Ca va bien, merci. Je suis pret a t'aider sur la base video si tu veux."
    if any(token in lower for token in ("au revoir", "a bientot", "bonne journee", "bonne soiree")):
        return "A bientot."
    return "Bonjour. Je peux t'aider a trouver une video, un transcript, un resume ou repondre a une question a partir de la base."


def normalize_extracted_intent(payload: dict[str, Any]) -> dict[str, Any]:
    intent = str(payload.get("intent") or "").strip()
    sql_sub_intent = str(payload.get("sql_sub_intent") or "").strip() or None

    legacy_sql_intents = {"video_lookup", "video_transcript", "video_summary"}
    if intent in legacy_sql_intents:
        payload["intent"] = "sql_request"
        payload["sql_sub_intent"] = intent
        return payload

    if intent != "sql_request":
        payload["sql_sub_intent"] = None
        return payload

    if sql_sub_intent not in legacy_sql_intents:
        payload["sql_sub_intent"] = "video_lookup"
    return payload


def extract_speaker_hint(question: str) -> list[str]:
    patterns = [
        r"(?:video|vidéo)\s+avec\s+([A-Za-zÀ-ÿ][\w'À-ÿ-]+(?:\s+[A-Za-zÀ-ÿ][\w'À-ÿ-]+)?)",
        r"avec\s+([A-Za-zÀ-ÿ][\w'À-ÿ-]+(?:\s+[A-Za-zÀ-ÿ][\w'À-ÿ-]+)?)",
        r"(?:où|ou)\s+parle\s+([A-Za-zÀ-ÿ][\w'À-ÿ-]+(?:\s+[A-Za-zÀ-ÿ][\w'À-ÿ-]+)?)",
        r"intervention\s+de\s+([A-Za-zÀ-ÿ][\w'À-ÿ-]+(?:\s+[A-Za-zÀ-ÿ][\w'À-ÿ-]+)?)",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            value = (match.group(1) or "").strip(" ?!.,;:")
            if not value:
                continue
            value = re.sub(r"^(un|une|le|la|les|du|de la|de l')\s+", "", value, flags=re.IGNORECASE)
            normalized = " ".join(part.capitalize() for part in value.split())
            if normalized not in found:
                found.append(normalized)
    return found


def extract_filters(question: str, client: OpenAI | None) -> tuple[ExtractedFilters, str | None, str | None, bool]:
    heuristic_speakers = extract_speaker_hint(question)

    system_prompt, user_prompt = build_extraction_prompt(question)
    raw_prompt = json.dumps({"system": system_prompt, "user": user_prompt}, ensure_ascii=False)

    if client is None:
        fallback = ExtractedFilters(
            intent="rag_chunks",
            query_text=question,
            speakers=heuristic_speakers,
        )
        return fallback, raw_prompt, json.dumps(fallback.model_dump(), ensure_ascii=False), False

    response = client.responses.create(
        model=DEFAULT_EXTRACTION_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = getattr(response, "output_text", "").strip()
    if not raw:
        fallback = ExtractedFilters(
            intent="rag_chunks",
            query_text=question,
            speakers=heuristic_speakers,
        )
        return fallback, raw_prompt, json.dumps(fallback.model_dump(), ensure_ascii=False), False

    try:
        parsed = normalize_extracted_intent(safe_json_loads(raw))
        if heuristic_speakers:
            existing = parsed.get("speakers") or []
            merged: list[str] = []
            for item in [*existing, *heuristic_speakers]:
                cleaned = str(item).strip()
                if cleaned and cleaned not in merged:
                    merged.append(cleaned)
            parsed["speakers"] = merged
        if not parsed.get("intent"):
            parsed["intent"] = "rag_chunks"
        if not parsed.get("query_text"):
            parsed["query_text"] = question
        validated = ExtractedFilters.model_validate(parsed)
        return validated, raw_prompt, raw, True
    except Exception:
        fallback = ExtractedFilters(
            intent="rag_chunks",
            query_text=question,
            speakers=heuristic_speakers,
        )
        return fallback, raw_prompt, raw, False


def build_validated_query(payload: RagRequest, extracted: ExtractedFilters) -> ValidatedQuery:
    bm25_query = (extracted.query_text_bm25 or "").strip()
    if not bm25_query:
        bm25_query = (extracted.query_text or payload.question).strip() or payload.question

    return ValidatedQuery(
        intent=extracted.intent or "rag_chunks",
        sql_sub_intent=extracted.sql_sub_intent,
        raw_question=payload.question,
        query_text=(extracted.query_text or payload.question).strip() or payload.question,
        query_text_bm25=bm25_query,
        speakers=extracted.speakers,
        published_after=extracted.published_after,
        published_before=extracted.published_before,
        top_k=DEFAULT_BM25_LIMIT,
        final_k=DEFAULT_FINAL_K,
    )


def has_structured_sql_filters(query: ValidatedQuery) -> bool:
    return any(
        [
            bool(query.speakers),
            bool(query.published_after),
            bool(query.published_before),
        ]
    )


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
                WHERE speaker_name ILIKE %s
            )
            """
        )
        params.append(f"%{cleaned}%")


def build_prefilter_conditions(query: ValidatedQuery) -> tuple[list[str], list[Any]]:
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


def prefilter_candidate_chunk_ids(query: ValidatedQuery) -> tuple[list[int] | None, dict[str, Any]]:
    clauses, params = build_prefilter_conditions(query)
    if not clauses:
        return None, {
            "applied": False,
            "general_question_only": True,
            "candidate_chunk_ids": None,
            "candidate_count": None,
            "sql": None,
            "params": [],
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
        "sql": sql,
        "params": sql_params,
    }


def query_terms(query: ValidatedQuery) -> str:
    return query.query_text.strip() or query.raw_question


def candidate_sql_clause(candidate_chunk_ids: list[int] | None) -> tuple[str, list[Any]]:
    if candidate_chunk_ids is None:
        return "", []
    if not candidate_chunk_ids:
        return " AND 1 = 0", []
    return " AND c.id = ANY(%s)", [candidate_chunk_ids]


def build_video_lookup_conditions(query: ValidatedQuery) -> tuple[list[str], list[Any]]:
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


def lookup_video_document(query: ValidatedQuery, intent: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if intent == "video_lookup":
        clauses, params = build_video_lookup_conditions(query)
        where_sql = " AND ".join(clauses) if clauses else "TRUE"
        sql = f"""
            SELECT
                v.id,
                v.title,
                v.url,
                coalesce(v.speakers, ARRAY[]::text[]),
                v.published_at
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
                "chunk_index": 0,
                "text": f"Titre: {row[1]}\nURL: {row[2]}\nSpeakers: {', '.join(row[3] or [])}",
                "speakers": row[3] or [],
                "score": None,
            }
            for row in rows
        ]
        return results, {
            "mode": intent,
            "sql": sql,
            "params": sql_params,
            "result_count": len(results),
        }

    if intent == "video_transcript":
        document_expr = "coalesce(t.transcript_timecodes, t.transcript)"
    elif intent == "video_summary":
        document_expr = "t.video_summary"
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
            "sql": sql,
            "params": sql_params,
            "result_count": 0,
        }

    result = {
        "chunk_id": int(row[0]),
        "video_title": row[1],
        "video_url": row[2],
        "chunk_index": 0,
        "text": row[3],
        "speakers": row[4] or [],
        "score": float(row[5]) if row[5] is not None else None,
    }
    return [result], {
        "mode": intent,
        "sql": sql,
        "params": sql_params,
        "result_count": 1,
    }


def fetch_bm25_chunks(query: ValidatedQuery, candidate_chunk_ids: list[int] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terms = query.query_text_bm25.strip() or query_terms(query)
    candidate_sql, candidate_params = candidate_sql_clause(candidate_chunk_ids)
    sql = f"""
        SELECT
            c.id,
            v.title,
            v.url,
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
            "chunk_index": row[3],
            "text": row[4],
            "speakers": row[5] or [],
            "score": float(row[6]) if row[6] is not None else None,
        }
        for row in rows
    ]
    return chunks, {
        "mode": "bm25",
        "query_text_bm25": terms,
        "sql": sql,
        "params": params,
        "result_count": len(chunks),
    }


def fetch_vector_chunks(
    query: ValidatedQuery,
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
            "chunk_index": row[3],
            "text": row[4],
            "speakers": row[5] or [],
            "score": float(row[6]) if row[6] is not None else None,
        }
        for row in rows
    ]
    return chunks, {
        "mode": "vector",
        "sql": sql,
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


def ensure_conversation(connection: psycopg.Connection[Any], conversation_id: int | None) -> int:
    with connection.cursor() as cursor:
        if conversation_id is not None:
            cursor.execute("SELECT id FROM chat.conversations WHERE id = %s", (conversation_id,))
            row = cursor.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Conversation introuvable: {conversation_id}")
            return int(row[0])
        cursor.execute("INSERT INTO chat.conversations DEFAULT VALUES RETURNING id")
        return int(cursor.fetchone()[0])


def store_chat_message(
    conversation_id: int | None,
    user_message: str,
    answer_message: str,
    extraction_prompt: str | None,
    extraction_response_raw: str | None,
    intent_source: str,
    pydantic_verification: bool,
    filters_json: dict[str, Any],
    sql_query: str | None,
    prefilter_trace: dict[str, Any],
    bm25_trace: dict[str, Any],
    vector_trace: dict[str, Any],
    rrf_trace: dict[str, Any],
    rerank_trace: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[int, int]:
    with psycopg.connect(get_database_url()) as connection:
        resolved_conversation_id = ensure_conversation(connection, conversation_id)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat.messages (
                    conversation_id,
                    user_message,
                    answer_message,
                    extraction_prompt,
                    extraction_response_raw,
                    intent_source,
                    pydantic_verification,
                    filters_json,
                    sql_query,
                    prefilter_trace,
                    bm25_trace,
                    vector_trace,
                    rrf_trace,
                    rerank_trace,
                    retrieved_chunks
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    resolved_conversation_id,
                    user_message,
                    answer_message,
                    extraction_prompt,
                    extraction_response_raw,
                    intent_source,
                    pydantic_verification,
                    Jsonb(filters_json),
                    sql_query,
                    Jsonb(prefilter_trace),
                    Jsonb(bm25_trace),
                    Jsonb(vector_trace),
                    Jsonb(rrf_trace),
                    Jsonb(rerank_trace),
                    Jsonb(retrieved_chunks),
                ),
            )
            message_id = int(cursor.fetchone()[0])
        connection.commit()
    return resolved_conversation_id, message_id


def rerank_chunks(client: OpenAI | None, question: str, chunks: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not chunks:
        return [], {"applied": False, "reason": "no_chunks", "input_count": 0, "output_count": 0}
    if client is None:
        output = chunks[:limit]
        return output, {
            "applied": False,
            "reason": "no_openai_client",
            "input_count": len(chunks),
            "output_count": len(output),
            "selected_chunk_ids": [item["chunk_id"] for item in output],
        }

    prompt = "\n\n".join(f"[{index}] {chunk['text']}" for index, chunk in enumerate(chunks, start=1))
    response = client.responses.create(
        model=DEFAULT_ANSWER_MODEL,
        input=[
            {"role": "system", "content": "Tu classes des extraits pour un moteur RAG. Retourne uniquement les numeros des extraits les plus pertinents, du plus pertinent au moins pertinent, limite 5, format CSV."},
            {"role": "user", "content": f"Question: {question}\n\nExtraits:\n{prompt}"},
        ],
    )
    raw = getattr(response, "output_text", "").strip()
    chosen_indices: list[int] = []
    for token in raw.replace("\n", ",").split(","):
        token = token.strip().strip("[]()")
        if token.isdigit():
            chosen_indices.append(int(token))

    ordered: list[dict[str, Any]] = []
    for chunk_number in chosen_indices:
        position = chunk_number - 1
        if 0 <= position < len(chunks):
            ordered.append(chunks[position])

    if ordered:
        output = ordered[:limit]
        return output, {
            "applied": True,
            "input_count": len(chunks),
            "output_count": len(output),
            "raw_response": raw,
            "selected_indices": chosen_indices,
            "selected_chunk_ids": [item["chunk_id"] for item in output],
        }
    output = chunks[:limit]
    return output, {
        "applied": True,
        "input_count": len(chunks),
        "output_count": len(output),
        "raw_response": raw,
        "selected_indices": chosen_indices,
        "fallback": True,
        "selected_chunk_ids": [item["chunk_id"] for item in output],
    }


def generate_answer(client: OpenAI | None, question: str, answer_model: str | None, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "Je n'ai trouve aucun chunk pertinent dans la base pour repondre a cette question."
    if client is None or not answer_model:
        joined_titles = ", ".join(f"{item['video_title']}#{item['chunk_index']}" for item in sources)
        return (
            "Reponse generee sans appel modele externe.\n\n"
            f"Question: {question}\n\n"
            f"Sources retenues: {joined_titles}\n\n"
            + "\n\n".join(source["text"] for source in sources)
        )

    context_blocks = []
    for index, source in enumerate(sources, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"Titre: {source['video_title']}",
                    f"URL: {source['video_url']}",
                    f"Chunk: {source['chunk_index']}",
                    f"Texte: {source['text']}",
                ]
            )
        )

    response = client.responses.create(
        model=answer_model,
        input=[
            {
                "role": "system",
                "content": "Tu es un assistant RAG. Reponds en francais, de facon concise, en t'appuyant uniquement sur les sources fournies. Si l'information manque, dis-le explicitement.",
            },
            {
                "role": "user",
                "content": f"Question utilisateur: {question}\n\nContexte:\n\n" + "\n\n".join(context_blocks),
            },
        ],
    )
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return answer
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable.")


def retrieve_chunks(payload: RagRequest, validated_query: ValidatedQuery) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = get_openai_client()
    answer_model = normalize_model_name(payload.answerModel, DEFAULT_ANSWER_MODEL)
    embedding_model = normalize_model_name(payload.embeddingModel, DEFAULT_EMBEDDING_MODEL)
    rerank_model = normalize_model_name(payload.rerankModel or "", "cohere-rerank")

    prefilter_candidate_ids, prefilter_debug = prefilter_candidate_chunk_ids(validated_query)
    question_embedding: list[float] | None = None
    if client is not None and payload.useSql:
        embedding_response = client.embeddings.create(model=embedding_model, input=query_terms(validated_query))
        question_embedding = embedding_response.data[0].embedding

    bm25_chunks, bm25_debug = fetch_bm25_chunks(validated_query, prefilter_candidate_ids)
    vector_chunks, vector_debug = fetch_vector_chunks(validated_query, question_embedding, prefilter_candidate_ids)
    fused_chunks, fusion_debug = reciprocal_rank_fusion(bm25_chunks, vector_chunks, DEFAULT_RRF_TOP_N)

    if payload.useRerank:
        final_chunks, rerank_debug = rerank_chunks(client, payload.question, fused_chunks, validated_query.final_k)
    else:
        final_chunks = fused_chunks[: validated_query.final_k]
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
        "sql_filter_applied": prefilter_debug["applied"],
        "general_question_only": prefilter_debug["general_question_only"],
        "sql_query": prefilter_debug["sql"],
        "prefilter": prefilter_debug,
        "bm25": {**bm25_debug, "results": bm25_chunks},
        "vector": {**vector_debug, "results": vector_chunks},
        "rrf": {**fusion_debug, "results": fused_chunks},
        "rerank": rerank_debug,
    }


def route_request(payload: RagRequest) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    client = get_openai_client()
    extracted, extraction_prompt, extracted_raw, pydantic_verification = extract_filters(payload.question, client)
    validated_query = build_validated_query(payload, extracted)

    if pydantic_verification:
        intent_source = "llm"
    else:
        intent_source = "fallback"

    base_retrieval = {
        "intent": validated_query.intent,
        "intent_source": intent_source,
        "extraction_prompt": extraction_prompt,
        "extraction_response_raw": extracted_raw,
        "pydantic_verification": pydantic_verification,
        "validated_query": validated_query.model_dump(),
    }

    if validated_query.intent == "social":
        answer = build_social_answer(payload.question)
        retrieval = {
            **base_retrieval,
            "answer_model": None,
            "embedding_model": None,
            "rerank_model": None,
            "retrieval_mode": "social",
            "bm25_top_k": 0,
            "vector_top_k": 0,
            "rrf_top_n": 0,
            "final_k": 0,
            "used_rerank": False,
            "sql_filter_applied": False,
            "general_question_only": True,
            "sql_query": None,
            "prefilter": {},
            "bm25": {},
            "vector": {},
            "rrf": {},
            "rerank": {},
            "direct_lookup": {},
        }
        return answer, [], retrieval

    if validated_query.intent == "sql_request":
        sql_sub_intent = validated_query.sql_sub_intent or "video_lookup"
        sources, direct_trace = lookup_video_document(validated_query, sql_sub_intent)
        if sql_sub_intent == "video_lookup":
            if sources:
                lines = ["Videos trouvees :"]
                for item in sources:
                    lines.append(f"- {item['video_title']} ({item['video_url']})")
                answer = "\n".join(lines)
            else:
                answer = "Je n'ai trouve aucune video correspondant a cette demande dans la base."
        else:
            answer = sources[0]["text"] if sources else "Je n'ai trouve aucun document correspondant a cette demande dans la base."

        retrieval = {
            **base_retrieval,
            "answer_model": None,
            "embedding_model": None,
            "rerank_model": None,
            "retrieval_mode": "sql_request",
            "bm25_top_k": 0,
            "vector_top_k": 0,
            "rrf_top_n": 0,
            "final_k": 1 if sql_sub_intent != "video_lookup" else len(sources),
            "used_rerank": False,
            "sql_filter_applied": has_structured_sql_filters(validated_query),
            "general_question_only": not has_structured_sql_filters(validated_query),
            "sql_query": direct_trace["sql"],
            "prefilter": direct_trace,
            "bm25": {},
            "vector": {},
            "rrf": {},
            "rerank": {},
            "direct_lookup": direct_trace,
            "sql_sub_intent": sql_sub_intent,
        }
        return answer, sources, retrieval

    sources, retrieval = retrieve_chunks(payload, validated_query)
    retrieval.update(base_retrieval)
    return "", sources, retrieval


app = FastAPI(title="RAG IONIS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "index.html")


@app.get("/styles.css")
def read_styles() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "styles.css", media_type="text/css")


@app.post("/api/rag", response_model=RagResponse)
def rag(payload: RagRequest) -> RagResponse:
    if not payload.useSql:
        raise HTTPException(status_code=400, detail="Le backend actuel attend useSql=true pour interroger la base.")

    try:
        answer, sources, retrieval = route_request(payload)
        if not answer:
            answer = generate_answer(get_openai_client(), payload.question, retrieval["answer_model"], sources)
        conversation_id, message_id = store_chat_message(
            conversation_id=payload.conversationId,
            user_message=payload.question,
            answer_message=answer,
            extraction_prompt=retrieval["extraction_prompt"],
            extraction_response_raw=retrieval["extraction_response_raw"],
            intent_source=retrieval["intent_source"],
            pydantic_verification=retrieval["pydantic_verification"],
            filters_json=retrieval["validated_query"],
            sql_query=retrieval["sql_query"],
            prefilter_trace=retrieval["prefilter"],
            bm25_trace=retrieval["bm25"],
            vector_trace=retrieval["vector"],
            rrf_trace=retrieval["rrf"],
            rerank_trace=retrieval["rerank"],
            retrieved_chunks=sources,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RagResponse(
        conversation_id=conversation_id,
        message_id=message_id,
        answer=answer,
        sources=[ChunkSource(**source) for source in sources],
        retrieval=retrieval,
    )
