from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from psycopg.types.json import Jsonb


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERFACE_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

DEFAULT_PLANNER_MODEL = "gpt-5.6-luna"
DEFAULT_GENERATION_MODEL = "gpt-5.6-luna"
DEFAULT_RERANK_MODEL = "cohere-rerank"
DEFAULT_COHERE_RERANK_MODEL = "rerank-v4.0-fast"
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
SOURCE_RELEVANCE_MIN = 0.25
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()
_STARTED_AT: datetime | None = None

PlannerRoute = Literal["direct", "rag", "memory", "multi_source", "agent"]
DirectSubIntent = Literal["social"]
SqlSubIntent = Literal["video_lookup", "video_transcript", "video_summary"]
AnswerAction = Literal["answer", "clarify", "abstain"]


class RagRequest(BaseModel):
    question: str = Field(min_length=1)
    conversationId: int | None = None
    apiUrl: str | None = None
    answerModel: str = DEFAULT_GENERATION_MODEL
    embeddingModel: str = DEFAULT_EMBEDDING_MODEL
    rerankModel: str | None = None
    useSql: bool = True
    useRerank: bool = True
    topK: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    finalK: int = Field(default=DEFAULT_FINAL_K, ge=1, le=MAX_FINAL_K)


class PlannerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: PlannerRoute = "rag"
    direct_sub_intent: DirectSubIntent | None = None
    sql_sub_intent: SqlSubIntent | None = None
    query_text: str
    query_text_bm25: str | None = None
    title_hint: str | None = None
    speakers: list[str] = Field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None
    use_memory: bool = False
    use_rag: bool = False
    sql_main_source: bool = False
    plan_notes: list[str] = Field(default_factory=list, max_length=3)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: PlannerRoute = "rag"
    direct_sub_intent: DirectSubIntent | None = None
    sql_sub_intent: SqlSubIntent | None = None
    raw_question: str
    query_text: str
    query_text_bm25: str
    title_hint: str | None = None
    speakers: list[str] = Field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None
    use_memory: bool = False
    use_rag: bool = False
    sql_main_source: bool = False
    plan_notes: list[str] = Field(default_factory=list, max_length=3)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    final_k: int = Field(default=DEFAULT_FINAL_K, ge=1, le=MAX_FINAL_K)


class ChunkSource(BaseModel):
    chunk_id: int
    video_title: str
    video_url: str
    thumbnail_medium_url: str | None = None
    chunk_index: int
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    cohere_relevance_score: float | None = None
    rank_sources: dict[str, int] = Field(default_factory=dict)
    text: str
    speakers: list[str] = Field(default_factory=list)


class RagResponse(BaseModel):
    conversation_id: int
    message_id: int
    answer: str
    action: AnswerAction
    sources: list[ChunkSource]
    retrieval: dict[str, Any]


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL manquant dans l'environnement.")
    return database_url


def ensure_chat_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return

    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return

        with psycopg.connect(get_database_url()) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE SCHEMA IF NOT EXISTS chat")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat.conversations (
                        id BIGSERIAL PRIMARY KEY,
                        date TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat.messages (
                        id BIGSERIAL PRIMARY KEY,
                        conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
                        user_message TEXT NOT NULL,
                        question_reformulation_prompt TEXT,
                        question_reformulation_response_raw TEXT,
                        contextual_question TEXT,
                        planner_prompt TEXT,
                        planner_response_raw TEXT,
                        pydantic_verification BOOLEAN NOT NULL DEFAULT FALSE,
                        execution_plan_json JSONB,
                        sql_query JSONB,
                        prefilter_trace JSONB,
                        bm25_trace JSONB,
                        vector_trace JSONB,
                        rrf_trace JSONB,
                        rerank_trace JSONB,
                        source_evaluation_trace JSONB,
                        retrieved_chunks JSONB,
                        answer_prompt TEXT,
                        answer_response_raw TEXT,
                        answer_message TEXT,
                        date TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'chat'
                      AND table_name = 'messages'
                      AND column_name = 'source_evaluation_trace'
                    """
                )
                if cursor.fetchone() is None:
                    # Migration volontaire : l'historique de chat est supprime
                    # pour reconstruire la table avec les colonnes dans le bon ordre.
                    cursor.execute("DROP TABLE chat.messages")
                    cursor.execute(
                        """
                        CREATE TABLE chat.messages (
                            id BIGSERIAL PRIMARY KEY,
                            conversation_id BIGINT NOT NULL REFERENCES chat.conversations(id) ON DELETE CASCADE,
                            user_message TEXT NOT NULL,
                            question_reformulation_prompt TEXT,
                            question_reformulation_response_raw TEXT,
                            contextual_question TEXT,
                            planner_prompt TEXT,
                            planner_response_raw TEXT,
                            pydantic_verification BOOLEAN NOT NULL DEFAULT FALSE,
                            execution_plan_json JSONB,
                            sql_query JSONB,
                            prefilter_trace JSONB,
                            bm25_trace JSONB,
                            vector_trace JSONB,
                            rrf_trace JSONB,
                            rerank_trace JSONB,
                            source_evaluation_trace JSONB,
                            retrieved_chunks JSONB,
                            answer_prompt TEXT,
                            answer_response_raw TEXT,
                            answer_message TEXT,
                            date TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
                for statement in (
                    """DO $$ BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat' AND table_name = 'messages'
                              AND column_name = 'question_reformulation_trace'
                        ) AND NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat' AND table_name = 'messages'
                              AND column_name = 'question_reformulation_prompt'
                        ) THEN
                            ALTER TABLE chat.messages RENAME COLUMN question_reformulation_trace TO question_reformulation_prompt;
                        END IF;
                    END $$""",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS question_reformulation_prompt TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS question_reformulation_response_raw TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS contextual_question TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS planner_prompt TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS planner_response_raw TEXT",
                    "ALTER TABLE chat.messages DROP COLUMN IF EXISTS intent_source",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS pydantic_verification BOOLEAN NOT NULL DEFAULT FALSE",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS execution_plan_json JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS sql_query JSONB",
                    """DO $$ BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'chat' AND table_name = 'messages'
                              AND column_name = 'sql_query' AND data_type = 'text'
                        ) THEN
                            ALTER TABLE chat.messages
                            ALTER COLUMN sql_query TYPE JSONB
                            USING CASE
                                WHEN sql_query IS NULL THEN NULL
                                ELSE jsonb_build_object(
                                    'sql', sql_query,
                                    'params', '[]'::jsonb,
                                    'source', 'legacy'
                                )
                            END;
                        END IF;
                    END $$""",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS prefilter_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS bm25_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS vector_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS rrf_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS rerank_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS source_evaluation_trace JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS retrieved_chunks JSONB",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS answer_prompt TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS answer_response_raw TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS answer_message TEXT",
                    "ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS date TIMESTAMPTZ NOT NULL DEFAULT now()",
                ):
                    cursor.execute(statement)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat.messages(conversation_id)"
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_date ON chat.messages(date)")
            connection.commit()

        _SCHEMA_READY = True


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_cohere_client() -> Any | None:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    try:
        cohere = import_module("cohere")
    except ImportError:
        return None
    return cohere.ClientV2(api_key)


def normalize_model_name(value: str, default: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return default
    lower = cleaned.lower()
    if lower in {"meme modele que la base", "même modèle que la base"}:
        return DEFAULT_EMBEDDING_MODEL
    if lower in {"cohere rerank", "cohere-rerank"}:
        return "cohere-rerank"
    if lower in {"5.6 luna", "gpt 5.6 luna", "gpt-5.6 luna"}:
        return DEFAULT_GENERATION_MODEL
    # Compatibilite avec d'anciens clients : le modele de generation officiel
    # reste toujours celui defini par DEFAULT_GENERATION_MODEL.
    if lower in {"gpt5.4nano", "gpt-5.4-nano"}:
        return DEFAULT_GENERATION_MODEL
    return cleaned


def resolve_cohere_rerank_model(value: str) -> str:
    normalized = normalize_model_name(value, DEFAULT_RERANK_MODEL)
    if normalized == "cohere-rerank":
        return DEFAULT_COHERE_RERANK_MODEL
    return normalized


def safe_json_loads(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(value[start : end + 1])
        raise


def format_sql_for_trace(sql: str | None) -> str | None:
    if not sql:
        return None
    return " ".join(sql.split())


def build_planner_prompt(question: str) -> tuple[str, str]:
    system_prompt = (
        "Tu es le planner d'un assistant conversationnel video. "
        "Ton role est uniquement de comprendre la demande utilisateur et de produire un plan d'execution strict en JSON. "
        "Tu ne dois jamais pretendre acceder aux donnees, ni repondre a la question utilisateur. "
        "Tu ne dois jamais mentionner ni utiliser de details techniques d'implementation. "
        "Retourne uniquement un JSON valide avec les cles exactes: "
        "route, direct_sub_intent, sql_sub_intent, query_text, query_text_bm25, title_hint, speakers, published_after, published_before, use_memory, use_rag, sql_main_source, plan_notes. "
        "route doit etre l'une de ces valeurs exactes: direct, rag, memory, multi_source, agent. "
        "direct = reponse sans recherche. "
        "rag = recherche documentaire dans les contenus video et documents associes. "
        "Les demandes structurees sur les metadonnees, speakers ou transcripts utilisent aussi rag, avec sql_main_source=true et le sous-intent SQL adapte. "
        "memory = recherche dans l'historique conversationnel. "
        "multi_source = combinaison de plusieurs sources. "
        "agent = uniquement pour les demandes necessitant plusieurs etapes ou un raisonnement complexe. "
        "direct_sub_intent peut etre null ou social. "
        "sql_sub_intent peut etre null ou l'une de ces valeurs exactes: video_lookup, video_transcript, video_summary. "
        "Si route=direct et que le message est une salutation, politesse, small talk ou message purement conversationnel, mets direct_sub_intent=social. "
        "Si sql_main_source=false, sql_sub_intent doit etre null. "
        "Si route=memory, use_memory=true et use_rag=false et sql_main_source=false. "
        "Si route=rag, use_rag=true et use_memory=false et sql_main_source=false. "
        "Pour une video, un transcript, des speakers ou des metadonnees, route=rag, sql_main_source=true et sql_sub_intent=video_lookup ou video_transcript. "
        "Si route=multi_source, active au moins deux booleens parmi use_memory, use_rag, sql_main_source. "
        "Si route=agent, tu peux activer plusieurs booleens si necessaire. "
        "Exemples: 'bonjour' => route=direct et direct_sub_intent=social. "
        "'qu'est-ce qui est dit sur Parcoursup ?' => route=rag. "
        "'trouve une video avec Andy Leveque' => route=rag, sql_main_source=true et sql_sub_intent=video_lookup. "
        "'donne le transcript complet de la video sur Parcoursup' => route=rag, sql_main_source=true et sql_sub_intent=video_transcript. "
        "'donne le sommaire de cette video sur l'alternance' => route=rag, sql_main_source=true et sql_sub_intent=video_summary. "
        "Ne choisis video_summary que si le mot exact 'sommaire' est present dans la question utilisateur. "
        "Si l'utilisateur demande un resume, une synthese, ce que dit quelqu'un dans une video, ou les points principaux, ne choisis pas video_summary par defaut. "
        "'que t'ai-je demande juste avant ?' => route=memory. "
        "'compare ce que dit la base et ce qu'on s'est deja dit' => route=multi_source. "
        "query_text doit contenir la reformulation utile pour la recherche semantique/vectorielle. "
        "query_text_bm25 doit etre une version tres courte orientee mots-cles. "
        "title_hint contient le titre ou fragment de titre explicitement fourni par l'utilisateur, sinon null. "
        "query_text_bm25 ne doit contenir que des noms propres, acronymes, entites nommees, termes metier ou mots-cles concrets. "
        "Pour direct ou memory, query_text peut etre proche de la question brute. "
        "Pour rag ou sql, evite les verbes, les questions naturelles, les reformulations longues, les mots vides et les termes generiques comme "
        "'trouver', 'identifier', 'expliquer', 'parler', 'video', 'contenu', 'personne', 'role'. "
        "Si la question porte sur une personne nommee Andy Leveque, query_text_bm25 doit ressembler a 'Andy Leveque' et pas a une phrase. "
        "speakers est un tableau. "
        "Les dates peuvent etre null. "
        "plan_notes est une liste courte de notes d'execution, 0 a 3 elements maximum."
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


def normalize_planner_output(payload: dict[str, Any]) -> dict[str, Any]:
    route = str(payload.get("route") or "").strip()
    direct_sub_intent = str(payload.get("direct_sub_intent") or "").strip() or None
    sql_sub_intent = str(payload.get("sql_sub_intent") or "").strip() or None

    if "sql_main_source" not in payload and "use_sql" in payload:
        payload["sql_main_source"] = bool(payload.get("use_sql"))
    payload.pop("use_sql", None)

    legacy_sql_intents = {"video_lookup", "video_transcript", "video_summary"}
    legacy_routes = {"rag_chunks": "rag", "sql_request": "rag", "social": "direct"}

    if route in legacy_sql_intents:
        payload["route"] = "rag"
        payload["sql_sub_intent"] = route
        payload["sql_main_source"] = True
        payload["use_rag"] = True
        payload["use_memory"] = False
        return payload

    if route in legacy_routes:
        payload["route"] = legacy_routes[route]
        route = payload["route"]

    if route == "sql":
        payload["route"] = "rag"
        payload["sql_main_source"] = True
        payload["use_rag"] = True
        payload["use_memory"] = False
        route = "rag"

    if route == "direct" and direct_sub_intent is None:
        payload["direct_sub_intent"] = "social"

    if route not in {"multi_source", "agent"} and not payload.get("sql_main_source"):
        payload["sql_sub_intent"] = None
    elif sql_sub_intent not in legacy_sql_intents:
        payload["sql_sub_intent"] = "video_lookup"

    if route == "memory":
        payload["use_memory"] = True
        payload["use_rag"] = False
        payload["sql_main_source"] = False
    elif route == "rag":
        payload["use_memory"] = False
        payload["use_rag"] = True
        payload["sql_main_source"] = bool(payload.get("sql_main_source"))
    elif route == "direct":
        payload["use_memory"] = False
        payload["use_rag"] = False
        payload["sql_main_source"] = False

    if route not in {"direct", "rag", "memory", "multi_source", "agent"}:
        payload["route"] = "rag"
        payload["use_memory"] = False
        payload["use_rag"] = True
        payload["sql_main_source"] = False
        return payload
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


def run_planner(question: str, client: OpenAI | None) -> tuple[PlannerPlan, str | None, str | None, bool]:
    heuristic_speakers = extract_speaker_hint(question)

    system_prompt, user_prompt = build_planner_prompt(question)
    planner_input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw_prompt = json.dumps(
        {"model": DEFAULT_PLANNER_MODEL, "input": planner_input},
        ensure_ascii=False,
    )

    if client is None:
        fallback = PlannerPlan(
            route="rag",
            query_text=question,
            speakers=heuristic_speakers,
            use_rag=True,
        )
        return fallback, raw_prompt, json.dumps(fallback.model_dump(), ensure_ascii=False), False

    response = client.responses.create(model=DEFAULT_PLANNER_MODEL, input=planner_input)
    raw_response = serialize_openai_response(response)
    raw = getattr(response, "output_text", "").strip()
    if not raw:
        fallback = PlannerPlan(
            route="rag",
            query_text=question,
            speakers=heuristic_speakers,
            use_rag=True,
        )
        return fallback, raw_prompt, raw_response, False

    try:
        parsed = normalize_planner_output(safe_json_loads(raw))
        if heuristic_speakers:
            existing = parsed.get("speakers") or []
            merged: list[str] = []
            for item in [*existing, *heuristic_speakers]:
                cleaned = str(item).strip()
                if cleaned and cleaned not in merged:
                    merged.append(cleaned)
            parsed["speakers"] = merged
        if not parsed.get("route"):
            parsed["route"] = "rag"
        if not parsed.get("query_text"):
            parsed["query_text"] = question
        validated = PlannerPlan.model_validate(parsed)
        return validated, raw_prompt, raw_response, True
    except Exception:
        fallback = PlannerPlan(
            route="rag",
            query_text=question,
            speakers=heuristic_speakers,
            use_rag=True,
        )
        return fallback, raw_prompt, raw_response, False


def build_execution_plan(payload: RagRequest, planner_plan: PlannerPlan) -> ExecutionPlan:
    bm25_query = (planner_plan.query_text_bm25 or "").strip()
    if not bm25_query:
        bm25_query = (planner_plan.query_text or payload.question).strip() or payload.question

    return ExecutionPlan(
        route=planner_plan.route or "rag",
        direct_sub_intent=planner_plan.direct_sub_intent,
        sql_sub_intent=planner_plan.sql_sub_intent,
        raw_question=payload.question,
        query_text=(planner_plan.query_text or payload.question).strip() or payload.question,
        query_text_bm25=bm25_query,
        title_hint=planner_plan.title_hint,
        speakers=planner_plan.speakers,
        published_after=planner_plan.published_after,
        published_before=planner_plan.published_before,
        use_memory=planner_plan.use_memory,
        use_rag=planner_plan.use_rag,
        sql_main_source=planner_plan.sql_main_source,
        plan_notes=planner_plan.plan_notes,
        top_k=DEFAULT_BM25_LIMIT,
        final_k=DEFAULT_FINAL_K,
    )


def has_explicit_structured_sql_request(question: str) -> bool:
    normalized = "".join(
        char for char in unicodedata.normalize("NFD", question.lower()) if unicodedata.category(char) != "Mn"
    )
    return bool(
        re.search(
            r"\b(?:url|lien|titre|date de publication|publie|publiee|publiees|"
            r"transcript|transcription|verbatim|timecodes?|sous-titres?|intervenant(?:e|s)?|speaker(?:s)?)\b",
            normalized,
        )
        or re.search(r"\b(?:quelle?|quelles?)\s+(?:video|videos|url|lien|titre|date)\b", normalized)
        or re.search(r"\b(?:trouve|trouver|cherche|chercher|liste|lister)\b.*\b(?:video|videos|transcript|transcription)\b", normalized)
    )


def apply_deterministic_sql_policy(question: str, planner_plan: PlannerPlan) -> None:
    """Empêche le planner de basculer arbitrairement la source SQL principale."""
    if planner_plan.route in {"direct", "memory"}:
        planner_plan.sql_main_source = False
        planner_plan.sql_sub_intent = None
        return

    if not has_explicit_structured_sql_request(question):
        # Le planner peut conserver SQL pour une question video complexe.
        # Si la recherche structuree echoue, orchestrate_request tentera le RAG.
        return

    planner_plan.sql_main_source = True
    normalized = question.lower()
    if any(term in normalized for term in ("transcript", "transcription", "verbatim", "timecode", "sous-titre")):
        planner_plan.sql_sub_intent = "video_transcript"
    else:
        planner_plan.sql_sub_intent = planner_plan.sql_sub_intent or "video_lookup"


def has_structured_sql_filters(query: ExecutionPlan) -> bool:
    return any(
        [
            bool(query.speakers),
            bool(query.published_after),
            bool(query.published_before),
        ]
    )


def resolve_speaker_filters(question: str, planner_plan: PlannerPlan) -> list[str]:
    """Ne conserve que les noms réellement présents dans videos.speakers."""
    candidates = [str(value).strip() for value in planner_plan.speakers if str(value).strip()]

    # Pour les questions d'identité, le nom peut être dans query_text_bm25
    # sans avoir été classé comme speaker par le planner.
    identity_match = re.search(
        r"\b(?:qui est|qui était|que fait|parle de|à propos de)\s+([^?!.;,]+)",
        question,
        flags=re.IGNORECASE,
    )
    if not candidates and identity_match:
        candidate = identity_match.group(1).strip()
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return []

    resolved: list[str] = []
    with psycopg.connect(get_database_url()) as connection:
        with connection.cursor() as cursor:
            for candidate in candidates:
                cursor.execute(
                    """
                    SELECT DISTINCT speaker_name
                    FROM videos v
                    CROSS JOIN LATERAL unnest(coalesce(v.speakers, ARRAY[]::text[])) AS speaker_name
                    WHERE unaccent(lower(speaker_name)) LIKE unaccent(lower(%s))
                    ORDER BY speaker_name
                    LIMIT 5
                    """,
                    (f"%{candidate}%",),
                )
                for (speaker_name,) in cursor.fetchall():
                    cleaned = str(speaker_name).strip()
                    if cleaned and cleaned not in resolved:
                        resolved.append(cleaned)
    return resolved


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


def extract_video_title_hint(question: str) -> str | None:
    """Extrait un titre explicitement fourni par l'utilisateur pour le SQL vidéo."""
    quoted_match = re.search(r"[\"«“]([^\"»”]+)[\"»”]\s+dans\s+le\s+titre", question, flags=re.IGNORECASE)
    if quoted_match:
        return quoted_match.group(1).strip() or None

    plain_match = re.search(r"(?:avec|intitulee?|intitulee?\s+la\s+video)\s+(.+?)\s+dans\s+le\s+titre", question, flags=re.IGNORECASE)
    if plain_match:
        return plain_match.group(1).strip(" \"«“»”'\t") or None

    return None


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
                "thumbnail_medium_url": row[3],
                "chunk_index": 0,
                "text": f"Titre: {row[1]}\nURL: {row[2]}\nSpeakers: {', '.join(row[3] or [])}",
                "speakers": row[4] or [],
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


def fetch_conversation_memory(conversation_id: int | None, limit: int = 8) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if conversation_id is None:
        return [], {"applied": False, "reason": "no_conversation_id", "message_count": 0}

    ensure_chat_schema()

    sql = """
        SELECT user_message, answer_message, retrieved_chunks
        FROM chat.messages
        WHERE conversation_id = %s
        ORDER BY id DESC
        LIMIT %s
    """
    with psycopg.connect(get_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (conversation_id, limit))
            rows = cursor.fetchall()

    items: list[dict[str, str]] = []
    for row in reversed(rows):
        user_message = str(row[0] or "").strip()
        answer_message = str(row[1] or "").strip()
        if user_message:
            items.append({"role": "user", "text": user_message})
        if answer_message:
            source_context = ""
            try:
                retrieved_chunks = row[2]
                if isinstance(retrieved_chunks, str):
                    retrieved_chunks = json.loads(retrieved_chunks)
                if isinstance(retrieved_chunks, list):
                    source_labels = []
                    for source in retrieved_chunks[:5]:
                        if not isinstance(source, dict):
                            continue
                        title = str(source.get("video_title") or "").strip()
                        speakers = source.get("speakers") or []
                        if title:
                            label = title
                            if speakers:
                                label += f" (intervenants : {', '.join(map(str, speakers))})"
                            source_labels.append(label)
                    if source_labels:
                        source_context = "\nSources de la réponse précédente : " + " ; ".join(source_labels)
            except (TypeError, ValueError, json.JSONDecodeError):
                source_context = ""
            items.append({"role": "assistant", "text": answer_message + source_context})

    return items, {
        "applied": True,
        "reason": None,
        "message_count": len(items),
        "sql": sql,
        "params": [conversation_id, limit],
    }


def build_question_reformulation_prompt(
    question: str,
    memory_items: list[dict[str, str]],
) -> tuple[str, str]:
    history = "\n".join(
        f"{item['role']}: {item['text']}" for item in memory_items
    )
    system_prompt = (
        "Tu reformules une question utilisateur pour la rendre autonome avant une recherche documentaire. "
        "Utilise uniquement l'historique fourni pour résoudre les références implicites comme "
        "'cette personne', 'ce sujet', 'et pour Emric ?'. "
        "Conserve l'intention, les contraintes et les noms propres de la question actuelle. "
        "Si la question est déjà autonome, renvoie-la sans changement. "
        "Ne réponds pas à la question, n'ajoute aucune explication et renvoie uniquement la question reformulée en français."
    )
    user_prompt = f"Question actuelle : {question}\n\nHistorique récent :\n{history}"
    system_prompt = (
        "Tu analyses puis reformules une question utilisateur avant une recherche documentaire. "
        "Utilise l'historique uniquement pour resoudre les references implicites. "
        "Si la question introduit un nouveau sujet, une nouvelle personne ou une nouvelle intention, "
        "considere-la comme autonome et ne reutilise pas l'historique. "
        "Si le dernier message de l'assistant demandait d'identifier une video et que la question actuelle fournit un titre, un mot-cle ou un nom de video, traite cette question comme une precision de la demande precedente. "
        "Dans ce cas, conserve l'intention precedente : ne transforme pas une question de contenu comme 'qu'apprend-on dans cette video ?' en question de recherche de titre comme 'quelle est la video ?'. "
        "Exemple : si l'historique demande 'qu'apprend-on dans la video sur la recherche de stage ?' et que l'utilisateur precise 'la video avec recherche de stage dans le titre', reformule en 'qu'apprend-on dans la video dont le titre contient recherche de stage ?'. "
        "Retourne uniquement un JSON valide avec exactement deux champs : "
        "follow_up (booleen) et reformulated_question (chaine en francais). "
        "follow_up=true uniquement si la question depend du contexte precedent. "
        "Si follow_up=false, reformule quand meme la question en francais correct, "
        "mais sans reutiliser le contenu de l'historique."
    )
    return system_prompt, user_prompt


def repair_video_clarification_follow_up(
    question: str,
    memory_items: list[dict[str, str]],
) -> str | None:
    """Preserve l'intention initiale quand l'utilisateur identifie une vidéo demandée."""
    normalized_question = normalize_text(question).strip()
    if not re.match(r"^(?:la|le|cette|ce|une|un)\s+video\b", normalized_question):
        return None

    last_assistant = next(
        (item["text"] for item in reversed(memory_items) if item.get("role") == "assistant"),
        "",
    )
    if "de quelle video" not in normalize_text(last_assistant):
        return None

    previous_user = next(
        (item["text"] for item in reversed(memory_items) if item.get("role") == "user"),
        "",
    ).strip()
    if not previous_user:
        return None

    return (
        f"{previous_user}\n"
        f"Reference video precisée par l'utilisateur : {question.strip()}"
    )


def serialize_openai_response(response: Any) -> str:
    """Serialize the complete SDK response while keeping it valid JSON."""
    to_json = getattr(response, "to_json", None)
    if callable(to_json):
        return str(to_json())

    model_dump_json = getattr(response, "model_dump_json", None)
    if callable(model_dump_json):
        return str(model_dump_json())

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(mode="json"), ensure_ascii=False)

    return json.dumps(response, default=str, ensure_ascii=False)


def reformulate_question(
    question: str,
    conversation_id: int | None,
    client: OpenAI | None,
) -> tuple[str, dict[str, Any]]:
    """Rend une relance autonome avant le planner, sans modifier le message stocké."""
    memory_items, memory_trace = fetch_conversation_memory(conversation_id)
    trace: dict[str, Any] = {
        "applied": False,
        "original_question": question,
        "reformulated_question": question,
        "memory_message_count": len(memory_items),
        "memory": memory_trace,
    }
    if client is None:
        trace["reason"] = "no_openai_client"
        return question, trace

    system_prompt, user_prompt = build_question_reformulation_prompt(question, memory_items)
    trace["prompt"] = json.dumps(
        {"model": DEFAULT_PLANNER_MODEL, "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]},
        ensure_ascii=False,
    )
    try:
        response = client.responses.create(
            model=DEFAULT_PLANNER_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        trace["response_raw"] = serialize_openai_response(response)
        raw_output = (getattr(response, "output_text", "") or "").strip()
        if not raw_output:
            trace["reason"] = "empty_response"
            return question, trace
        # Évite qu'une réponse accidentellement multi-ligne devienne une nouvelle consigne.
        try:
            parsed = safe_json_loads(raw_output)
            follow_up = bool(parsed.get("follow_up", False))
            reformulated = str(parsed.get("reformulated_question") or question).strip()
        except (TypeError, ValueError, json.JSONDecodeError):
            trace["reason"] = "invalid_json_response"
            return question, trace

        trace["follow_up"] = follow_up
        repaired = repair_video_clarification_follow_up(question, memory_items)
        if repaired:
            reformulated = repaired
            follow_up = True
            trace["reason"] = "video_followup_intent_preserved"
        trace["applied"] = reformulated != question.strip()
        trace["reformulated_question"] = reformulated
        return reformulated or question, trace
    except Exception as exc:  # La recherche doit rester disponible si la reformulation échoue.
        trace["reason"] = "reformulation_error"
        trace["error"] = str(exc)
        return question, trace


def sql_trace_for_storage(retrieval: dict[str, Any]) -> dict[str, Any] | None:
    candidate = retrieval.get("sql_query")
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str) or not candidate.strip():
        return None

    if retrieval.get("direct_lookup"):
        source = "structured_sql"
        params = retrieval["direct_lookup"].get("params", [])
    elif retrieval.get("prefilter"):
        source = "prefilter"
        params = retrieval["prefilter"].get("params", [])
    elif retrieval.get("memory"):
        source = "memory"
        params = retrieval["memory"].get("params", [])
    else:
        source = "unknown"
        params = []
    return {"sql": candidate, "params": params, "source": source}


def store_chat_message(
    conversation_id: int | None,
    user_message: str,
    answer_message: str,
    question_reformulation_response_raw: str | None,
    contextual_question: str,
    question_reformulation_prompt: str | None,
    planner_prompt: str | None,
    planner_response_raw: str | None,
    pydantic_verification: bool,
    execution_plan_json: dict[str, Any],
    sql_query: dict[str, Any] | None,
    prefilter_trace: dict[str, Any],
    bm25_trace: dict[str, Any],
    vector_trace: dict[str, Any],
    rrf_trace: dict[str, Any],
    rerank_trace: dict[str, Any],
    source_evaluation_trace: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    answer_prompt: str | None,
    answer_response_raw: str | None,
) -> tuple[int, int]:
    ensure_chat_schema()

    with psycopg.connect(get_database_url()) as connection:
        resolved_conversation_id = ensure_conversation(connection, conversation_id)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat.messages (
                    conversation_id,
                    user_message,
                    question_reformulation_prompt,
                    question_reformulation_response_raw,
                    contextual_question,
                    planner_prompt,
                    planner_response_raw,
                    pydantic_verification,
                    execution_plan_json,
                    sql_query,
                    prefilter_trace,
                    bm25_trace,
                    vector_trace,
                    rrf_trace,
                    rerank_trace,
                    source_evaluation_trace,
                    retrieved_chunks,
                    answer_prompt,
                    answer_response_raw,
                    answer_message
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    resolved_conversation_id,
                    user_message,
                    question_reformulation_prompt,
                    question_reformulation_response_raw,
                    contextual_question,
                    planner_prompt,
                    planner_response_raw,
                    pydantic_verification,
                    Jsonb(execution_plan_json),
                    Jsonb(sql_query) if sql_query is not None else None,
                    Jsonb(prefilter_trace),
                    Jsonb(bm25_trace),
                    Jsonb(vector_trace),
                    Jsonb(rrf_trace),
                    Jsonb(rerank_trace),
                    Jsonb(source_evaluation_trace),
                    Jsonb(retrieved_chunks),
                    answer_prompt,
                    answer_response_raw,
                    answer_message,
                ),
            )
            message_id = int(cursor.fetchone()[0])
        connection.commit()
    return resolved_conversation_id, message_id


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


FINAL_ANSWER_STYLE = (
    "Réponds directement à la question avec les éléments disponibles. "
    "N'introduis pas ta réponse par une formule comme « d'après les sources » "
    "ou « selon les documents ». "
    "Ne termine pas par une phrase indiquant qu'il manque des informations, "
    "que tu n'en as pas d'autres ou que tu ne peux pas aller plus loin. "
    "Ne parle pas de tes limites ni de la recherche effectuée."
)
SOURCE_MARKER_INSTRUCTION = (
    "Pour chaque information importante provenant d'un chunk, ajoute son marqueur "
    "[S1], [S2], etc. correspondant au numéro du chunk dans le contexte. "
    "N'utilise que les marqueurs des chunks réellement utilisés."
)
ANSWER_ACTION_INSTRUCTION = (
    "Retourne uniquement un objet JSON valide avec exactement deux cles : "
    "answer et action. action doit valoir exactement answer, clarify ou abstain. "
    "Utilise answer si les sources permettent de repondre. "
    "Utilise clarify si la question n'est pas assez precise pour savoir quelle information ou quelle video est demandee ; dans ce cas, answer doit etre une seule question de precision adressee a l'utilisateur. "
    "Utilise abstain si la question est claire mais que les sources ne contiennent pas l'information necessaire. "
    "Le champ answer contient uniquement le message final a afficher a l'utilisateur."
)


def record_answer_trace(
    trace: dict[str, str] | None,
    model: str,
    input_messages: list[dict[str, str]],
    response: Any,
) -> None:
    if trace is None:
        return
    trace["prompt"] = json.dumps(
        {"model": model, "input": input_messages},
        ensure_ascii=False,
    )
    trace["response_raw"] = serialize_openai_response(response)


def parse_answer_output(raw_answer: str, trace: dict[str, str] | None = None) -> str:
    """Valide l'enveloppe JSON du modele et conserve l'action choisie."""
    fallback_action: AnswerAction = "answer"
    try:
        payload = safe_json_loads(raw_answer)
    except (TypeError, ValueError, json.JSONDecodeError):
        if trace is not None:
            trace["action"] = fallback_action
        return raw_answer

    if not isinstance(payload, dict):
        if trace is not None:
            trace["action"] = fallback_action
        return raw_answer

    action = payload.get("action")
    if action not in {"answer", "clarify", "abstain"}:
        action = fallback_action
    answer = str(payload.get("answer") or "").strip()
    if trace is not None:
        trace["action"] = action
    return answer or raw_answer


def select_answer_sources(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Retire les marqueurs de citation et conserve les sources utilisées par la réponse."""
    marker_indexes = {
        int(value)
        for value in re.findall(r"\[S(\d+)\]", answer, flags=re.IGNORECASE)
        if 1 <= int(value) <= len(sources)
    }
    cleaned_answer = re.sub(r"\s*\[S\d+\]", "", answer, flags=re.IGNORECASE).strip()
    selected_sources = [
        source for index, source in enumerate(sources, start=1) if index in marker_indexes
    ]
    return cleaned_answer, selected_sources


def evaluate_source_sufficiency(
    question: str,
    sources: list[dict[str, Any]],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    """Calcule une confiance technique avant de demander une réponse au modèle."""
    source_scores = [
        {
            "chunk_id": source.get("chunk_id"),
            "video_title": source.get("video_title"),
            "cohere_relevance_score": source.get("cohere_relevance_score"),
            "rrf_score": source.get("rrf_score"),
            "bm25_score": source.get("bm25_score"),
            "vector_score": source.get("vector_score"),
        }
        for source in sources
    ]
    cohere_scores = [
        float(source["cohere_relevance_score"])
        for source in source_scores
        if source.get("cohere_relevance_score") is not None
    ]
    normalized_question = normalize_text(question)
    has_unresolved_video_reference = bool(
        re.search(r"\b(?:la|le|cette|ce|une|un)\s+video\b", normalized_question)
        or re.search(r"\bvideo\b.*\b(?:sur|de|a propos de)\b", normalized_question)
    )

    if not sources:
        action_hint: AnswerAction = "clarify" if has_unresolved_video_reference else "abstain"
        reason = "ambiguous_video_reference" if action_hint == "clarify" else "no_sources"
    elif cohere_scores and max(cohere_scores) < SOURCE_RELEVANCE_MIN:
        action_hint = "abstain"
        reason = "low_rerank_relevance"
    else:
        action_hint = "answer"
        reason = "sources_available"

    return {
        "action_hint": action_hint,
        "reason": reason,
        "question": question,
        "source_count": len(sources),
        "top_cohere_relevance_score": max(cohere_scores) if cohere_scores else None,
        "source_scores": source_scores,
        "retrieval_mode": retrieval.get("retrieval_mode"),
        "clarification_message": (
            "Peux-tu préciser le titre exact de la vidéo ou le sujet dont tu parles ?"
            if action_hint == "clarify"
            else None
        ),
    }


def generate_answer(
    client: OpenAI | None,
    question: str,
    answer_model: str | None,
    sources: list[dict[str, Any]],
    trace: dict[str, str] | None = None,
) -> str:
    if not sources:
        if trace is not None:
            trace["action"] = "abstain"
        return "Je n'ai trouve aucun chunk pertinent dans la base pour repondre a cette question."
    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer"
        return "\n\n".join(
            f"[S{index}] {source['text']}"
            for index, source in enumerate(sources, start=1)
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

    input_messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant RAG. Réponds en français, de façon concise, "
                    "en t'appuyant uniquement sur les sources fournies. "
                     + SOURCE_MARKER_INSTRUCTION + " "
                     + ANSWER_ACTION_INSTRUCTION + " "
                     + FINAL_ANSWER_STYLE
                ),
            },
            {
                "role": "user",
                "content": f"Question utilisateur: {question}\n\nContexte:\n\n" + "\n\n".join(context_blocks),
            },
        ]
    response = client.responses.create(model=answer_model, input=input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable.")


def generate_memory_answer(
    client: OpenAI | None,
    question: str,
    answer_model: str | None,
    memory_items: list[dict[str, str]],
    trace: dict[str, str] | None = None,
) -> str:
    if not memory_items:
        if trace is not None:
            trace["action"] = "abstain"
        return "Je n'ai pas trouve d'historique de conversation exploitable pour repondre a cette demande."

    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer"
        history = "\n".join(f"{item['role']}: {item['text']}" for item in memory_items)
        return history

    history = "\n".join(f"{item['role']}: {item['text']}" for item in memory_items)
    input_messages = [
            {
                "role": "system",
                "content": (
                    "Tu réponds uniquement à partir de l'historique de conversation fourni. "
                    + ANSWER_ACTION_INSTRUCTION + " "
                    + FINAL_ANSWER_STYLE
                ),
            },
            {
                "role": "user",
                "content": f"Question actuelle: {question}\n\nHistorique:\n{history}",
            },
        ]
    response = client.responses.create(model=answer_model, input=input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable pour la route memory.")


def generate_multi_source_answer(
    client: OpenAI | None,
    question: str,
    answer_model: str | None,
    route_name: str,
    memory_items: list[dict[str, str]],
    sources: list[dict[str, Any]],
    trace: dict[str, str] | None = None,
) -> str:
    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer" if (memory_items or sources) else "abstain"
        memory_text = "\n".join(item["text"] for item in memory_items)
        source_text = "\n\n".join(
            f"[S{index}] {source['text']}"
            for index, source in enumerate(sources, start=1)
        )
        return "\n\n".join(part for part in (memory_text, source_text) if part)

    memory_block = "\n".join(f"{item['role']}: {item['text']}" for item in memory_items) or "Aucun historique exploitable."
    source_blocks = []
    for index, source in enumerate(sources, start=1):
        source_blocks.append(
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
    source_block = "\n\n".join(source_blocks) or "Aucune source documentaire exploitable."

    input_messages = [
            {
                "role": "system",
                "content": (
                    "Tu synthétises plusieurs sources pour répondre en français. "
                    "Distingue clairement ce qui vient de l'historique conversationnel et ce qui vient de la base si utile. "
                     + SOURCE_MARKER_INSTRUCTION + " "
                     + ANSWER_ACTION_INSTRUCTION + " "
                     + FINAL_ANSWER_STYLE
                ),
            },
            {
                "role": "user",
                "content": f"Route planifiee: {route_name}\n\nQuestion: {question}\n\nHistorique:\n{memory_block}\n\nSources:\n{source_block}",
            },
        ]
    response = client.responses.create(model=answer_model, input=input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError(f"Le modele n'a pas renvoye de texte exploitable pour la route {route_name}.")


def generate_sql_answer(
    client: OpenAI | None,
    question: str,
    answer_model: str | None,
    sql_sub_intent: str | None,
    sources: list[dict[str, Any]],
    trace: dict[str, str] | None = None,
) -> str:
    if not sources:
        if trace is not None:
            trace["action"] = "abstain"
        if sql_sub_intent == "video_lookup":
            return "Je n'ai trouve aucune video correspondant a cette demande dans la base."
        return (
            "Je n'ai trouve aucun contenu correspondant a cette demande. "
            "Si tu fais reference a une video precise, indique son titre exact ou un mot-cle du titre."
        )

    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer"
        if sql_sub_intent == "video_lookup":
            lines = ["Videos trouvees :"]
            for index, item in enumerate(sources, start=1):
                lines.append(f"- [S{index}] {item['video_title']} ({item['video_url']})")
            return "\n".join(lines)
        return f"[S1] {sources[0]['text']}"

    context_blocks = []
    for index, source in enumerate(sources, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Resultat {index}",
                    f"Titre: {source['video_title']}",
                    f"URL: {source['video_url']}",
                    f"Texte: {source['text']}",
                ]
            )
        )

    system_prompt = (
        "Tu formules une reponse finale en francais a partir de resultats structures deja recuperes. "
        "N'invente aucune information absente. "
        "Si plusieurs videos sont trouvees, presente-les clairement. "
        + SOURCE_MARKER_INSTRUCTION + " "
        + ANSWER_ACTION_INSTRUCTION + " "
        + FINAL_ANSWER_STYLE
    )
    input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sous-route SQL: {sql_sub_intent}\n\nQuestion: {question}\n\nResultats:\n\n" + "\n\n".join(context_blocks)},
        ]
    response = client.responses.create(model=answer_model, input=input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable pour la route sql.")


def generate_final_answer(
    client: OpenAI | None,
    question: str,
    answer_model: str | None,
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
    trace: dict[str, str] | None = None,
) -> str:
    route = retrieval.get("route") or retrieval.get("retrieval_mode")
    source_evaluation = retrieval.get("source_evaluation") or {}
    action_hint = source_evaluation.get("action_hint")
    if route not in {"direct", "memory"} and action_hint == "clarify":
        if trace is not None:
            trace["action"] = "clarify"
        return source_evaluation.get("clarification_message") or "Peux-tu préciser ta question ?"
    if route not in {"direct", "memory"} and action_hint == "abstain" and source_evaluation.get("reason") == "low_rerank_relevance":
        if trace is not None:
            trace["action"] = "abstain"
        return "Je n'ai pas trouvé de source suffisamment pertinente pour répondre à cette question."
    if route == "direct":
        return retrieval.get("direct_answer") or "Je peux repondre directement a cette demande."
    if route == "rag" and retrieval.get("retrieval_mode") == "rag+structured_sql":
        return generate_sql_answer(client, question, answer_model, retrieval.get("sql_sub_intent"), sources, trace)
    if route == "rag":
        return generate_answer(client, question, answer_model, sources, trace)
    if route == "sql":
        return generate_sql_answer(client, question, answer_model, retrieval.get("sql_sub_intent"), sources, trace)
    if route == "memory":
        return generate_memory_answer(client, question, answer_model, retrieval.get("memory_items", []), trace)
    if route in {"multi_source", "agent"}:
        return generate_multi_source_answer(
            client,
            question,
            answer_model,
            route,
            retrieval.get("memory_items", []),
            sources,
            trace,
        )
    return generate_answer(client, question, answer_model, sources, trace)


def retrieve_chunks(payload: RagRequest, execution_plan: ExecutionPlan) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = get_openai_client()
    answer_model = normalize_model_name(payload.answerModel, DEFAULT_GENERATION_MODEL)
    embedding_model = normalize_model_name(payload.embeddingModel, DEFAULT_EMBEDDING_MODEL)
    rerank_model = normalize_model_name(payload.rerankModel or "", DEFAULT_RERANK_MODEL)

    prefilter_candidate_ids, prefilter_debug = prefilter_candidate_chunk_ids(execution_plan)
    question_embedding: list[float] | None = None
    if client is not None and payload.useSql:
        embedding_response = client.embeddings.create(model=embedding_model, input=query_terms(execution_plan))
        question_embedding = embedding_response.data[0].embedding

    bm25_chunks, bm25_debug = fetch_bm25_chunks(execution_plan, prefilter_candidate_ids)
    vector_chunks, vector_debug = fetch_vector_chunks(execution_plan, question_embedding, prefilter_candidate_ids)
    fused_chunks, fusion_debug = reciprocal_rank_fusion(bm25_chunks, vector_chunks, DEFAULT_RRF_TOP_N)

    if payload.useRerank:
        final_chunks, rerank_debug = rerank_chunks(
            execution_plan.query_text,
            fused_chunks,
            execution_plan.final_k,
            rerank_model,
        )
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
    client = get_openai_client()
    contextual_question, reformulation_trace = reformulate_question(
        payload.question,
        payload.conversationId,
        client,
    )
    planner_plan, planner_prompt, planner_raw, pydantic_verification = run_planner(contextual_question, client)
    planner_plan.title_hint = planner_plan.title_hint or extract_video_title_hint(contextual_question)
    planner_plan.speakers = resolve_speaker_filters(contextual_question, planner_plan)
    apply_deterministic_sql_policy(contextual_question, planner_plan)
    execution_plan = build_execution_plan(payload, planner_plan)

    base_retrieval = {
        "route": execution_plan.route,
        "direct_sub_intent": execution_plan.direct_sub_intent,
        "sql_sub_intent": execution_plan.sql_sub_intent,
        "planner_prompt": planner_prompt,
        "planner_response_raw": planner_raw,
        "pydantic_verification": pydantic_verification,
        "question_reformulation": reformulation_trace,
        "contextual_question": contextual_question,
        "planner_plan": planner_plan.model_dump(),
        "execution_plan": execution_plan.model_dump(),
        "validated_query": execution_plan.model_dump(),
    }

    if execution_plan.route == "direct":
        if execution_plan.direct_sub_intent == "social":
            return build_direct_retrieval(base_retrieval, build_social_answer(contextual_question), "direct")
        return build_direct_retrieval(
            base_retrieval,
            "Je peux repondre directement a ce type de message sans interroger la base, mais aucun sous-type direct n'a ete defini pour cette demande.",
            "direct",
        )

    if execution_plan.route == "memory":
        memory_items, memory_trace = fetch_conversation_memory(payload.conversationId)
        answer_model = normalize_model_name(payload.answerModel, DEFAULT_GENERATION_MODEL)
        retrieval = {
            **base_retrieval,
            "answer_model": answer_model,
            "embedding_model": fallback_trace.get("embedding_model"),
            "rerank_model": fallback_trace.get("rerank_model"),
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
        sql_sub_intent = execution_plan.sql_sub_intent or "video_lookup"
        sources, direct_trace = lookup_video_document(execution_plan, sql_sub_intent)
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

    if execution_plan.route in {"multi_source", "agent"}:
        memory_items: list[dict[str, str]] = []
        memory_trace: dict[str, Any] = {}
        doc_sources: list[dict[str, Any]] = []
        doc_trace: dict[str, Any] = {}

        if execution_plan.use_memory:
            memory_items, memory_trace = fetch_conversation_memory(payload.conversationId)

        if execution_plan.sql_main_source:
            sql_sub_intent = execution_plan.sql_sub_intent or "video_lookup"
            doc_sources, doc_trace = lookup_video_document(execution_plan, sql_sub_intent)
        elif execution_plan.use_rag or execution_plan.route == "agent":
            doc_sources, doc_trace = retrieve_chunks(payload, execution_plan)

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
        }
        return "", doc_sources, retrieval

    sources, retrieval = retrieve_chunks(payload, execution_plan)
    retrieval["route"] = "rag"
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


@app.on_event("startup")
def startup() -> None:
    global _STARTED_AT
    _STARTED_AT = datetime.now().astimezone()
    print(f"[backend] started_at={_STARTED_AT.isoformat(timespec='seconds')}", flush=True)
    ensure_chat_schema()


@app.get("/health")
def health() -> dict[str, str]:
    ensure_chat_schema()
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str | None]:
    return {
        "started_at": _STARTED_AT.isoformat(timespec="seconds") if _STARTED_AT else None,
    }


@app.get("/api/video-thumbnails", response_model=list[str])
def video_thumbnails() -> list[str]:
    with psycopg.connect(get_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT thumbnail_medium_url
                FROM videos
                WHERE thumbnail_medium_url IS NOT NULL
                  AND thumbnail_medium_url <> ''
                ORDER BY published_at DESC NULLS LAST, id DESC
                LIMIT 24
                """
            )
            return [row[0] for row in cursor.fetchall()]


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
        answer, sources, retrieval = orchestrate_request(payload)
        retrieval["source_evaluation"] = evaluate_source_sufficiency(
            retrieval.get("contextual_question", payload.question),
            sources,
            retrieval,
        )
        answer_trace: dict[str, str] = {}
        if not answer:
            answer = generate_final_answer(
                get_openai_client(),
                retrieval.get("contextual_question", payload.question),
                retrieval["answer_model"],
                retrieval,
                sources,
                answer_trace,
            )
        answer_action = answer_trace.get("action", "answer")
        retrieval["answer_action"] = answer_action
        if answer_action == "answer":
            answer, carousel_sources = select_answer_sources(answer, sources)
        else:
            carousel_sources = []
        retrieval["answer_source_indexes"] = [
            index for index, source in enumerate(sources, start=1) if source in carousel_sources
        ]
        conversation_id, message_id = store_chat_message(
            conversation_id=payload.conversationId,
            user_message=payload.question,
            answer_message=answer,
            question_reformulation_prompt=retrieval.get("question_reformulation", {}).get("prompt"),
            question_reformulation_response_raw=retrieval.get("question_reformulation", {}).get("response_raw"),
            contextual_question=retrieval.get("contextual_question", payload.question),
            planner_prompt=retrieval["planner_prompt"],
            planner_response_raw=retrieval["planner_response_raw"],
            pydantic_verification=retrieval["pydantic_verification"],
            execution_plan_json=retrieval["execution_plan"],
            sql_query=sql_trace_for_storage(retrieval),
            prefilter_trace=retrieval["prefilter"],
            bm25_trace=retrieval["bm25"],
            vector_trace=retrieval["vector"],
            rrf_trace=retrieval["rrf"],
            rerank_trace=retrieval["rerank"],
            source_evaluation_trace=retrieval["source_evaluation"],
            retrieved_chunks=carousel_sources,
            answer_prompt=answer_trace.get("prompt"),
            answer_response_raw=answer_trace.get("response_raw"),
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RagResponse(
        conversation_id=conversation_id,
        message_id=message_id,
        answer=answer,
        action=answer_action,
        sources=[ChunkSource(**source) for source in carousel_sources],
        retrieval=retrieval,
    )
