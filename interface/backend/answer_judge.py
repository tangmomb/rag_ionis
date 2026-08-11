from __future__ import annotations

import json
from typing import Any

from interface.backend.llm_providers import LLMClientProtocol
from interface.backend.utilities import safe_json_loads, serialize_openai_response


ANSWER_JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "retry_stage": {
            "type": "string",
            "enum": ["none", "sql", "generation"],
        },
        "reason": {"type": "string"},
        "correction": {"type": "string"},
    },
    "required": ["valid", "retry_stage", "reason", "correction"],
    "additionalProperties": False,
}

ANALYTICS_VIDEO_METADATA_CORRECTION = (
    "Retourner une ligne par vidéo avec video_id, video_title, video_url et "
    "thumbnail_medium_url. Pour les totaux ou comparaisons par personne, utiliser "
    "une fonction fenêtre COUNT/SUM OVER (PARTITION BY ...) afin de conserver ces "
    "métadonnées sur chaque ligne."
)


def _compact_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = []
    for index, source in enumerate(sources[:10], start=1):
        compacted.append(
            {
                "index": index,
                "video_title": source.get("video_title"),
                "video_url": source.get("video_url"),
                "persons": source.get("persons", []),
                "text": str(source.get("text") or "")[:800],
            }
        )
    return compacted


def _analytics_sources_lack_video_metadata(
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
) -> bool:
    if retrieval.get("sql_sub_intent") != "analytics" or not sources:
        return False
    return any(
        not str(source.get("video_title") or "").strip()
        or str(source.get("video_title") or "").strip() == "Résultat analytique"
        or not str(source.get("video_url") or "").strip()
        for source in sources
    )


def build_answer_judge_prompt(
    question: str,
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
    answer: str,
    action: str,
) -> tuple[str, str]:
    system_prompt = (
        "Tu vérifies la cohérence d'une réponse produite par un assistant RAG. "
        "Contrôle que le plan et le SQL respectent l'intention de la question, que "
        "les résultats permettent la réponse et que le brouillon n'invente rien. "
        "Pour une comparaison entre plusieurs personnes, leurs vidéos forment une "
        "union, sauf si la question demande explicitement leur présence ensemble dans "
        "la même vidéo. Un résultat SQL vide ne justifie pas une clarification si le SQL "
        "a transformé à tort une comparaison en intersection. Choisis retry_stage=sql "
        "si une autre requête SQL peut corriger le problème, retry_stage=generation si "
        "les données sont bonnes mais la réponse les interprète mal, sinon none. "
        "Une source analytique portant sur des vidéos doit contenir un video_title "
        "et un video_url réels. Sinon choisis retry_stage=sql et exige une ligne par "
        "vidéo avec ses métadonnées, en conservant les agrégats via une fonction fenêtre. "
        "Si valid=true, retry_stage doit valoir none et correction doit être vide. "
        "La correction doit être une instruction courte et directement exploitable."
    )
    sql_context = retrieval.get("direct_lookup") or {}
    if not sql_context:
        sql_actions = [
            item
            for item in retrieval.get("multi_source_actions", [])
            if item.get("source") == "sql"
        ]
        sql_context = sql_actions[-1] if sql_actions else {}
    context = {
        "question_originale": question,
        "question_reformulee": retrieval.get("contextual_question"),
        "route": retrieval.get("route"),
        "sql_sub_intent": retrieval.get("sql_sub_intent"),
        "execution_plan": retrieval.get("execution_plan", {}),
        "sql": sql_context.get("sql") or sql_context.get("request", {}).get("sql"),
        "sql_params": sql_context.get("params") or sql_context.get("request", {}).get("params", []),
        "sql_status": sql_context.get("status"),
        "sql_result_count": sql_context.get("result_count", len(sources)),
        "sources": _compact_sources(sources),
        "brouillon": {"answer": answer, "action": action},
    }
    return system_prompt, json.dumps(context, ensure_ascii=False)


def judge_final_answer(
    client: LLMClientProtocol | None,
    model: str | None,
    question: str,
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
    answer: str,
    action: str,
) -> dict[str, Any]:
    if client is None or not model:
        return {
            "status": "skipped",
            "valid": True,
            "retry_stage": "none",
            "reason": "no_llm_client",
            "correction": "",
        }

    if _analytics_sources_lack_video_metadata(retrieval, sources):
        return {
            "status": "completed",
            "valid": False,
            "retry_stage": "sql",
            "reason": (
                "Les résultats analytiques ne sont pas rattachés à des vidéos "
                "concrètes : video_title ou video_url manque."
            ),
            "correction": ANALYTICS_VIDEO_METADATA_CORRECTION,
        }

    system_prompt, user_prompt = build_answer_judge_prompt(
        question,
        retrieval,
        sources,
        answer,
        action,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = client.responses.create(
            model=model,
            input=messages,
            max_output_tokens=600,
            response_schema=ANSWER_JUDGE_RESPONSE_SCHEMA,
        )
        payload = safe_json_loads(
            str(getattr(response, "output_text", "") or "").strip()
        )
        valid = bool(payload.get("valid", False))
        retry_stage = str(payload.get("retry_stage") or "none")
        if retry_stage not in {"none", "sql", "generation"}:
            retry_stage = "none"
        if valid:
            retry_stage = "none"
        return {
            "status": "completed",
            "valid": valid,
            "retry_stage": retry_stage,
            "reason": str(payload.get("reason") or "").strip(),
            "correction": str(payload.get("correction") or "").strip(),
            "prompt": messages,
            "response_raw": serialize_openai_response(response),
        }
    except Exception as exc:
        return {
            "status": "error",
            "valid": True,
            "retry_stage": "none",
            "reason": "judge_error",
            "correction": "",
            "error": str(exc),
        }
