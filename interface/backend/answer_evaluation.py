from __future__ import annotations

import json
import os
from typing import Any

from interface.backend.llm_providers import LLMClientProtocol
from interface.backend.utilities import safe_json_loads, serialize_openai_response


SHADOW_EVALUATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["acceptable", "needs_correction"],
        },
        "issue": {
            "type": "string",
            "enum": [
                "none",
                "bad_retrieval",
                "insufficient_sources",
                "unsupported_answer",
                "ambiguous_question",
            ],
        },
        "reason": {"type": "string"},
        "retrieval_quality": {"type": "number", "minimum": 0, "maximum": 1},
        "answer_grounded": {"type": "boolean"},
        "suggested_correction": {
            "type": ["string", "null"],
        },
    },
    "required": [
        "verdict",
        "issue",
        "reason",
        "retrieval_quality",
        "answer_grounded",
        "suggested_correction",
    ],
    "additionalProperties": False,
}


SHADOW_EVALUATION_SYSTEM_PROMPT = (
    "Tu évalues une réponse RAG sans la réécrire. Vérifie si elle répond à la "
    "question et si chaque affirmation factuelle est soutenue par les sources. "
    "Le verdict porte uniquement sur la réponse affichée : acceptable si elle "
    "peut être conservée, needs_correction si elle doit être remplacée. Une "
    "question ambiguë correctement traitée par une demande de précision a donc "
    "verdict=acceptable et issue=ambiguous_question. Utilise issue=none en "
    "l'absence de problème particulier, bad_retrieval si les sources sont hors "
    "sujet, insufficient_sources si elles sont pertinentes mais incomplètes, "
    "unsupported_answer si la réponse dépasse les sources, et ambiguous_question "
    "si la demande nécessite une précision. "
    "Le diagnostic est interne et ne doit contenir aucun message destiné à l'utilisateur."
)


def shadow_evaluation_enabled() -> bool:
    return os.getenv("RAG_SHADOW_EVALUATION_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def shadow_evaluation_model(default_model: str) -> str:
    return os.getenv("RAG_SHADOW_EVALUATION_MODEL", "").strip() or default_model


def _source_excerpt(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_title": source.get("video_title"),
        "video_url": source.get("video_url"),
        "text": str(source.get("text") or "")[:6000],
        "section_context": source.get("section_context"),
        "global_context": source.get("global_context"),
    }


def evaluate_answer_shadow(
    client: LLMClientProtocol,
    model: str,
    question: str,
    answer: str,
    action: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    input_messages = [
        {"role": "system", "content": SHADOW_EVALUATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "answer": answer,
                    "action": action,
                    "sources": [_source_excerpt(source) for source in sources],
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = client.responses.create(
        model=model,
        input=input_messages,
        response_schema=SHADOW_EVALUATION_RESPONSE_SCHEMA,
    )
    raw_output = str(getattr(response, "output_text", "") or "").strip()
    parsed = safe_json_loads(raw_output)
    if not isinstance(parsed, dict):
        raise ValueError("Le diagnostic shadow n'est pas un objet JSON.")
    verdict = parsed.get("verdict")
    if verdict not in {"acceptable", "needs_correction"}:
        raise ValueError("Le diagnostic shadow contient un verdict invalide.")
    issue = parsed.get("issue")
    if issue not in {
        "none",
        "bad_retrieval",
        "insufficient_sources",
        "unsupported_answer",
        "ambiguous_question",
    }:
        raise ValueError("Le diagnostic shadow contient une cause invalide.")
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Le diagnostic shadow doit fournir une raison.")
    retrieval_quality = parsed.get("retrieval_quality")
    if (
        not isinstance(retrieval_quality, (int, float))
        or isinstance(retrieval_quality, bool)
        or not 0 <= float(retrieval_quality) <= 1
    ):
        raise ValueError("La qualité du retrieval shadow doit être comprise entre 0 et 1.")
    answer_grounded = parsed.get("answer_grounded")
    if not isinstance(answer_grounded, bool):
        raise ValueError("Le diagnostic shadow answer_grounded doit être booléen.")
    suggested_correction = parsed.get("suggested_correction")
    if suggested_correction is not None and not isinstance(suggested_correction, str):
        raise ValueError("La correction shadow doit être une chaîne ou null.")
    return {
        "enabled": True,
        "mode": "shadow",
        "model": model,
        "verdict": verdict,
        "issue": issue,
        "status": (
            "acceptable"
            if verdict == "acceptable"
            else (issue if issue != "none" else "unsupported_answer")
        ),
        "reason": reason.strip(),
        "retrieval_quality": float(retrieval_quality),
        "answer_grounded": answer_grounded,
        "suggested_correction": suggested_correction,
        "prompt": json.dumps(
            {"model": model, "input": input_messages},
            ensure_ascii=False,
        ),
        "response_raw": serialize_openai_response(response),
    }
