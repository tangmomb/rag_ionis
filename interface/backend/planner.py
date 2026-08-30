from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from interface.backend.config import (
    DEFAULT_BM25_LIMIT,
    DEFAULT_FINAL_K,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
)
from interface.backend.database import connect_database, fetch_conversation_history
from interface.backend.llm_providers import LLMClientProtocol
from interface.backend.schemas import ExecutionPlan, PlannerPlan, RagRequest
from interface.backend.utilities import normalize_text, safe_json_loads, serialize_openai_response


PLANNER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["direct", "rag", "multi_source"],
        },
        "sql_sub_intent": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "specific_persons",
                        "analytics",
                        "description",
                        "transcript_verbatim",
                    ],
                },
                {"type": "null"},
            ],
        },
        "query_text": {"type": "string"},
        "query_text_bm25": {"type": "string"},
        "title_hint": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "persons": {
            "type": "array",
            "items": {"type": "string"},
        },
        "companies": {
            "type": "array",
            "items": {"type": "string"},
        },
        "published_after": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "published_before": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
    },
    "required": [
        "route",
        "sql_sub_intent",
        "query_text",
        "query_text_bm25",
        "title_hint",
        "persons",
        "companies",
        "published_after",
        "published_before",
    ],
    "additionalProperties": False,
}


REFORMULATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "follow_up": {"type": "boolean"},
        "reformulated_question": {"type": "string"},
    },
    "required": ["follow_up", "reformulated_question"],
    "additionalProperties": False,
}


# La correction tolère une faute légère dans un prénom ou un nom, sans faire
# remonter des noms qui ne partagent qu'une syllabe courte.
PERSON_NAME_PART_SIMILARITY_THRESHOLD = 0.85
COMPANY_TITLE_SIMILARITY_THRESHOLD = 0.85
VIDEO_TITLE_SIMILARITY_THRESHOLD = 0.85
REFORMULATION_HISTORY_EXCHANGES = 3
REFORMULATION_HISTORY_MAX_MESSAGES = REFORMULATION_HISTORY_EXCHANGES * 2
REFORMULATION_HISTORY_MAX_CHARS_PER_MESSAGE = 1_600


def find_persons_in_enriched_transcripts(
    requested_persons: list[str],
) -> list[str]:
    """Retourne les noms présents comme expressions complètes dans transcript_enriched."""
    candidates = [
        str(value).strip()
        for value in requested_persons
        if str(value).strip()
    ]
    if not candidates:
        return []

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT requested.name
                FROM unnest(%s::text[]) WITH ORDINALITY AS requested(name, ordinal)
                WHERE EXISTS (
                    SELECT 1
                    FROM transcripts transcript_row
                    WHERE transcript_row.transcript_enriched IS NOT NULL
                      AND concat(
                          ' ',
                          btrim(regexp_replace(
                              unaccent(lower(transcript_row.transcript_enriched)),
                              '[^[:alnum:]]+',
                              ' ',
                              'g'
                          )),
                          ' '
                      ) LIKE concat(
                          chr(37),
                          ' ',
                          btrim(regexp_replace(
                              unaccent(lower(requested.name)),
                              '[^[:alnum:]]+',
                              ' ',
                              'g'
                          )),
                          ' ',
                          chr(37)
                      )
                )
                ORDER BY requested.ordinal
                """,
                (candidates,),
            )
            return [str(row[0]).strip() for row in cursor.fetchall()]


def build_planner_prompt(
    question: str,
    system_prompt_override: str | None = None,
) -> tuple[str, str]:
    default_system_prompt = (
        "Tu planifies la requete d'un assistant RAG sans y repondre. "
        "Première étape, identifier les personnes ou entreprises mentionnées dans la question. Les stocker dans persons et companies. "
        "Deuxième étape, identifier les dates de publication mentionnées dans la question. Les stocker dans published_after et published_before sous forme de chaînes ISO 8601 (YYYY-MM-DD). "
        "Troisième étape, identifier un titre de video mentionné dans la question. Le stocker dans title_hint. "
        "Quatrième étape, produire les clés query_text et query_text_bm25. query_text est la question reformulée pour la recherche RAG, c'est elle qui sera calculée pour l'embedding donc attention à son écriture sémantique. query_text_bm25 est la question reformulée pour la recherche BM25, elle doit être plus courte et plus directe, adaptée pour une recherche par mots-clés. "
        "Cinquième et dernière étape, choisir la stratégie pour répondre à la question via les clés route et sql_sub_intent. route peut être 'direct', 'rag' ou 'multi_source'. sql_sub_intent peut être 'specific_persons', 'analytics', 'description', 'transcript_verbatim' ou 'null'. "
        "route='direct' si la question ou le message est une salutation ou une formule de politesse. route='rag' pour toute question qui demande une information. route='multi_source' si tu as identifié plus d'une personne ou entreprise cumulées dans la question. (1 personne + 1 entreprise = 2)."
        "sql_sub_intent='specific_persons' si tu as identifié des personnes ou entreprises dans la question, sauf si elle demande une analyse structurée. sql_sub_intent='analytics' pour les statistiques, comptages, classements et métadonnées structurées comme la date de publication, la durée, le type de vidéo ou la présence de sous-titres. sql_sub_intent='description' uniquement si le mot exact 'description' apparaît dans la question et demande la description d'une video. sql_sub_intent='transcript_verbatim' si la question demande explicitement le transcript complet d'une video. sql_sub_intent='null' si la question ne demande pas explicitement de données structurées. "
        "Toutes les valeurs textuelles doivent être en texte normal, sans Markdown."

    )
    system_prompt = (system_prompt_override or "").strip() or default_system_prompt
    return system_prompt, question


def is_social_message(question: str) -> bool:
    """N'accepte la route directe que pour un message entièrement social."""
    normalized = normalize_text(question).strip()
    normalized = re.sub(r"[^a-z0-9' ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    social_patterns = (
        r"(?:bonjour|bonsoir|salut|coucou|hello|hey)(?: (?:comment )?ca va)?",
        r"(?:merci|merci beaucoup|merci bien|je te remercie|je vous remercie)",
        r"(?:ca va|comment ca va|comment vas tu|tu vas bien|vous allez bien)",
        r"(?:au revoir|a bientot|bonne journee|bonne soiree|bye)",
    )
    return any(re.fullmatch(pattern, normalized) for pattern in social_patterns)


def build_social_answer(question: str) -> str:
    lower = normalize_text(question).strip()
    if any(token in lower for token in ("merci",)):
        return "Avec plaisir. Si tu veux, je peux aussi t'aider à chercher une vidéo ou répondre à une question sur la base."
    if any(token in lower for token in ("ca va", "ca roule", "comment ca va")):
        return "Ça va bien, merci. Je suis prêt à t'aider sur la base vidéo si tu veux."
    if any(token in lower for token in ("au revoir", "a bientot", "bonne journee", "bonne soiree")):
        return "À bientôt."
    return "Bonjour. Je peux t'aider à trouver une vidéo, un transcript, un résumé ou répondre à une question à partir de la base."


def normalize_planner_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise uniquement les choix sémantiques; les sources sont dérivées ensuite."""
    normalized = dict(payload)
    if "companies" not in normalized and "company" in normalized:
        normalized["companies"] = normalized.pop("company")
    route = str(normalized.get("route") or "").strip()
    sql_sub_intent = str(normalized.get("sql_sub_intent") or "").strip() or None
    legacy_intent_map = {
        "video_lookup": "specific_persons",
        "lookup": "specific_persons",
        "video_transcript": "transcript_verbatim",
        "video_description": "description",
        "video_stats": "analytics",
        "stats": "analytics",
    }
    sql_intents = {
        "specific_persons",
        "analytics",
        "description",
        "transcript_verbatim",
        "transcript_qa",
    }
    legacy_routes = {"rag_chunks": "rag", "sql_request": "rag", "social": "direct"}

    if route in legacy_intent_map or route in sql_intents:
        sql_sub_intent = legacy_intent_map.get(route, route)
        route = "rag"
    elif route == "sql":
        route = "rag"
        sql_sub_intent = (
            legacy_intent_map.get(sql_sub_intent, sql_sub_intent)
            or "specific_persons"
        )
    else:
        route = legacy_routes.get(route, route)
        sql_sub_intent = legacy_intent_map.get(sql_sub_intent, sql_sub_intent)

    if route not in {"direct", "rag", "multi_source"}:
        route = "rag"
    if route == "direct":
        sql_sub_intent = None
    elif sql_sub_intent not in sql_intents:
        sql_sub_intent = None

    normalized["route"] = route
    normalized["sql_sub_intent"] = sql_sub_intent
    for derived_key in ("use_sql", "use_rag", "sql_main_source"):
        normalized.pop(derived_key, None)
    return normalized


def derive_plan_sources(planner_plan: PlannerPlan) -> None:
    """Déduit les sources d'exécution sans demander ces booléens au LLM."""
    has_sql_intent = (
        planner_plan.sql_sub_intent is not None
        and not (
            planner_plan.sql_sub_intent == "specific_persons"
            and not (planner_plan.persons or planner_plan.companies)
        )
    )

    if planner_plan.route == "direct":
        planner_plan.sql_sub_intent = None
        planner_plan.use_rag = False
        planner_plan.sql_main_source = False
    elif planner_plan.route == "multi_source":
        planner_plan.use_rag = not has_sql_intent
        planner_plan.sql_main_source = has_sql_intent
    else:
        planner_plan.use_rag = True
        planner_plan.sql_main_source = has_sql_intent


def run_planner(
    question: str,
    client: LLMClientProtocol | None,
    model: str = DEFAULT_PLANNER_MODEL,
    system_prompt_override: str | None = None,
) -> tuple[PlannerPlan, str | None, str | None, bool]:
    system_prompt, user_prompt = build_planner_prompt(
        question,
        system_prompt_override,
    )
    planner_input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw_prompt = json.dumps(
        {"model": model, "input": planner_input},
        ensure_ascii=False,
    )

    if client is None:
        fallback = PlannerPlan(
            route="rag",
            query_text=question,
        )
        derive_plan_sources(fallback)
        return fallback, raw_prompt, json.dumps(fallback.model_dump(), ensure_ascii=False), False

    response = client.responses.create(
        model=model,
        input=planner_input,
        response_schema=PLANNER_RESPONSE_SCHEMA,
    )
    raw_response = serialize_openai_response(response)
    raw = getattr(response, "output_text", "").strip()
    if not raw:
        fallback = PlannerPlan(
            route="rag",
            query_text=question,
        )
        derive_plan_sources(fallback)
        return fallback, raw_prompt, raw_response, False

    try:
        parsed = normalize_planner_output(safe_json_loads(raw))
        if not parsed.get("route"):
            parsed["route"] = "rag"
        if not parsed.get("query_text"):
            parsed["query_text"] = question
        validated = PlannerPlan.model_validate(parsed)
        derive_plan_sources(validated)
        return validated, raw_prompt, raw_response, True
    except Exception:
        fallback = PlannerPlan(
            route="rag",
            query_text=question,
        )
        derive_plan_sources(fallback)
        return fallback, raw_prompt, raw_response, False


def build_execution_plan(
    payload: RagRequest,
    planner_plan: PlannerPlan,
) -> ExecutionPlan:
    derive_plan_sources(planner_plan)
    bm25_query = (planner_plan.query_text_bm25 or "").strip()
    if not bm25_query:
        bm25_query = (planner_plan.query_text or payload.question).strip() or payload.question

    sql_main_source = planner_plan.sql_main_source
    return ExecutionPlan(
        route=planner_plan.route or "rag",
        sql_sub_intent=planner_plan.sql_sub_intent,
        raw_question=payload.question,
        query_text=(planner_plan.query_text or payload.question).strip() or payload.question,
        query_text_bm25=bm25_query,
        title_hint=planner_plan.title_hint,
        persons=planner_plan.persons,
        companies=planner_plan.companies,
        published_after=planner_plan.published_after,
        published_before=planner_plan.published_before,
        use_rag=planner_plan.use_rag,
        sql_main_source=sql_main_source,
        top_k=None if sql_main_source else DEFAULT_BM25_LIMIT,
        final_k=None if sql_main_source else DEFAULT_FINAL_K,
    )


def has_explicit_structured_sql_request(question: str) -> bool:
    normalized = "".join(
        char for char in unicodedata.normalize("NFD", question.lower()) if unicodedata.category(char) != "Mn"
    )
    return bool(
        re.search(
            r"\b(?:url|lien|titre|description|descriptif|statistiques?|stats?|vues?|likes?|commentaires?|date de publication|publie|publiee|publiees|"
            r"transcript|transcription|verbatim|timecodes?|sous-titres?|intervenant(?:e|s)?|speaker(?:s)?|"
            r"metier|profession|poste|fonction|role)\b",
            normalized,
        )
        or re.search(r"\bqui\s+(?:intervient|parle)\b", normalized)
        or re.search(r"\b(?:quelle?|quelles?)\s+(?:video|videos|url|lien|titre|date)\b", normalized)
        or re.search(r"\b(?:trouve|trouver|cherche|chercher|liste|lister)\b.*\b(?:video|videos|transcript|transcription)\b", normalized)
    )


def has_analytics_request(question: str) -> bool:
    normalized = normalize_text(question)
    return bool(
        re.search(
            r"\b(?:statistiques?|stats?|vues?|likes?|commentaires?)\b",
            normalized,
        )
        or re.search(
            r"\b(?:date de publication|publiee?|publiees?|mise en ligne)\b",
            normalized,
        )
        or re.search(r"\b(?:duree|combien de temps)\b", normalized)
        or re.search(r"\btype de video\b", normalized)
        or re.search(r"\b(?:sous titres?|sous-titres?)\b", normalized)
    )


def has_document_content_request(question: str) -> bool:
    """Détecte une demande de synthèse ou d'analyse du contenu vidéo."""
    normalized = normalize_text(question)
    return bool(
        re.search(r"\bque\s+dit\b", normalized)
        or re.search(r"\bqu[' ]?est ce que\b.*\bdit\b", normalized)
        or re.search(r"\b(?:resume|resumer|synthese|synthetise|points? principaux?)\b", normalized)
        or re.search(r"\b(?:de quoi parle|qu[' ]?apprend|explique)\b", normalized)
    )


def has_temporal_transcript_request(question: str) -> bool:
    """Détecte une demande de localisation temporelle dans une vidéo."""
    normalized = normalize_text(question)
    return bool(
        re.search(r"\b(?:a quel moment|a quelle minute|dans quel passage)\b", normalized)
        or re.search(r"\b(?:quand|ou)\b.*\bdans\s+(?:la|cette|une)\s+video\b", normalized)
        or re.search(r"\b(?:timecode|horodatage)\b", normalized)
    )


def has_person_title_request(question: str) -> bool:
    """Détecte une question portant sur le poste ou la fonction d'une personne."""
    normalized = normalize_text(question)
    return bool(
        re.search(
            r"\b(?:metier|profession|poste|fonction|role)\s+(?:de|d[' ])\b",
            normalized,
        )
        or re.search(
            r"\b(?:quel(?:le)? est|connaitre|donne|indiquer?)\b.*"
            r"\b(?:metier|profession|poste|fonction|role)\b",
            normalized,
        )
        or re.search(
            r"\btravaille(?:-t-il|-t-elle)?\s+(?:comme|en tant que)\b",
            normalized,
        )
    )


def apply_deterministic_sql_policy(
    question: str,
    planner_plan: PlannerPlan,
) -> str | None:
    """Empêche le planner de basculer arbitrairement la source SQL principale."""
    policy_correction: str | None = None
    if planner_plan.route == "direct":
        if planner_plan.persons or planner_plan.companies:
            planner_plan.route = "rag"
            policy_correction = "direct_with_entities_to_rag"
        elif not is_social_message(question):
            planner_plan.route = "rag"
            policy_correction = "direct_non_social_to_rag"
        else:
            planner_plan.sql_sub_intent = None
            derive_plan_sources(planner_plan)
            return None

    if planner_plan.sql_sub_intent == "analytics" or has_analytics_request(question):
        planner_plan.sql_sub_intent = "analytics"
        derive_plan_sources(planner_plan)
        return policy_correction

    if planner_plan.persons or planner_plan.companies:
        planner_plan.sql_sub_intent = "specific_persons"
        derive_plan_sources(planner_plan)
        return policy_correction

    if has_temporal_transcript_request(question):
        planner_plan.sql_sub_intent = "transcript_qa"
        derive_plan_sources(planner_plan)
        return policy_correction

    if has_person_title_request(question):
        planner_plan.sql_sub_intent = "specific_persons"
        derive_plan_sources(planner_plan)
        return policy_correction

    if has_document_content_request(question):
        if planner_plan.title_hint:
            planner_plan.sql_sub_intent = "transcript_qa"
        else:
            planner_plan.sql_sub_intent = None
        derive_plan_sources(planner_plan)
        return policy_correction

    if not has_explicit_structured_sql_request(question):
        # Le planner peut conserver SQL pour une question video complexe.
        # Si la recherche structuree echoue, orchestrate_request tentera le RAG.
        derive_plan_sources(planner_plan)
        return policy_correction

    normalized = normalize_text(question)
    if has_analytics_request(question):
        planner_plan.sql_sub_intent = "analytics"
    elif re.search(r"\b(?:description|descriptif|decris)\b", normalized):
        planner_plan.sql_sub_intent = "description"
    elif any(term in normalized for term in ("transcript", "transcription", "verbatim", "timecode", "sous-titre")):
        planner_plan.sql_sub_intent = "transcript_verbatim"
    elif re.search(r"\b(?:intervenants?|speakers?|metier|profession|poste|fonction|role)\b", normalized) or re.search(
        r"\bqui\s+(?:intervient|parle)\b",
        normalized,
    ):
        planner_plan.sql_sub_intent = "specific_persons"
    else:
        planner_plan.sql_sub_intent = "specific_persons"
    derive_plan_sources(planner_plan)
    return policy_correction


def has_structured_sql_filters(query: ExecutionPlan) -> bool:
    return any(
        [
            bool(query.title_hint),
            bool(query.persons),
            bool(query.companies),
            bool(query.published_after),
            bool(query.published_before),
        ]
    )


def resolve_person_filters(
    requested_persons: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Score every speaker suggestion and retain only confident SQL filters."""
    candidates = [
        str(value).strip()
        for value in requested_persons
        if str(value).strip()
    ]

    if not candidates:
        return [], {
            "applied": False,
            "ambiguous": False,
            "requested": [],
            "suggestion_speakers": [],
            "suggestion_transcripts": [],
        }

    resolved: list[str] = []
    unresolved_candidates: list[str] = []
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT name
                FROM speakers
                WHERE name IS NOT NULL AND btrim(name) <> ''
                ORDER BY name
                """
            )
            database_persons = [str(row[0]).strip() for row in cursor.fetchall()]

    transcript_matches = find_persons_in_enriched_transcripts(candidates)
    transcript_match_keys = {
        normalize_text(person) for person in transcript_matches
    }
    speaker_scores: dict[str, float] = {}

    def person_similarity(candidate: str, person: str) -> float:
        # Les tirets font partie de la graphie d'un prénom composé, mais
        # l'utilisateur peut les omettre ("Lou Ann" / "Lou-Ann").
        candidate_parts = re.sub(r"[-'’]", " ", normalize_text(candidate)).split()
        person_parts = re.sub(r"[-'’]", " ", normalize_text(person)).split()
        if not candidate_parts or not person_parts:
            return 0.0

        if len(candidate_parts) == 1:
            # Une recherche sur un seul mot peut désigner le prénom ou le
            # nom, mais jamais un mot intermédiaire arbitraire.
            comparable_parts = (person_parts[0], person_parts[-1])
            return max(
                (SequenceMatcher(None, candidate_parts[0], part).ratio() for part in comparable_parts),
                default=0.0,
            )

        # Pour un nom complet, prénom et nom doivent contribuer ensemble.
        endpoint_score = (
            SequenceMatcher(None, candidate_parts[0], person_parts[0]).ratio()
            + SequenceMatcher(None, candidate_parts[-1], person_parts[-1]).ratio()
        ) / 2

        partial_scores = [endpoint_score]
        if len(candidate_parts) < len(person_parts):
            prefix_parts = person_parts[: len(candidate_parts)]
            suffix_parts = person_parts[-len(candidate_parts) :]
            partial_scores.extend(
                sum(
                    SequenceMatcher(None, candidate_part, person_part).ratio()
                    for candidate_part, person_part in zip(
                        candidate_parts,
                        aligned_parts,
                    )
                )
                / len(candidate_parts)
                for aligned_parts in (prefix_parts, suffix_parts)
            )
        return max(partial_scores)

    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        candidate_matches = [
            (person_similarity(candidate, person), person)
            for person in database_persons
        ]
        confident_match = False
        for score, person in candidate_matches:
            speaker_scores[person] = max(speaker_scores.get(person, 0.0), score)
            if score > PERSON_NAME_PART_SIMILARITY_THRESHOLD:
                confident_match = True
                if person not in resolved:
                    resolved.append(person)
        if not confident_match and normalized_candidate not in transcript_match_keys:
            unresolved_candidates.append(candidate)

    suggestion_speakers = [
        {"person": person, "score": round(score, 3)}
        for person, score in sorted(
            speaker_scores.items(), key=lambda item: (-item[1], item[0])
        )
        if score > PERSON_NAME_PART_SIMILARITY_THRESHOLD
    ]
    suggestion_transcripts = [
        {"person": person, "score": 1.0}
        for person in transcript_matches
    ]

    if unresolved_candidates:
        return resolved, {
            "applied": True,
            "ambiguous": True,
            "requested": candidates,
            "ambiguous_requests": unresolved_candidates,
            "unresolved_requests": unresolved_candidates,
            "suggestion_speakers": suggestion_speakers,
            "suggestion_transcripts": suggestion_transcripts,
            "message": "Peux-tu préciser le nom de l'intervenant ?",
        }

    return resolved, {
        "applied": True,
        "ambiguous": False,
        "requested": candidates,
        "suggestion_speakers": suggestion_speakers,
        "suggestion_transcripts": suggestion_transcripts,
    }


def resolve_company_filters(
    requested_companies: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Suggest confident company terms found in speaker titles."""
    candidates = [
        str(value).strip()
        for value in requested_companies
        if str(value).strip()
    ]
    if not candidates:
        return [], {
            "requested": [],
            "suggestion_companies": [],
        }

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT title
                FROM speakers
                WHERE title IS NOT NULL AND btrim(title) <> ''
                ORDER BY title
                """
            )
            database_titles = [str(row[0]).strip() for row in cursor.fetchall()]

    company_scores: dict[str, float] = {}
    for candidate in candidates:
        candidate_tokens = re.findall(r"\w+", normalize_text(candidate))
        if not candidate_tokens:
            continue

        phrase_size = len(candidate_tokens)
        for title in database_titles:
            title_tokens = re.findall(r"\w+", normalize_text(title))
            for index in range(0, len(title_tokens) - phrase_size + 1):
                phrase = " ".join(title_tokens[index:index + phrase_size])
                score = SequenceMatcher(
                    None,
                    " ".join(candidate_tokens),
                    phrase,
                ).ratio()
                company_scores[phrase] = max(company_scores.get(phrase, 0.0), score)

    suggestion_companies = [
        {"company": company, "score": round(score, 3)}
        for company, score in sorted(
            company_scores.items(), key=lambda item: (-item[1], item[0])
        )
        if score >= COMPANY_TITLE_SIMILARITY_THRESHOLD
    ]
    resolved = [item["company"] for item in suggestion_companies]

    return resolved, {
        "requested": candidates,
        "suggestion_companies": suggestion_companies,
    }


def resolve_title_hint(title_hint: str | None) -> tuple[str | None, dict[str, Any]]:
    """Suggest confident canonical video titles for an explicit title hint."""
    requested = str(title_hint or "").strip()
    if not requested:
        return None, {"requested": None, "suggestion_titles": []}

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT title
                FROM videos
                WHERE title IS NOT NULL AND btrim(title) <> ''
                ORDER BY title
                """
            )
            database_titles = [str(row[0]).strip() for row in cursor.fetchall()]

    normalized_requested = normalize_text(requested)
    suggestions = [
        {"title": title, "score": round(score, 3)}
        for score, title in sorted(
            (
                (SequenceMatcher(None, normalized_requested, normalize_text(title)).ratio(), title)
                for title in database_titles
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if score >= VIDEO_TITLE_SIMILARITY_THRESHOLD
    ]
    resolved = str(suggestions[0]["title"]) if suggestions else None
    return resolved, {"requested": requested, "suggestion_titles": suggestions}


def extract_video_title_hint(question: str) -> str | None:
    """Extrait un titre explicitement fourni par l'utilisateur."""
    video_quoted_match = re.search(
        r"\b(?:video|vidéo)\s*[\"«“]([^\"»”]+)[\"»”]",
        question,
        flags=re.IGNORECASE,
    )
    if video_quoted_match:
        return video_quoted_match.group(1).strip() or None

    quoted_match = re.search(r"[\"«“]([^\"»”]+)[\"»”]\s+dans\s+le\s+titre", question, flags=re.IGNORECASE)
    if quoted_match:
        return quoted_match.group(1).strip() or None

    plain_match = re.search(r"(?:avec|intitulee?|intitulee?\s+la\s+video)\s+(.+?)\s+dans\s+le\s+titre", question, flags=re.IGNORECASE)
    if plain_match:
        return plain_match.group(1).strip(" \"«“»”'\t") or None

    return None


def sanitize_video_title_hint(question: str, title_hint: str | None) -> str | None:
    """Ignore les faux titres issus de formulations generiques de la question."""
    if not title_hint:
        return None

    normalized_question = normalize_text(question)
    has_explicit_title_label = bool(
        re.search(r"\b(?:titre|intitulee?|nommee?|appelee?)\b", normalized_question)
        or re.search(r"[\"«“].+[\"»”]", question)
    )
    if not has_explicit_title_label and re.search(
        r"\bvideo\b.*\b(?:avec|sur|a propos de|qui parle de)\b",
        normalized_question,
    ):
        return None
    return title_hint.strip() or None


def is_prior_video_comparison(question: str) -> bool:
    normalized = normalize_text(question)
    has_history_reference = bool(
        re.search(r"\b(?:les?|des?|leurs?)\s+(?:\d+|deux|trois)\b", normalized)
        or re.search(r"\b(?:ces|celles|ceux|eux|laquelle|lequel|parmi|entre)\b", normalized)
    )
    has_comparison = bool(
        re.search(r"\b(?:plus|moins|meilleur|meilleure|compare|comparatif|laquelle|lequel)\b", normalized)
        or re.search(r"\b(?:vues?|likes?|commentaires?|statistiques?|stats?)\b", normalized)
    )
    return has_history_reference and has_comparison


def select_reformulation_history(
    question: str,
    history_items: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Priorise les échanges utiles aux comparaisons elliptiques."""
    if not is_prior_video_comparison(question):
        return history_items

    user_indexes = [
        index
        for index, item in enumerate(history_items)
        if item.get("role") == "user"
    ]
    if not user_indexes:
        return history_items

    latest_user_index = user_indexes[-1]
    latest_user = normalize_text(history_items[latest_user_index].get("text", ""))
    latest_exchange_has_comparison_pair = bool(
        re.search(r"\bentre\b.+\bet\b", latest_user)
        or re.search(r"\bcompare\b.+\b(?:a|avec|et)\b", latest_user)
        or re.search(r"\bpoints? communs?\b.+\b(?:avec|entre|et)\b", latest_user)
    )
    if latest_exchange_has_comparison_pair:
        return history_items[latest_user_index:]

    # Une comparaison comme « compare leurs deux vidéos » peut dépendre des deux
    # derniers échanges, chacun ayant introduit une personne différente.
    if len(user_indexes) >= 2:
        return history_items[user_indexes[-2]:]
    return history_items[latest_user_index:]


def build_question_reformulation_prompt(
    question: str,
    history_items: list[dict[str, str]],
    system_prompt_override: str | None = None,
    memory_context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    history = "\n\n".join(
        f"{item['role']}: {item['text']}" for item in history_items
    )
    default_system_prompt = """Reformule le dernier message utilisateur sans y répondre.
Indique dans `follow_up` s'il dépend de l'historique ; il peut aussi changer de sujet.

Si nécessaire, remplace tout pronom, ordinal ou référence implicite par le nom, titre
ou objet exact. Priorité au dernier échange ; ne consulte les précédents que s'il ne
suffit pas. Conserve tous les référents réellement demandés.

Ne change pas le sens. `reformulated_question` doit être concis, autonome, en texte normal, sans Markdown.

Test obligatoire : on doit pouvoir lire la question reformulée sans son historique et la comprendre."""
    system_prompt = (system_prompt_override or "").strip() or default_system_prompt
    memory = memory_context or {}
    active_topic = memory.get("active_topic") or {}
    related_topics = memory.get("related_topics") or []
    episodes = memory.get("episodes") or []
    memory_sections: list[str] = []
    if active_topic:
        memory_sections.append(
            "Sujet actif (résumé compact, prioritaire pour les pronoms singuliers) :\n"
            + str(active_topic)
        )
    if related_topics:
        rendered_topics = "\n".join(
            f"Sujet {index} : {topic.get('summary', '')}"
            for index, topic in enumerate(related_topics, start=1)
            if topic.get("summary")
        )
        if rendered_topics:
            memory_sections.append(
                "Sujets proches récupérés par similarité sémantique (aide seulement si nécessaire) :\n"
                + rendered_topics
            )
    if episodes:
        rendered_episodes = "\n\n".join(
            f"Épisode {index} :\n{episode.get('content', '')}"
            for index, episode in enumerate(episodes, start=1)
        )
        memory_sections.append(
            "Épisodes récupérés de la mémoire longue (aide seulement si nécessaire) :\n"
            + rendered_episodes
        )
    memory_text = "\n\n".join(memory_sections)
    user_prompt = (
        f"Message actuel : {question}\n\n"
        "Historique récent (du plus vieux au plus récent ; le dernier bloc est "
        f"prioritaire) :\n\n{history or '(vide)'}"
    )
    if memory_text:
        user_prompt += f"\n\n{memory_text}"
    return system_prompt, user_prompt


def compact_reformulation_history(
    history_items: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Conserve un contexte récent et borné pour éviter les anciens référents parasites."""
    compacted: list[dict[str, str]] = []
    for item in history_items[-REFORMULATION_HISTORY_MAX_MESSAGES:]:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if len(text) > REFORMULATION_HISTORY_MAX_CHARS_PER_MESSAGE:
            tail_length = 400
            head_length = REFORMULATION_HISTORY_MAX_CHARS_PER_MESSAGE - tail_length
            text = f"{text[:head_length].rstrip()}\n[…]\n{text[-tail_length:].lstrip()}"
        compacted.append(
            {
                "role": str(item.get("role") or "user"),
                "text": text,
            }
        )
    return compacted


def repair_video_clarification_follow_up(
    question: str,
    history_items: list[dict[str, str]],
) -> str | None:
    """Preserve l'intention initiale quand l'utilisateur identifie une vidéo demandée."""
    normalized_question = normalize_text(question).strip()
    if not re.match(r"^(?:la|le|cette|ce|une|un)\s+video\b", normalized_question):
        return None

    last_assistant = next(
        (item["text"] for item in reversed(history_items) if item.get("role") == "assistant"),
        "",
    )
    if "de quelle video" not in normalize_text(last_assistant):
        return None

    previous_user = next(
        (item["text"] for item in reversed(history_items) if item.get("role") == "user"),
        "",
    ).strip()
    if not previous_user:
        return None

    return (
        f"{previous_user}\n"
        f"Reference video precisée par l'utilisateur : {question.strip()}"
    )


def is_obvious_follow_up(
    question: str,
    history_items: list[dict[str, str]],
) -> bool:
    """Repère les relances elliptiques que le modèle ne doit pas déclarer autonomes."""
    if not history_items:
        return False
    normalized = normalize_text(question).strip()
    return bool(
        re.match(r"^(?:et|aussi|pareil|idem)\b", normalized)
        or re.match(
            r"^(?:qui|quel(?:le|s|les)?)\b.*\b(?:son|sa|ses|leur|leurs)\b",
            normalized,
        )
        or re.match(
            r"^(?:et )?(?:pour|avec|chez) (?:lui|elle|eux|elles|celui|celle)\b",
            normalized,
        )
    )


def reformulate_question(
    question: str,
    conversation_id: int | None,
    client: LLMClientProtocol | None,
    model: str = DEFAULT_REFORMULATION_MODEL,
    system_prompt_override: str | None = None,
    *,
    history_override: list[dict[str, str]] | None = None,
    memory_context: dict[str, Any] | None = None,
    phase: str = "single_pass",
) -> tuple[str, dict[str, Any]]:
    """Rend une relance autonome avant le planner, sans modifier le message stocké."""
    if history_override is None:
        source_history_items, history_trace = fetch_conversation_history(
            conversation_id,
            limit=REFORMULATION_HISTORY_EXCHANGES,
        )
    else:
        source_history_items = history_override
        history_trace = {"applied": True, "reason": "memory_immediate_history", "message_count": len(history_override)}
    history_items = compact_reformulation_history(source_history_items)
    prompt_history_items = select_reformulation_history(question, history_items)
    history_trace = {
        **history_trace,
        "source_message_count": len(source_history_items),
        "message_count": len(history_items),
        "prompt_message_count": len(prompt_history_items),
    }
    trace: dict[str, Any] = {
        "applied": False,
        "original_question": question,
        "reformulated_question": question,
        "history_message_count": len(history_items),
        "history": history_trace,
        "phase": phase,
        "memory": {
            "active_topic": bool((memory_context or {}).get("active_topic")),
            "episode_count": len((memory_context or {}).get("episodes") or []),
        },
    }
    if client is None:
        trace["reason"] = "no_openai_client"
        return question, trace

    system_prompt, user_prompt = build_question_reformulation_prompt(
        question,
        prompt_history_items,
        system_prompt_override,
        memory_context,
    )
    reformulation_prompt = f"{system_prompt}\n\n{user_prompt}"
    reformulation_input = [
        {"role": "user", "content": reformulation_prompt},
    ]
    trace["prompt"] = json.dumps(
        {"model": model, "input": reformulation_input},
        ensure_ascii=False,
    )
    try:
        response = client.responses.create(
            model=model,
            input=reformulation_input,
            response_schema=REFORMULATION_RESPONSE_SCHEMA,
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

        repaired = repair_video_clarification_follow_up(question, history_items)
        if repaired:
            reformulated = repaired
            follow_up = True
            trace["reason"] = "video_followup_intent_preserved"
        elif not follow_up and is_obvious_follow_up(question, history_items):
            follow_up = True
            trace["reason"] = "deterministic_follow_up_detected"
        trace["follow_up"] = follow_up
        trace["applied"] = reformulated != question.strip()
        trace["reformulated_question"] = reformulated
        return reformulated or question, trace
    except Exception as exc:  # La recherche doit rester disponible si la reformulation échoue.
        trace["reason"] = "reformulation_error"
        trace["error"] = str(exc)
        return question, trace
