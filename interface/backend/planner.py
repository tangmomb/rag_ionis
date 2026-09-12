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
    "description": "Plan de requête : décider route et analytics en premier, puis remplir les paramètres associés et les champs de recherche.",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["direct", "search"],
            "description": "Première décision : direct pour un message uniquement social, search pour une question documentaire.",
        },
        "analytics": {
            "type": "boolean",
            "description": "Deuxième décision : false pour le contenu des vidéos ; true uniquement pour les statistiques ou métadonnées des vidéos.",
        },
        "analytics_scope": {
            "anyOf": [
                {"type": "string", "enum": ["global", "specific"]},
                {"type": "null"},
            ],
        },
        "analytics_metric": {
            "anyOf": [{"type": "string", "enum": ["all", "views", "likes", "comments"]}, {"type": "null"}],
        },
        "analytics_order": {
            "anyOf": [{"type": "string", "enum": ["asc", "desc"]}, {"type": "null"}],
        },
        "analytics_rank_start": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": 100}, {"type": "null"}]},
        "analytics_rank_end": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": 100}, {"type": "null"}]},
        "query_text": {"type": "string"},
        "query_text_bm25": {"type": "string"},
        "title_hints": {
            "type": "array",
            "items": {"type": "string"},
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
        "analytics",
        "analytics_scope",
        "analytics_metric",
        "analytics_order",
        "analytics_rank_start",
        "analytics_rank_end",
        "query_text",
        "query_text_bm25",
        "title_hints",
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
        "topic": {"type": "string"},
    },
    "required": ["follow_up", "reformulated_question", "topic"],
    "additionalProperties": False,
}

FINAL_REFORMULATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reformulated_question": {"type": "string"},
    },
    "required": ["reformulated_question"],
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
        "Tu planifies la requête d'un assistant RAG sans y répondre.\n\n"
        "1. Première étape : choisir route et analytics.\n"
        "- route='direct' si la question ou le message est une salutation ou une formule de politesse, sans demande documentaire ; sinon 'search'.\n"
        "- analytics=false pour le contenu des vidéos : propos, questions, identité, métier, résumé, comparaison. Également false pour 'direct'.\n"
        "- analytics=true uniquement pour les statistiques des vidéos (vues, likes, commentaires, comptages, classements) ou leurs métadonnées (publication, durée, type, sous-titres).\n"
        "Une personne, un titre, un filtre de publication ou un chiffre cité dans un entretien ne justifient pas analytics.\n"
        "Exemples : « Qui est Lou Ann ? », « Quelles questions pose-t-on à Fadila ? » → search/false ; « Combien de vues a sa vidéo ? » → search/true.\n\n"
        "2. Remplir les paramètres analytics.\n"
        "- Si analytics=false : tous les champs analytics_* valent null.\n"
        "- analytics_scope='global' est autorisé uniquement si title_hints, persons et companies sont tous vides : la statistique porte alors sur tout le corpus.\n"
        "- Dès qu'au moins un élément est présent dans title_hints, persons ou companies, analytics_scope doit être 'specific', même si la question demande un classement ou « le plus de vues ».\n"
        "- Scope specific : analytics_metric, analytics_order, analytics_rank_start et analytics_rank_end valent null.\n"
        "- Scope global avec classement : analytics_metric='views', 'likes' ou 'comments' ; analytics_order='desc' pour les plus élevés, 'asc' pour les moins élevés ; analytics_rank_start/end délimitent les rangs demandés (top 5 : 1 à 5).\n"
        "- Scope global sans classement : analytics_metric='all', analytics_order=null, rangs 1 à 3.\n\n"
        "3. Préparer la recherche sans changer l'intention.\n"
        "- query_text : question autonome pour la recherche sémantique.\n"
        "- query_text_bm25 : mots-clés courts et précis.\n\n"
        "4. Extraire les filtres explicites.\n"
        "- title_hints : titres de vidéos ; persons et companies : personnes ou entreprises mentionnées.\n"
        "- published_after/before : dates de publication au format YYYY-MM-DD, jamais les dates évoquées dans l'entretien.\n"
        "- Valeurs absentes : [] pour les listes, null pour les dates.\n\n"
        "Retourne les clés du schéma JSON, route et analytics en premier. Utilise true, false et null sans guillemets et du texte normal, sans Markdown."
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
    """Validate the only planner choices retained by the current pipeline."""
    normalized = dict(payload)
    route = str(normalized.get("route") or "").strip()
    sql_sub_intent = str(normalized.get("sql_sub_intent") or "").strip() or None
    # Adapt the LLM boolean to the execution contract. Older saved plans and
    # custom prompts may still supply sql_sub_intent.
    if "analytics" in normalized:
        analytics = normalized.pop("analytics")
        if not isinstance(analytics, bool):
            raise ValueError("analytics doit être un booléen JSON")
        sql_sub_intent = "analytics" if analytics else None
    if route not in {"direct", "search"}:
        route = "search"
    if route == "direct":
        sql_sub_intent = None
    elif sql_sub_intent != "analytics":
        sql_sub_intent = None

    normalized["route"] = route
    normalized["sql_sub_intent"] = sql_sub_intent
    analytics_scope = str(normalized.get("analytics_scope") or "").strip().lower() or None
    normalized["analytics_scope"] = (
        analytics_scope if sql_sub_intent == "analytics" and analytics_scope in {"global", "specific"} else None
    )
    if normalized["analytics_scope"] == "global" and any(
        normalized.get(key) for key in ("title_hints", "persons", "companies")
    ):
        normalized["analytics_scope"] = "specific"
    if normalized["analytics_scope"] == "global":
        metric = str(normalized.get("analytics_metric") or "").strip().lower()
        order = str(normalized.get("analytics_order") or "").strip().lower() or None
        normalized["analytics_metric"] = metric if metric in {"all", "views", "likes", "comments"} else None
        normalized["analytics_order"] = order if order in {"asc", "desc"} else None
    else:
        for key in ("analytics_metric", "analytics_order", "analytics_rank_start", "analytics_rank_end"):
            normalized[key] = None
    for derived_key in ("use_sql", "use_rag", "sql_main_source"):
        normalized.pop(derived_key, None)
    return normalized


def derive_plan_sources(planner_plan: PlannerPlan) -> None:
    """Normalise les seules contraintes de source encore portées par le planner."""
    if planner_plan.route == "direct":
        planner_plan.sql_sub_intent = None


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
            route="search",
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
            route="search",
            query_text=question,
        )
        fallback._output_rejection_reason = "empty_llm_output"
        derive_plan_sources(fallback)
        return fallback, raw_prompt, raw_response, False

    try:
        parsed = normalize_planner_output(safe_json_loads(raw))
        if (
            parsed.get("sql_sub_intent") == "analytics"
            and parsed.get("analytics_scope") not in {"global", "specific"}
        ):
            raise ValueError("analytics_scope manquant ou invalide pour une intention analytics")
        if parsed.get("analytics_scope") == "global" and (
            parsed.get("analytics_metric") not in {"all", "views", "likes", "comments"}
            or not isinstance(parsed.get("analytics_rank_start"), int)
            or not isinstance(parsed.get("analytics_rank_end"), int)
            or (
                parsed.get("analytics_metric") != "all"
                and parsed.get("analytics_order") not in {"asc", "desc"}
            )
            or (
                parsed.get("analytics_metric") == "all"
                and parsed.get("analytics_order") is not None
            )
        ):
            raise ValueError("fenêtre de classement manquante ou invalide pour une intention analytics globale")
        if not parsed.get("route"):
            parsed["route"] = "search"
        if not parsed.get("query_text"):
            parsed["query_text"] = question
        validated = PlannerPlan.model_validate(parsed)
        derive_plan_sources(validated)
        return validated, raw_prompt, raw_response, True
    except Exception as exc:
        fallback = PlannerPlan(
            route="search",
            query_text=question,
        )
        fallback._output_rejection_reason = (
            "analytics_scope_missing"
            if "analytics_scope manquant ou invalide" in str(exc)
            else "invalid_llm_plan"
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

    sql_search = (
        planner_plan.sql_sub_intent == "analytics"
        or planner_plan.description_requested
    )
    return ExecutionPlan(
        route=(
            "direct"
            if planner_plan.route == "direct"
            else "sql_search"
            if sql_search
            else "vector_search"
        ),
        sql_sub_intent=planner_plan.sql_sub_intent,
        analytics_scope=planner_plan.analytics_scope,
        analytics_metric=planner_plan.analytics_metric,
        analytics_order=planner_plan.analytics_order,
        analytics_rank_start=planner_plan.analytics_rank_start,
        analytics_rank_end=planner_plan.analytics_rank_end,
        raw_question=payload.question,
        query_text=(planner_plan.query_text or payload.question).strip() or payload.question,
        query_text_bm25=bm25_query,
        title_hints=planner_plan.title_hints,
        persons=planner_plan.persons,
        companies=planner_plan.companies,
        published_after=planner_plan.published_after,
        published_before=planner_plan.published_before,
        description_requested=planner_plan.description_requested,
        top_k=None if sql_search else DEFAULT_BM25_LIMIT,
        final_k=None if sql_search else DEFAULT_FINAL_K,
    )


def has_explicit_sql_request(question: str) -> bool:
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
            planner_plan.route = "search"
            policy_correction = "direct_with_entities_to_rag"
        elif not is_social_message(question):
            planner_plan.route = "search"
            policy_correction = "direct_non_social_to_rag"
        else:
            planner_plan.sql_sub_intent = None
            derive_plan_sources(planner_plan)
            return None

    planner_plan.description_requested = bool(
        re.search(r"\b(?:description|descriptif|decris)\b", normalize_text(question))
    )
    if planner_plan.sql_sub_intent == "analytics" or has_analytics_request(question):
        planner_plan.sql_sub_intent = "analytics"
        planner_plan.analytics_scope = planner_plan.analytics_scope or "specific"
        derive_plan_sources(planner_plan)
        return policy_correction

    planner_plan.sql_sub_intent = None
    derive_plan_sources(planner_plan)
    return policy_correction


def has_sql_filters(query: ExecutionPlan) -> bool:
    return any(
        [
            bool(query.title_hints),
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


def resolve_title_hints(title_hints: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Resolve every explicit title hint while preserving their input order."""
    resolved_titles: list[str] = []
    resolutions: list[dict[str, Any]] = []
    for title_hint in title_hints:
        resolved, resolution = resolve_title_hint(title_hint)
        resolutions.append(resolution)
        if resolved and resolved not in resolved_titles:
            resolved_titles.append(resolved)
    return resolved_titles, {"requested": title_hints, "resolutions": resolutions}


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


def sanitize_video_title_hints(question: str, title_hints: list[str]) -> list[str]:
    """Keep distinct, explicit title hints only."""
    sanitized: list[str] = []
    for title_hint in title_hints:
        title = sanitize_video_title_hint(question, str(title_hint or ""))
        if title and title not in sanitized:
            sanitized.append(title)
    return sanitized


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
    *,
    include_follow_up: bool = True,
) -> tuple[str, str]:
    history = "\n\n".join(
        f"{item['role']}: {item['text']}" for item in history_items
    )
    follow_up_instruction = (
        "ÉTAPE 1 — Avant toute reformulation, décide `follow_up` : `true` si le message dépend de l'historique ou du sujet actif ; "
        "`false` s'il ouvre un nouveau sujet ; il peut aussi changer de sujet. Indique `topic`, puis reformule.\n\n"
        if include_follow_up
        else ""
    )
    default_system_prompt = f"""{follow_up_instruction}Reformule le dernier message utilisateur en une question autonome, sans y répondre.
Le destinataire ne reçoit que `reformulated_question` : il ne voit ni historique, ni mémoire, ni `topic`.

Résous pronoms, ordinaux et références implicites. Conserve tous les référents réellement demandés, l'intention et les contraintes, sans inventer de titre ni de nom.

« La chaîne » désigne toujours la chaîne interrogée : ne la rattache jamais à une personne, entreprise ou autre entité.

Priorité au dernier échange et à `current_topic`. Si nécessaire, consulte aussi `previous_topics` et leurs résumés. N'ajoute aucun sujet ancien sans lien ; conserve l'incertitude sans inventer.

Exemple : sujet précédent = entretien d'Alice ; sujet courant = métier de Bruno. « laquelle des 2 a le plus de vues ? » devient « Entre la vidéo d'Alice et celle de Bruno, laquelle a le plus de vues ? ». « Laquelle des deux vidéos a le plus de vues ? » n'est pas autonome.

Si le message est déjà autonome, conserve-le. `reformulated_question` doit être concis, en texte normal, sans Markdown. L'autonomie prime sur la concision.

Test obligatoire : sans accès à la conversation, le destinataire peut-il identifier chaque objet de la demande et comprendre la demande ? Sinon, complète la question avec les référents disponibles."""
    custom_system_prompt = (system_prompt_override or "").strip()
    system_prompt = custom_system_prompt or default_system_prompt
    memory = memory_context or {}
    memory_sections: list[str] = []
    conversation_memory = memory.get("conversation_memory")
    if isinstance(conversation_memory, dict):
        memory_sections.append(
            "Mémoire complète de la conversation (JSON ; le sujet courant et ses messages sont prioritaires) :\n"
            + json.dumps(conversation_memory, ensure_ascii=False)
        )
    active_topic = memory.get("active_topic") or {}
    related_topics = memory.get("related_topics") or []
    episodes = memory.get("episodes") or []
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
    user_prompt = f"Message actuel : {question}"
    if history:
        user_prompt += (
            "\n\nHistorique récent (du plus vieux au plus récent ; le dernier bloc est "
            f"prioritaire) :\n\n{history}"
        )
    if custom_system_prompt and include_follow_up:
        user_prompt += (
            "\n\nContrat obligatoire : `follow_up` vaut true seulement si le message "
            "continue le sujet actif ; sinon il vaut false et `topic` nomme le nouveau sujet."
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

    include_follow_up = phase != "final"
    system_prompt, user_prompt = build_question_reformulation_prompt(
        question,
        prompt_history_items,
        system_prompt_override,
        memory_context,
        include_follow_up=include_follow_up,
    )
    reformulation_input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    trace["prompt"] = json.dumps(
        {"model": model, "input": reformulation_input},
        ensure_ascii=False,
    )
    try:
        response = client.responses.create(
            model=model,
            input=reformulation_input,
            response_schema=(
                REFORMULATION_RESPONSE_SCHEMA
                if include_follow_up
                else FINAL_REFORMULATION_RESPONSE_SCHEMA
            ),
        )
        trace["response_raw"] = serialize_openai_response(response)
        raw_output = (getattr(response, "output_text", "") or "").strip()
        if not raw_output:
            trace["reason"] = "empty_response"
            return question, trace
        # Évite qu'une réponse accidentellement multi-ligne devienne une nouvelle consigne.
        try:
            parsed = safe_json_loads(raw_output)
            reformulated = str(parsed.get("reformulated_question") or question).strip()
        except (TypeError, ValueError, json.JSONDecodeError):
            trace["reason"] = "invalid_json_response"
            return question, trace

        if include_follow_up:
            follow_up = bool(parsed.get("follow_up", False))
            trace["topic"] = str(parsed.get("topic") or "").strip()
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
