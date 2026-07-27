from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from openai import OpenAI

from interface.backend.config import (
    DEFAULT_BM25_LIMIT,
    DEFAULT_FINAL_K,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
)
from interface.backend.database import connect_database, fetch_conversation_memory
from interface.backend.schemas import ExecutionPlan, PlannerPlan, RagRequest
from interface.backend.utilities import normalize_text, safe_json_loads, serialize_openai_response


# La correction tolère une faute légère dans un prénom ou un nom, sans faire
# remonter des noms qui ne partagent qu'une syllabe courte.
PERSON_NAME_PART_SIMILARITY_THRESHOLD = 0.85
COMPANY_TITLE_SIMILARITY_THRESHOLD = 0.90


def build_planner_prompt(
    question: str,
    system_prompt_override: str | None = None,
) -> tuple[str, str]:
    default_system_prompt = (
        "Tu planifies la requete d'un assistant RAG sans y repondre. Retourne uniquement un objet JSON avec exactement ces cles: "
        "route, sql_sub_intent, query_text, query_text_bm25, title_hint, persons, companies, published_after, published_before. "
        "Première étape, identifier les personnes ou entreprises mentionnées dans la question. Les stocker dans les clés persons et companies sous forme de tableaux json."
        "Deuxième étape, identifier les dates de publication mentionnées dans la question. Les stocker dans published_after et published_before sous forme de chaînes ISO 8601 (YYYY-MM-DD). "
        "Troisième étape, identifier un titre de video mentionné dans la question. Le stocker dans title_hint. "
        "Quatrième étape, produire les clés query_text et query_text_bm25. query_text est la question reformulée pour la recherche RAG, c'est elle qui sera calculée pour l'embedding donc attention à son écriture sémantique. query_text_bm25 est la question reformulée pour la recherche BM25, elle doit être plus courte et plus directe, adaptée pour une recherche par mots-clés. "
        "Cinquième et dernière étatpe, choisir la stratégie pour répondre à la question via les clés route et sql_sub_intent. route peut être 'direct', 'rag' ou 'multi_source'. sql_sub_intent peut être 'specific_persons', 'stats', 'description', 'transcript_verbatim' ou 'null'."
        "route='direct' si la question ou message ne demande rien à propos de la base de données. route='rag' si la question concerne la base de données. route='multi_source' si tu as identifié plus d'une personne ou entreprise cumulées dans la question. (1 personne + 1 entreprise = 2)."
        "sql_sub_intent='specific_persons' si tu as identifié des personnes ou entreprises dans la question. sql_sub_intent='stats' si la question demande des statistiques sur une video. sql_sub_intent='description' si la question demande explicitement la description d'une video. sql_sub_intent='transcript_verbatim' si la question demande explicitement le transcript complet d'une video. sql_sub_intent='null' si la question ne demande pas explicitement de données structurées."

    )
    system_prompt = (system_prompt_override or "").strip() or default_system_prompt
    return system_prompt, question


def build_social_answer(question: str) -> str:
    lower = normalize_text(question).strip()
    if any(token in lower for token in ("merci",)):
        return "Avec plaisir. Si tu veux, je peux aussi t'aider a chercher une video ou repondre a une question sur la base."
    if any(token in lower for token in ("ca va", "ca roule", "comment ca va")):
        return "Ca va bien, merci. Je suis pret a t'aider sur la base video si tu veux."
    if any(token in lower for token in ("au revoir", "a bientot", "bonne journee", "bonne soiree")):
        return "A bientot."
    return "Bonjour. Je peux t'aider a trouver une video, un transcript, un resume ou repondre a une question a partir de la base."


def _normalize_legacy_planner_output(payload: dict[str, Any]) -> dict[str, Any]:
    route = str(payload.get("route") or "").strip()
    sql_sub_intent = str(payload.get("sql_sub_intent") or "").strip() or None

    if "sql_main_source" not in payload and "use_sql" in payload:
        payload["sql_main_source"] = bool(payload.get("use_sql"))
    payload.pop("use_sql", None)

    legacy_intent_map = {
        "video_lookup": "specific_persons",
        "lookup": "specific_persons",
        "video_transcript": "transcript_verbatim",
        "video_description": "description",
        "video_stats": "stats",
    }
    sql_intents = {
        "specific_persons",
        "stats",
        "description",
        "transcript_verbatim",
        "transcript_qa",
    }
    legacy_routes = {"rag_chunks": "rag", "sql_request": "rag", "social": "direct"}

    if route in legacy_intent_map or route in sql_intents:
        payload["route"] = "rag"
        payload["sql_sub_intent"] = legacy_intent_map.get(route, route)
        payload["sql_main_source"] = True
        payload["use_rag"] = True
        payload["use_memory"] = False
        return payload

    if sql_sub_intent in legacy_intent_map:
        payload["sql_sub_intent"] = legacy_intent_map[sql_sub_intent]
        sql_sub_intent = payload["sql_sub_intent"]

    if route in legacy_routes:
        payload["route"] = legacy_routes[route]
        route = payload["route"]

    if route == "sql":
        payload["route"] = "rag"
        payload["sql_main_source"] = True
        payload["use_rag"] = True
        payload["use_memory"] = False
        route = "rag"

    if route != "multi_source" and not payload.get("sql_main_source"):
        payload["sql_sub_intent"] = None
    elif sql_sub_intent not in sql_intents:
        payload["sql_sub_intent"] = "specific_persons"

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

    if route not in {"direct", "rag", "memory", "multi_source"}:
        payload["route"] = "rag"
        payload["use_memory"] = False
        payload["use_rag"] = True
        payload["sql_main_source"] = False
        return payload
    return payload


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
        "video_stats": "stats",
    }
    sql_intents = {
        "specific_persons",
        "stats",
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

    if route not in {"direct", "rag", "memory", "multi_source"}:
        route = "rag"
    if route in {"direct", "memory"}:
        sql_sub_intent = None
    elif sql_sub_intent not in sql_intents:
        sql_sub_intent = None

    normalized["route"] = route
    normalized["sql_sub_intent"] = sql_sub_intent
    for derived_key in ("use_sql", "use_memory", "use_rag", "sql_main_source"):
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
        planner_plan.use_memory = False
        planner_plan.use_rag = False
        planner_plan.sql_main_source = False
    elif planner_plan.route == "memory":
        planner_plan.sql_sub_intent = None
        planner_plan.use_memory = True
        planner_plan.use_rag = False
        planner_plan.sql_main_source = False
    elif planner_plan.route == "multi_source":
        planner_plan.use_memory = True
        planner_plan.use_rag = not has_sql_intent
        planner_plan.sql_main_source = has_sql_intent
    else:
        planner_plan.use_memory = False
        planner_plan.use_rag = True
        planner_plan.sql_main_source = has_sql_intent


def run_planner(
    question: str,
    client: OpenAI | None,
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

    response = client.responses.create(model=model, input=planner_input)
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
        use_memory=planner_plan.use_memory,
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


def apply_deterministic_sql_policy(question: str, planner_plan: PlannerPlan) -> None:
    """Empêche le planner de basculer arbitrairement la source SQL principale."""
    if planner_plan.route in {"direct", "memory"}:
        planner_plan.sql_sub_intent = None
        derive_plan_sources(planner_plan)
        return

    if planner_plan.persons or planner_plan.companies:
        planner_plan.sql_sub_intent = "specific_persons"
        derive_plan_sources(planner_plan)
        return

    if has_temporal_transcript_request(question):
        planner_plan.sql_sub_intent = "transcript_qa"
        derive_plan_sources(planner_plan)
        return

    if has_person_title_request(question):
        planner_plan.sql_sub_intent = "specific_persons"
        derive_plan_sources(planner_plan)
        return

    if has_document_content_request(question):
        if planner_plan.title_hint:
            planner_plan.sql_sub_intent = "transcript_qa"
        else:
            planner_plan.sql_sub_intent = None
        derive_plan_sources(planner_plan)
        return

    if not has_explicit_structured_sql_request(question):
        # Le planner peut conserver SQL pour une question video complexe.
        # Si la recherche structuree echoue, orchestrate_request tentera le RAG.
        derive_plan_sources(planner_plan)
        return

    normalized = normalize_text(question)
    if re.search(r"\b(?:statistiques?|stats?|vues?|likes?|commentaires?)\b", normalized):
        planner_plan.sql_sub_intent = "stats"
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
    """Ne conserve que les personnes réellement présentes dans la table SQL."""
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
            "suggestions": [],
            "suggestion_scores": [],
            "auto_resolved": False,
        }

    resolved: list[str] = []
    suggestions: list[str] = []
    suggestion_scores: dict[str, float] = {}
    auto_resolved_candidates: list[str] = []
    ambiguous_candidates: list[str] = []
    ambiguous_suggestions: list[str] = []
    ambiguous_suggestion_scores: dict[str, float] = {}
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

    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        exact_matches = [
            person for person in database_persons
            if normalize_text(person) == normalized_candidate
        ]
        if exact_matches:
            for person in exact_matches:
                if person not in resolved:
                    resolved.append(person)
            continue

        def person_similarity(person: str) -> float:
            # Les tirets font partie de la graphie d'un prénom composé, mais
            # l'utilisateur peut les omettre ("Lou Ann" / "Lou-Ann").
            candidate_parts = re.sub(r"[-'’]", " ", normalized_candidate).split()
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

            # Pour un nom complet, on aligne prénom et nom : pas de produit
            # croisé entre tous les mots (ex. "Ann" ne doit pas matcher
            # "Yannick" dans un autre nom).
            return max(
                SequenceMatcher(None, candidate_parts[0], person_parts[0]).ratio(),
                SequenceMatcher(None, candidate_parts[-1], person_parts[-1]).ratio(),
            )

        ranked = sorted(
            [
                (
                    person_similarity(person),
                    person,
                )
                for person in database_persons
            ],
            reverse=True,
        )
        close_matches = [
            (score, person)
            for score, person in ranked
            if score >= PERSON_NAME_PART_SIMILARITY_THRESHOLD
        ][:3]
        if close_matches:
            top_score = close_matches[0][0]
            best_matches = [
                (score, person)
                for score, person in close_matches
                if top_score - score <= 0.02
            ]
            if len(best_matches) == 1:
                score, person = best_matches[0]
                if person not in resolved:
                    resolved.append(person)
                if person not in suggestions:
                    suggestions.append(person)
                suggestion_scores[person] = max(suggestion_scores.get(person, 0.0), score)
                auto_resolved_candidates.append(candidate)
                continue

            ambiguous_candidates.append(candidate)
            for score, person in best_matches:
                if person not in ambiguous_suggestions:
                    ambiguous_suggestions.append(person)
                ambiguous_suggestion_scores[person] = max(
                    ambiguous_suggestion_scores.get(person, 0.0),
                    score,
                )
        else:
            unresolved_candidates.append(candidate)

    if ambiguous_candidates:
        unique_suggestions = ambiguous_suggestions[:3]
        return resolved, {
            "applied": True,
            "ambiguous": True,
            "requested": candidates,
            "ambiguous_requests": ambiguous_candidates,
            "suggestions": unique_suggestions,
            "suggestion_scores": [
                {
                    "person": person,
                    "score": round(ambiguous_suggestion_scores[person], 3),
                }
                for person in unique_suggestions
            ],
            "auto_resolved": bool(auto_resolved_candidates),
            "message": (
                "Vous parlez de " + " ou de ".join(unique_suggestions) + " ?"
                if unique_suggestions
                else "Peux-tu préciser le nom de l'intervenant ?"
            ),
        }

    if unresolved_candidates:
        return resolved, {
            "applied": True,
            "ambiguous": False,
            "requested": candidates,
            "unresolved_requests": unresolved_candidates,
            "suggestions": suggestions,
            "suggestion_scores": [
                {"person": person, "score": round(suggestion_scores[person], 3)}
                for person in suggestions
            ],
            "auto_resolved": bool(auto_resolved_candidates),
            "fallback_to_transcripts": True,
        }

    return resolved, {
        "applied": True,
        "ambiguous": False,
        "requested": candidates,
        "suggestions": suggestions,
        "suggestion_scores": [
            {"person": person, "score": round(suggestion_scores[person], 3)}
            for person in suggestions
        ],
        "auto_resolved": bool(auto_resolved_candidates),
    }


def resolve_company_filters(
    requested_companies: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Résout les entreprises vers des termes réellement contenus dans title."""
    candidates = [
        str(value).strip()
        for value in requested_companies
        if str(value).strip()
    ]
    if not candidates:
        return [], {
            "applied": False,
            "requested": [],
            "resolved": [],
            "matches": [],
            "unresolved": [],
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

    resolved: list[str] = []
    matches: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for candidate in candidates:
        candidate_tokens = re.findall(r"\w+", normalize_text(candidate))
        if not candidate_tokens:
            unresolved.append(candidate)
            continue

        phrase_size = len(candidate_tokens)
        ranked: list[tuple[float, str, str]] = []
        for title in database_titles:
            title_tokens = re.findall(r"\w+", normalize_text(title))
            for index in range(0, len(title_tokens) - phrase_size + 1):
                phrase = " ".join(title_tokens[index:index + phrase_size])
                score = SequenceMatcher(
                    None,
                    " ".join(candidate_tokens),
                    phrase,
                ).ratio()
                ranked.append((score, phrase, title))

        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] < COMPANY_TITLE_SIMILARITY_THRESHOLD:
            unresolved.append(candidate)
            continue

        score, matched_term, matched_title = ranked[0]
        if matched_term not in resolved:
            resolved.append(matched_term)
        matches.append(
            {
                "requested": candidate,
                "resolved": matched_term,
                "title": matched_title,
                "score": round(score, 3),
            }
        )

    return resolved, {
        "applied": True,
        "requested": candidates,
        "resolved": resolved,
        "matches": matches,
        "unresolved": unresolved,
        "similarity_threshold": COMPANY_TITLE_SIMILARITY_THRESHOLD,
    }


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


def is_memory_video_comparison(question: str) -> bool:
    normalized = normalize_text(question)
    has_memory_reference = bool(
        re.search(r"\b(?:les?|des?)\s+\d+\b", normalized)
        or re.search(r"\b(?:ces|celles|ceux|laquelle|lequel|parmi|entre)\b", normalized)
    )
    has_comparison = bool(
        re.search(r"\b(?:plus|moins|meilleur|meilleure|compare|comparatif|laquelle|lequel)\b", normalized)
        or re.search(r"\b(?:vues?|likes?|commentaires?|statistiques?|stats?)\b", normalized)
    )
    return has_memory_reference and has_comparison


def build_question_reformulation_prompt(
    question: str,
    memory_items: list[dict[str, str]],
    system_prompt_override: str | None = None,
) -> tuple[str, str]:
    history = "\n".join(
        f"{item['role']}: {item['text']}" for item in memory_items
    )
    default_system_prompt = (
        "Tu reformules le dernier message utilisateur sans y répondre. 1) indiquer si le message a besoin de l'historique de la conversation pour être compris. 2) si oui, reformule le message en incluant les informations pertinentes de l'historique pour que la question soit complètement autonome. Si non, reformule le message de manière propre et bien écrit sans changer son sens. Répond sous la forme d'un objet JSON avec exactement ces clés: follow_up (booléen), reformulated_question (string)."
    )
    system_prompt = (system_prompt_override or "").strip() or default_system_prompt
    user_prompt = f"Message actuel : {question}\n\nHistorique récent :\n{history or '(vide)'}"
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


def reformulate_question(
    question: str,
    conversation_id: int | None,
    client: OpenAI | None,
    model: str = DEFAULT_REFORMULATION_MODEL,
    system_prompt_override: str | None = None,
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

    system_prompt, user_prompt = build_question_reformulation_prompt(
        question,
        memory_items,
        system_prompt_override,
    )
    trace["prompt"] = json.dumps(
        {"model": model, "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]},
        ensure_ascii=False,
    )
    try:
        response = client.responses.create(
            model=model,
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
