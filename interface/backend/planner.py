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


def _build_legacy_planner_prompt(question: str) -> tuple[str, str]:
    system_prompt = (
        "Tu es le planner d'un assistant conversationnel video. "
        "Ton role est uniquement de comprendre la demande utilisateur et de produire un plan d'execution strict en JSON. "
        "Tu ne dois jamais pretendre acceder aux donnees, ni repondre a la question utilisateur. "
        "Tu ne dois jamais mentionner ni utiliser de details techniques d'implementation. "
        "Retourne uniquement un JSON valide avec les cles exactes: "
        "route, sql_sub_intent, query_text, query_text_bm25, title_hint, persons, company, published_after, published_before, use_memory, use_rag, sql_main_source. "
        "route doit etre l'une de ces valeurs exactes: direct, rag, memory, multi_source. "
        "direct = reponse sans recherche. "
        "rag = recherche documentaire dans les contenus video et documents associes. "
        "Les demandes structurees sur les metadonnees, personnes ou transcripts utilisent aussi rag, avec sql_main_source=true et le sous-intent SQL adapte. "
        "memory = recherche dans l'historique conversationnel. "
        "multi_source = combinaison de plusieurs sources. "
        "sql_sub_intent peut etre null ou l'une de ces valeurs exactes: lookup, stats, description, transcript_verbatim, transcript_qa. "
        "Chaque sous-intent a un seul objectif: lookup recherche ou liste des videos, notamment par personne; stats retourne leurs statistiques; description retourne leur description; transcript_verbatim restitue la colonne transcript; transcript_qa repond ou produit une synthese a partir de la colonne transcript_timecodes_enrichi. "
        "La base contient les tables et informations suivantes: videos pour les titres, URL, descriptions et dates; les personnes avec leur nom et leur metier, poste, profession, fonction ou role; stats pour les vues, likes et commentaires; transcripts.transcript pour le texte sans timecodes; transcripts.transcript_timecodes_enrichi pour le texte enrichi avec timecodes. "
        "Utilise transcript_qa pour toute question demandant quand un propos apparait, a quel moment, a quelle minute, ou dans la video, dans quel passage ou avec quel timecode. "
        "Utilise transcript_verbatim uniquement pour restituer la transcription elle-meme. "
        "Si sql_main_source=false, sql_sub_intent doit etre null. "
        "Si route=memory, use_memory=true et use_rag=false et sql_main_source=false. "
        "Si route=rag, use_rag=true et use_memory=false et sql_main_source=false. "
        "Pour rechercher ou lister une video, demander son transcript complet, ses personnes ou ses metadonnees, route=rag avec sql_main_source=true et le sql_sub_intent adapte. "
        "Si route=multi_source, active au moins deux booleens parmi use_memory, use_rag, sql_main_source. "
        "Si tu identifies au moins deux personnes distinctes dans la question, remplis persons avec tous leurs noms et choisis obligatoirement route=multi_source. "
        "Exemples: 'bonjour' => route=direct. "
        "'qu'est-ce qui est dit sur Parcoursup ?' => route=rag. "
        "'trouve une video avec Andy Leveque' => route=rag, sql_main_source=true et sql_sub_intent=lookup. "
        "'donne le transcript complet de la video sur Parcoursup' => route=rag, sql_main_source=true et sql_sub_intent=transcript_verbatim. "
        "'resume la video intitulee Parcoursup' => route=rag, sql_main_source=true et sql_sub_intent=transcript_qa. "
        "'que dit cette video sur les stages ?' => route=rag, sql_main_source=true et sql_sub_intent=transcript_qa si la video est identifiee. "
        "'donne la description de cette video' => route=rag, sql_main_source=true et sql_sub_intent=description. "
        "'donne les statistiques de cette video' => route=rag, sql_main_source=true et sql_sub_intent=stats. "
        "'qui intervient dans cette video ?' => route=rag, sql_main_source=true et sql_sub_intent=lookup. "
        "'quel est le metier de Gabriel Dumy ?' => route=rag, sql_main_source=true, sql_sub_intent=lookup et persons=['Gabriel Dumy']. "
        "Si l'utilisateur demande un resume, une synthese, ce que dit quelqu'un dans une video, ou les points principaux sans identifier une video precise, utilise la recherche RAG sur les chunks du transcript avec sql_main_source=false. "
        "Si cette demande de contenu fournit le titre exact de la video, utilise transcript_qa avec sql_main_source=true, aussi bien pour un resume que pour une question. "
        "Exemple: 'Que dit Matthieu dans la video « Apporter ma pierre a l'edifice » ?' => route=rag, use_rag=true, sql_main_source=true, sql_sub_intent=transcript_qa et title_hint contient le titre fourni. "
        "Si la question fait reference a plusieurs videos deja evoquees dans l'historique avec des formulations comme 'les precedentes', 'ces videos', 'les 3', 'laquelle', 'la plus vue' ou 'compare', choisis route=multi_source avec use_memory=true et sql_main_source=true lorsque des statistiques ou metadonnees SQL sont necessaires. "
        "'que t'ai-je demande juste avant ?' => route=memory. "
        "'compare ce que dit la base et ce qu'on s'est deja dit' => route=multi_source. "
        "query_text doit contenir la reformulation utile pour la recherche semantique/vectorielle. "
        "query_text_bm25 doit etre une version tres courte orientee mots-cles. "
        "title_hint contient le titre ou fragment de titre explicitement fourni par l'utilisateur, sinon null. "
        "Ne remplis title_hint que si le titre est explicitement présenté comme un titre. Les formulations 'une video avec [personne/groupe]', 'une video sur [sujet]' et 'une video qui parle de [sujet]' ne sont jamais un title_hint : dans ces cas, title_hint=null. "
        "Exemple: 'j'ai besoin de la description de la video avec les alumnis' => title_hint=null et sql_sub_intent=description. "
        "query_text_bm25 ne doit contenir que des noms propres, acronymes, entites nommees, termes metier ou mots-cles concrets. "
        "Pour direct ou memory, query_text peut etre proche de la question brute. "
        "Pour rag ou sql, evite les verbes, les questions naturelles, les reformulations longues, les mots vides et les termes generiques comme "
        "'trouver', 'identifier', 'expliquer', 'parler', 'video', 'contenu', 'personne', 'role'. "
        "Si la question porte sur une personne nommee Andy Leveque, query_text_bm25 doit ressembler a 'Andy Leveque' et pas a une phrase. "
        "persons contient toutes les personnes explicitement identifiees dans la question, sinon un tableau vide. "
        "company contient toutes les entreprises explicitement identifiees dans la question, sinon un tableau vide. "
        "Les dates peuvent etre null."
    )
    return system_prompt, question


def build_planner_prompt(question: str) -> tuple[str, str]:
    system_prompt = (
        "Tu planifies la requete d'un assistant video sans y repondre. Retourne uniquement un objet JSON avec exactement ces cles: "
        "route, sql_sub_intent, query_text, query_text_bm25, title_hint, persons, company, published_after, published_before. "
        "Routes: direct=sans recherche; rag=recherche dans les videos; memory=historique conversationnel; "
        "multi_source=historique combine aux videos. "
        "sql_sub_intent vaut null pour une recherche semantique, sinon une valeur parmi: "
        "lookup (trouver ou lister des videos, notamment par personne, avec les noms et informations de metier, poste ou fonction disponibles), stats (vues, likes, commentaires), description, "
        "transcript_verbatim (transcription complete sans timecodes, colonne transcripts.transcript), "
        "transcript_qa (question, resume ou localisation temporelle comme 'a quel moment' dans une video identifiee, colonne transcripts.transcript_timecodes_enrichi). "
        "Une question generale sur le contenu utilise route=rag et sql_sub_intent=null. "
        "Une video identifiee par son titre avec une question sur son contenu utilise transcript_qa. "
        "Une demande de transcript complet utilise transcript_verbatim. "
        "title_hint contient uniquement un titre explicitement presente comme tel; une video avec une personne ou sur un sujet n'est pas un titre. "
        "persons contient toutes les personnes explicitement identifiees dans la question, qu'elles soient ou non deja connues comme intervenantes. "
        "company contient toutes les entreprises explicitement identifiees dans la question, sinon un tableau vide. "
        "Si persons contient au moins deux personnes distinctes, route doit obligatoirement valoir multi_source. "
        "query_text est adapte a la recherche semantique. query_text_bm25 est court et ne garde que noms propres, acronymes et mots-cles concrets. "
        "Les champs inconnus sont null, sauf persons et company qui sont toujours des tableaux."
    )
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
        "video_lookup": "lookup",
        "video_transcript": "transcript_verbatim",
        "video_description": "description",
        "video_stats": "stats",
    }
    sql_intents = {
        "lookup",
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
        payload["sql_sub_intent"] = "lookup"

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
    route = str(normalized.get("route") or "").strip()
    sql_sub_intent = str(normalized.get("sql_sub_intent") or "").strip() or None
    legacy_intent_map = {
        "video_lookup": "lookup",
        "video_transcript": "transcript_verbatim",
        "video_description": "description",
        "video_stats": "stats",
    }
    sql_intents = {
        "lookup",
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
        sql_sub_intent = legacy_intent_map.get(sql_sub_intent, sql_sub_intent) or "lookup"
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
            planner_plan.sql_sub_intent == "lookup"
            and not (planner_plan.persons or planner_plan.company)
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


def run_planner(question: str, client: OpenAI | None) -> tuple[PlannerPlan, str | None, str | None, bool]:
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
        )
        derive_plan_sources(fallback)
        return fallback, raw_prompt, json.dumps(fallback.model_dump(), ensure_ascii=False), False

    response = client.responses.create(model=DEFAULT_PLANNER_MODEL, input=planner_input)
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
        company=planner_plan.company,
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

    if planner_plan.persons or planner_plan.company:
        planner_plan.sql_sub_intent = "lookup"
        derive_plan_sources(planner_plan)
        return

    if has_temporal_transcript_request(question):
        planner_plan.sql_sub_intent = "transcript_qa"
        derive_plan_sources(planner_plan)
        return

    if has_person_title_request(question):
        planner_plan.sql_sub_intent = "lookup"
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
        planner_plan.sql_sub_intent = "lookup"
    else:
        planner_plan.sql_sub_intent = "lookup"
    derive_plan_sources(planner_plan)


def has_structured_sql_filters(query: ExecutionPlan) -> bool:
    return any(
        [
            bool(query.title_hint),
            bool(query.persons),
            bool(query.company),
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
    ambiguous_candidates: list[str] = []
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
            ambiguous_candidates.append(candidate)
            for score, person in close_matches:
                if person not in suggestions:
                    suggestions.append(person)
                suggestion_scores[person] = max(suggestion_scores.get(person, 0.0), score)
        else:
            unresolved_candidates.append(candidate)

    if ambiguous_candidates:
        unique_suggestions = suggestions[:3]
        if len(unique_suggestions) == 1 and not unresolved_candidates:
            person = unique_suggestions[0]
            if person not in resolved:
                resolved.append(person)
            return resolved, {
                "applied": True,
                "ambiguous": False,
                "requested": candidates,
                "suggestions": unique_suggestions,
                "suggestion_scores": [
                    {"person": person, "score": round(suggestion_scores[person], 3)}
                ],
                "auto_resolved": True,
            }
        return [], {
            "applied": True,
            "ambiguous": True,
            "requested": candidates,
            "ambiguous_requests": ambiguous_candidates,
            "suggestions": unique_suggestions,
            "suggestion_scores": [
                {"person": person, "score": round(suggestion_scores[person], 3)}
                for person in unique_suggestions
            ],
            "auto_resolved": False,
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
            "suggestions": [],
            "suggestion_scores": [],
            "auto_resolved": False,
            "fallback_to_transcripts": True,
        }

    return resolved, {
        "applied": True,
        "ambiguous": False,
        "requested": candidates,
        "suggestions": [],
        "suggestion_scores": [],
        "auto_resolved": False,
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
        "reformulated_question doit obligatoirement être une vraie question autonome, "
        "avec une formulation interrogative et un point d'interrogation final. "
        "Ne renvoie jamais une réponse, une confirmation, une phrase déclarative ou "
        "une reformulation comme 'Oui, je parle bien de ...'. "
        "Si le message utilisateur est une confirmation ou une précision, transforme-le "
        "en question complète en conservant l'intention de la demande précédente. "
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
        {"model": DEFAULT_REFORMULATION_MODEL, "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]},
        ensure_ascii=False,
    )
    try:
        response = client.responses.create(
            model=DEFAULT_REFORMULATION_MODEL,
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
