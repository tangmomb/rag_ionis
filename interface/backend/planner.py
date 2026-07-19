from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from openai import OpenAI

from interface.backend.config import DEFAULT_BM25_LIMIT, DEFAULT_FINAL_K, DEFAULT_PLANNER_MODEL
from interface.backend.database import connect_database, fetch_conversation_memory
from interface.backend.schemas import ExecutionPlan, PlannerPlan, RagRequest
from interface.backend.utilities import normalize_text, safe_json_loads, serialize_openai_response


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
        "sql_sub_intent peut etre null ou l'une de ces valeurs exactes: video_lookup, video_transcript, video_description, video_stats. "
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
        "'donne la description de cette video' => route=rag, sql_main_source=true et sql_sub_intent=video_description. "
        "'donne les statistiques de cette video' => route=rag, sql_main_source=true et sql_sub_intent=video_stats. "
        "Si l'utilisateur demande un resume, une synthese, ce que dit quelqu'un dans une video, ou les points principaux, utilise la recherche RAG sur le transcript. "
        "Si la question fait reference a plusieurs videos deja evoquees dans l'historique avec des formulations comme 'les precedentes', 'ces videos', 'les 3', 'laquelle', 'la plus vue' ou 'compare', choisis route=multi_source avec use_memory=true et sql_main_source=true lorsque des statistiques ou metadonnees SQL sont necessaires. "
        "'que t'ai-je demande juste avant ?' => route=memory. "
        "'compare ce que dit la base et ce qu'on s'est deja dit' => route=multi_source. "
        "query_text doit contenir la reformulation utile pour la recherche semantique/vectorielle. "
        "query_text_bm25 doit etre une version tres courte orientee mots-cles. "
        "title_hint contient le titre ou fragment de titre explicitement fourni par l'utilisateur, sinon null. "
        "Ne remplis title_hint que si le titre est explicitement présenté comme un titre. Les formulations 'une video avec [personne/groupe]', 'une video sur [sujet]' et 'une video qui parle de [sujet]' ne sont jamais un title_hint : dans ces cas, title_hint=null. "
        "Exemple: 'j'ai besoin de la description de la video avec les alumnis' => title_hint=null et sql_sub_intent=video_description. "
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

    legacy_sql_intents = {"video_lookup", "video_transcript", "video_description", "video_stats"}
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


def build_execution_plan(
    payload: RagRequest,
    planner_plan: PlannerPlan,
    video_ids: list[int] | None = None,
) -> ExecutionPlan:
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
        video_ids=video_ids or [],
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
            r"\b(?:url|lien|titre|description|descriptif|statistiques?|stats?|vues?|likes?|commentaires?|date de publication|publie|publiee|publiees|"
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
    if re.search(r"\b(?:statistiques?|stats?|vues?|likes?|commentaires?)\b", normalized):
        planner_plan.sql_sub_intent = "video_stats"
    elif re.search(r"\b(?:description|descriptif|decris)\b", normalized):
        planner_plan.sql_sub_intent = "video_description"
    elif any(term in normalized for term in ("transcript", "transcription", "verbatim", "timecode", "sous-titre")):
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


def resolve_speaker_filters(
    question: str,
    planner_plan: PlannerPlan,
) -> tuple[list[str], dict[str, Any]]:
    """Ne conserve que les noms réellement présents dans la table speakers."""
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
        return [], {"applied": False, "ambiguous": False, "requested": [], "suggestions": []}

    resolved: list[str] = []
    suggestions: list[str] = []
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
            database_speakers = [str(row[0]).strip() for row in cursor.fetchall()]

    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        exact_matches = [
            speaker for speaker in database_speakers
            if normalize_text(speaker) == normalized_candidate
        ]
        if exact_matches:
            for speaker in exact_matches:
                if speaker not in resolved:
                    resolved.append(speaker)
            continue

        candidate_tokens = normalized_candidate.split()

        def speaker_similarity(speaker: str) -> float:
            normalized_speaker = normalize_text(speaker)
            whole_score = SequenceMatcher(None, normalized_candidate, normalized_speaker).ratio()
            speaker_tokens = normalized_speaker.split()
            token_score = max(
                (
                    SequenceMatcher(None, candidate_token, speaker_token).ratio()
                    for candidate_token in candidate_tokens
                    for speaker_token in speaker_tokens
                ),
                default=0.0,
            )
            return max(whole_score, token_score)

        ranked = sorted(
            [
                (
                    speaker_similarity(speaker),
                    speaker,
                )
                for speaker in database_speakers
            ],
            reverse=True,
        )
        close_matches = [speaker for score, speaker in ranked if score >= 0.58][:3]
        if close_matches:
            ambiguous_candidates.append(candidate)
            for speaker in close_matches:
                if speaker not in suggestions:
                    suggestions.append(speaker)
        else:
            unresolved_candidates.append(candidate)

    if ambiguous_candidates:
        return [], {
            "applied": True,
            "ambiguous": True,
            "requested": candidates,
            "ambiguous_requests": ambiguous_candidates,
            "suggestions": suggestions[:3],
            "message": (
                "Vous parlez de " + " ou de ".join(suggestions[:3]) + " ?"
                if suggestions
                else "Peux-tu préciser le nom de l'intervenant ?"
            ),
        }

    if unresolved_candidates:
        return [], {
            "applied": True,
            "ambiguous": True,
            "requested": candidates,
            "ambiguous_requests": unresolved_candidates,
            "suggestions": [],
            "message": "Je ne trouve aucun nom proche dans la base. Peux-tu préciser le nom de l'intervenant ?",
        }

    return resolved, {
        "applied": True,
        "ambiguous": False,
        "requested": candidates,
        "suggestions": [],
    }


def extract_video_title_hint(question: str) -> str | None:
    """Extrait un titre explicitement fourni par l'utilisateur pour le SQL vidéo."""
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
