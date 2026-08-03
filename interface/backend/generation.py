from __future__ import annotations

import json
import re
from typing import Any

from interface.backend.config import SOURCE_RELEVANCE_MIN
from interface.backend.llm_providers import LLMClientProtocol
from interface.backend.schemas import AnswerAction
from interface.backend.utilities import normalize_text, safe_json_loads, serialize_openai_response


FINAL_ANSWER_STYLE = (
    "Réponds directement à la question avec les éléments disponibles. "
    "Formate toujours la réponse en Markdown lisible, avec des paragraphes, listes ou tableaux lorsque cela améliore la clarté. "
    "N'introduis pas ta réponse par une formule comme « d'après les sources » "
    "ou « selon les documents ». "
    "Si ta réponse consiste uniquement à présenter des sources ou des vidéos, "
    "place une courte formule de politesse au début. "
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


DEFAULT_ANSWER_PROMPT_TEMPLATE = (
    "{route_instructions}\n\n"
    "{source_marker_instruction}\n\n"
    f"{ANSWER_ACTION_INSTRUCTION}\n\n"
    f"{FINAL_ANSWER_STYLE}"
)


def render_answer_system_prompt(
    prompt_template: str | None,
    *,
    route_instructions: str,
    source_marker_instruction: str = "",
) -> str:
    template = (prompt_template or "").strip() or DEFAULT_ANSWER_PROMPT_TEMPLATE
    replacements = {
        "{route_instructions}": route_instructions.strip(),
        "{source_marker_instruction}": source_marker_instruction.strip(),
        "{answer_action_instruction}": ANSWER_ACTION_INSTRUCTION.strip(),
        "{answer_style}": FINAL_ANSWER_STYLE.strip(),
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return re.sub(r"\n{3,}", "\n\n", rendered).strip()


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
    person_resolution = retrieval.get("person_resolution") or {}
    normalized_question = normalize_text(question)
    has_unresolved_video_reference = bool(
        re.search(r"\b(?:la|le|cette|ce|une|un)\s+video\b", normalized_question)
        or re.search(r"\bvideo\b.*\b(?:sur|de|a propos de)\b", normalized_question)
    )

    if person_resolution.get("ambiguous"):
        action_hint: AnswerAction = "clarify"
        reason = "ambiguous_person"
    elif not sources:
        action_hint: AnswerAction = "clarify" if has_unresolved_video_reference else "abstain"
        reason = "ambiguous_video_reference" if action_hint == "clarify" else "no_sources"
    elif cohere_scores and max(cohere_scores) < SOURCE_RELEVANCE_MIN:
        action_hint = "abstain"
        reason = "low_rerank_relevance"
    else:
        action_hint = "answer"
        reason = "sources_available"

    clarification_message = (
        "Peux-tu préciser le titre exact de la vidéo ou le sujet dont tu parles ?"
        if action_hint == "clarify"
        else None
    )
    message_source = (
        "person_resolution"
        if reason == "ambiguous_person"
        else "source_evaluation"
        if action_hint == "clarify"
        else None
    )
    selected_message = (
        person_resolution.get("message")
        if reason == "ambiguous_person"
        else clarification_message
    )

    return {
        "action_hint": action_hint,
        "reason": reason,
        "question": question,
        "source_count": len(sources),
        "top_cohere_relevance_score": max(cohere_scores) if cohere_scores else None,
        "source_scores": source_scores,
        "retrieval_mode": retrieval.get("retrieval_mode"),
        "message_source": message_source,
        "selected_message": selected_message,
        "clarification_message": clarification_message,
    }


def source_context_text(source: dict[str, Any]) -> str:
    parts = [str(source.get("text") or "").strip()]
    section_context = source.get("section_context") or {}
    global_context = source.get("global_context") or {}
    if section_context.get("text"):
        parts.append(f"Resume de section: {section_context['text']}")
    if global_context.get("text"):
        parts.append(f"Resume global: {global_context['text']}")
    return "\n".join(part for part in parts if part)


def generate_answer(
    client: LLMClientProtocol | None,
    question: str,
    answer_model: str | None,
    sources: list[dict[str, Any]],
    trace: dict[str, str] | None = None,
    prompt_template: str | None = None,
) -> str:
    if not sources:
        if trace is not None:
            trace["action"] = "abstain"
        return "Je n'ai trouve aucun chunk pertinent dans la base pour repondre a cette question."
    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer"
        return "\n\n".join(
            f"[S{index}] {source_context_text(source)}"
            for index, source in enumerate(sources, start=1)
        )

    context_blocks = []
    for index, source in enumerate(sources, start=1):
        section_context = source.get("section_context") or {}
        global_context = source.get("global_context") or {}
        context_blocks.append(
            "\n".join(
                [
                    f"Source {index} :",
                    f"Titre: {source['video_title']}",
                    f"URL: {source['video_url']}",
                    f"Chunk detail: {source['chunk_index']}",
                    f"Extrait pertinent: {source['text']}",
                    *(
                        [f"Resume de section: {section_context['text']}"]
                        if section_context.get("text")
                        else []
                    ),
                    *(
                        [f"Resume global: {global_context['text']}"]
                        if global_context.get("text")
                        else []
                    ),
                ]
            )
        )

    input_messages = [
            {
                "role": "system",
                "content": render_answer_system_prompt(
                    prompt_template,
                    route_instructions=(
                        "Tu es un assistant RAG. Réponds en français, de façon concise, "
                        "en t'appuyant uniquement sur les sources fournies."
                    ),
                    source_marker_instruction=SOURCE_MARKER_INSTRUCTION,
                ),
            },
            {
                "role": "user",
                "content": f"Question utilisateur: {question}\n\nSources pour répondre :\n\n" + "\n\n".join(context_blocks),
            },
        ]
    response = client.responses.create(model=answer_model, input=input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable.")


def generate_memory_answer(
    client: LLMClientProtocol | None,
    question: str,
    answer_model: str | None,
    memory_items: list[dict[str, str]],
    trace: dict[str, str] | None = None,
    prompt_template: str | None = None,
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
                "content": render_answer_system_prompt(
                    prompt_template,
                    route_instructions=(
                        "Tu réponds uniquement à partir de l'historique "
                        "de conversation fourni."
                    ),
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


def build_sql_sub_intent_prompt(sql_sub_intent: str | None) -> str:
    if sql_sub_intent == "stats":
        return (
            "Tu reponds a une demande de statistiques sur une video. "
            "Identifie la video correspondante et presente les dernieres statistiques disponibles : vues, likes, commentaires et date du snapshot. "
            "N'invente aucune valeur manquante et indique clairement lorsqu'une statistique n'est pas disponible. "
        )
    if sql_sub_intent == "description":
        return (
            "Tu reponds a une demande de description d'une video. "
            "Identifie la video a partir des resultats fournis. "
            "Presente la description de la video dans un paragraphe naturel et lisible. "
            "Ne recopie jamais la description brute seule et n'ajoute aucune information absente de la description. "
            "Il s'agit de restituer la description de la video, pas de la resumer ni de l'analyser. "
        )
    if sql_sub_intent == "transcript_verbatim":
        return (
            "Tu reponds a une demande de transcript de video. "
            "Identifie la video correspondante dans les resultats fournis. "
            "Restitue le transcript fidelement, sans le remplacer par un resume, sans inventer de contenu et sans ajouter d'analyse non demandee. "
        )
    if sql_sub_intent == "transcript_qa":
        return (
            "Tu reponds a une demande d'analyse, de synthese ou a une question sur le contenu d'une video en utilisant son transcript enrichi avec timecodes comme source. "
            "Respecte exactement l'operation demandee, synthetise les passages pertinents et ne restitue pas le transcript en entier. "
            "N'invente aucune information absente du transcript. "
        )
    return (
        "Tu reponds a une demande de recherche de videos dans les resultats structures fournis. "
        "Presente chaque video trouvee de maniere claire avec son titre et son lien. "
        "Si la question porte sur une personne, son poste ou sa fonction, utilise uniquement les informations d'intervenant fournies dans les resultats. "
        "Si plusieurs videos sont presentes, distingue-les nettement. "
        "Ne transforme pas une recherche de videos en description ou en resume. "
    )


def generate_multi_source_answer(
    client: LLMClientProtocol | None,
    question: str,
    answer_model: str | None,
    route_name: str,
    memory_items: list[dict[str, str]],
    sources: list[dict[str, Any]],
    sql_sub_intent: str | None = None,
    trace: dict[str, str] | None = None,
    prompt_template: str | None = None,
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
                    f"Source {index} :",
                    f"Titre: {source['video_title']}",
                    f"URL: {source['video_url']}",
                    f"Chunk: {source['chunk_index']}",
                    f"Texte: {source['text']}",
                ]
            )
        )
    source_block = "\n\n".join(source_blocks) or "Aucune source documentaire exploitable."
    source_marker_instruction = (
        "" if sql_sub_intent == "transcript_verbatim" else SOURCE_MARKER_INSTRUCTION + " "
    )

    input_messages = [
            {
                "role": "system",
                "content": render_answer_system_prompt(
                    prompt_template,
                    route_instructions=(
                        "Tu synthétises plusieurs sources pour répondre en français. "
                        "Distingue clairement ce qui vient de l'historique conversationnel "
                        "et ce qui vient de la base si utile. "
                        + build_sql_sub_intent_prompt(sql_sub_intent)
                    ),
                    source_marker_instruction=source_marker_instruction,
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nHistorique:\n{memory_block}\n\nSources pour répondre :\n{source_block}",
            },
        ]
    response = client.responses.create(model=answer_model, input=input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError(f"Le modele n'a pas renvoye de texte exploitable pour la route {route_name}.")


def generate_sql_answer(
    client: LLMClientProtocol | None,
    question: str,
    answer_model: str | None,
    sql_sub_intent: str | None,
    sources: list[dict[str, Any]],
    trace: dict[str, str] | None = None,
    prompt_template: str | None = None,
) -> str:
    if not sources:
        if trace is not None:
            trace["action"] = "abstain"
        if sql_sub_intent == "specific_persons":
            return "Je n'ai trouve aucune video correspondant a cette demande dans la base."
        return (
            "Je n'ai trouve aucun contenu correspondant a cette demande. "
            "Si tu fais reference a une video precise, indique son titre exact ou un mot-cle du titre."
        )

    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer"
        if sql_sub_intent == "specific_persons":
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
                    f"Source {index} :",
                    f"Titre: {source['video_title']}",
                    f"URL: {source['video_url']}",
                    f"Texte: {source['text']}",
                ]
            )
        )

    task_prompt = build_sql_sub_intent_prompt(sql_sub_intent)
    source_marker_instruction = (
        "" if sql_sub_intent == "transcript_verbatim" else SOURCE_MARKER_INSTRUCTION + " "
    )
    system_prompt = render_answer_system_prompt(
        prompt_template,
        route_instructions=(
            task_prompt + "N'invente aucune information absente des resultats."
        ),
        source_marker_instruction=source_marker_instruction,
    )
    input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nSources pour répondre :\n\n" + "\n\n".join(context_blocks)},
        ]
    response = client.responses.create(model=answer_model, input=input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable pour la route sql.")


def generate_final_answer(
    client: LLMClientProtocol | None,
    question: str,
    answer_model: str | None,
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
    trace: dict[str, str] | None = None,
    prompt_template: str | None = None,
) -> str:
    route = retrieval.get("route") or retrieval.get("retrieval_mode")
    source_evaluation = retrieval.get("source_evaluation") or {}
    action_hint = source_evaluation.get("action_hint")
    if route not in {"direct", "memory"} and source_evaluation.get("reason") == "ambiguous_person":
        if trace is not None:
            trace["action"] = "clarify"
        return (
            (retrieval.get("person_resolution") or {}).get("message")
            or "Peux-tu préciser le nom de l'intervenant ?"
        )
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
        return generate_sql_answer(
            client,
            question,
            answer_model,
            retrieval.get("sql_sub_intent"),
            sources,
            trace,
            prompt_template,
        )
    if route == "rag":
        return generate_answer(
            client, question, answer_model, sources, trace, prompt_template
        )
    if route == "sql":
        return generate_sql_answer(
            client,
            question,
            answer_model,
            retrieval.get("sql_sub_intent"),
            sources,
            trace,
            prompt_template,
        )
    if route == "memory":
        return generate_memory_answer(
            client,
            question,
            answer_model,
            retrieval.get("memory_items", []),
            trace,
            prompt_template,
        )
    if route == "multi_source":
        return generate_multi_source_answer(
            client,
            question,
            answer_model,
            route,
            retrieval.get("memory_items", []),
            sources,
            retrieval.get("sql_sub_intent"),
            trace,
            prompt_template,
        )
    return generate_answer(
        client, question, answer_model, sources, trace, prompt_template
    )
