from __future__ import annotations

import json
import re
from typing import Any

from interface.backend.llm_providers import LLMClientProtocol
from interface.backend.schemas import AnswerAction
from interface.backend.utilities import safe_json_loads, serialize_openai_response


FINAL_ANSWER_STYLE = (
    "Réponds directement à la question avec les éléments disponibles. "
    "Formate toujours la réponse en Markdown lisible, avec des paragraphes, listes ou tableaux lorsque cela améliore la clarté. "
    "N'introduis pas ta réponse par une formule comme « d'après les sources » "
    "ou « selon les documents ». "
    "Si ta réponse consiste uniquement à présenter des sources ou des vidéos, "
    "place une courte formule de politesse au début. "
    "Quand action vaut answer, ne termine pas par une phrase indiquant qu'il manque "
    "des informations et ne parle pas de tes limites ni de la recherche effectuée. "
    "Quand action vaut clarify ou abstain, formule uniquement la précision nécessaire "
    "ou l'impossibilité factuelle de répondre avec les éléments fournis."
)


SOURCE_MARKER_INSTRUCTION = (
    "Pour chaque information importante provenant d'un chunk, ajoute son marqueur "
    "[S1], [S2], etc. correspondant au numéro du chunk dans le contexte. "
    "N'utilise que les marqueurs des chunks réellement utilisés."
)


ANSWER_ACTION_INSTRUCTION = (
    "Retourne uniquement un objet JSON valide avec exactement deux clés : answer et action. "
    "action doit valoir exactement answer, clarify ou abstain. Choisis answer uniquement si "
    "le message répond suffisamment à la question à partir des éléments fournis. Choisis "
    "clarify si une ambiguïté empêche de savoir quelle information, personne ou vidéo est "
    "demandée ; answer contient alors une seule question de précision. Choisis abstain si la "
    "demande est claire mais que les éléments fournis ne permettent pas d'y répondre "
    "fidèlement. Le champ answer contient uniquement le message final à afficher."
)


ANSWER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "action": {
            "type": "string",
            "enum": ["answer", "clarify", "abstain"],
        },
    },
    "required": ["answer", "action"],
    "additionalProperties": False,
}


DEFAULT_ANSWER_PROMPT_TEMPLATE = (
    "{route_instructions}\n\n"
    "{source_marker_instruction}\n\n"
    f"{ANSWER_ACTION_INSTRUCTION}\n\n"
    f"{FINAL_ANSWER_STYLE}"
)


def create_answer_response(
    client: LLMClientProtocol,
    answer_model: str,
    input_messages: list[dict[str, str]],
) -> Any:
    return client.responses.create(
        model=answer_model,
        input=input_messages,
        response_schema=ANSWER_RESPONSE_SCHEMA,
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
        # Compatibilité avec les templates créés pendant les deux versions du contrat.
        "{answer_output_instruction}": ANSWER_ACTION_INSTRUCTION.strip(),
        "{answer_action_instruction}": ANSWER_ACTION_INSTRUCTION.strip(),
        "{answer_style}": FINAL_ANSWER_STYLE.strip(),
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    if "action doit valoir exactement answer, clarify ou abstain" not in rendered:
        rendered = f"{rendered}\n\n{ANSWER_ACTION_INSTRUCTION}"
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
    """Extrait la réponse et conserve l'action choisie par le modèle."""
    fallback_action: AnswerAction = "abstain"
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
    if trace is not None:
        trace["action"] = action
    answer = str(payload.get("answer") or "").strip()
    return answer or raw_answer


def select_answer_sources(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Retire les marqueurs et conserve les sources citées par marqueur ou URL."""
    selected_indexes = {
        int(value)
        for value in re.findall(r"\[S(\d+)\]", answer, flags=re.IGNORECASE)
        if 1 <= int(value) <= len(sources)
    }
    selected_indexes.update(
        index
        for index, source in enumerate(sources, start=1)
        if (video_url := str(source.get("video_url") or "").strip())
        and video_url in answer
    )
    cleaned_answer = re.sub(r"\s*\[S\d+\]", "", answer, flags=re.IGNORECASE).strip()
    selected_sources = [
        source for index, source in enumerate(sources, start=1) if index in selected_indexes
    ]
    return cleaned_answer, selected_sources


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
    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer" if sources else "abstain"
        if not sources:
            return "Je n'ai trouve aucun chunk pertinent dans la base pour repondre a cette question."
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
                "content": (
                    f"Question utilisateur: {question}\n\nSources pour répondre :\n\n"
                    + ("\n\n".join(context_blocks) or "Aucune source exploitable.")
                ),
            },
        ]
    response = create_answer_response(client, answer_model, input_messages)
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
    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer" if memory_items else "abstain"
        if not memory_items:
            return "Je n'ai pas trouve d'historique de conversation exploitable pour repondre a cette demande."
        history = "\n".join(f"{item['role']}: {item['text']}" for item in memory_items)
        return history

    history = (
        "\n".join(f"{item['role']}: {item['text']}" for item in memory_items)
        or "Aucun historique exploitable."
    )
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
    response = create_answer_response(client, answer_model, input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable pour la route memory.")


def build_sql_sub_intent_prompt(sql_sub_intent: str | None) -> str:
    if sql_sub_intent == "analytics":
        return (
            "Tu réponds à une demande analytique à partir du résultat SQL fourni. "
            "Respecte exactement l'opération demandée : comptage, agrégation, classement, extremum ou statistiques d'une vidéo. "
            "Présente uniquement les valeurs et entités présentes dans le résultat, sans extrapoler au-delà de son périmètre. "
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
        "Tu réponds directement à la question en t'appuyant sur les sources vidéo structurées fournies. "
        "Utilise les informations pertinentes contenues dans ces sources pour formuler une réponse naturelle et précise. "
        "Ne réduis pas la réponse à une liste de vidéos et ne commence pas automatiquement par présenter les vidéos trouvées. "
        "Mentionne le titre ou le lien d'une vidéo seulement si la question le demande ou si cela aide réellement à comprendre ou vérifier la réponse. "
        "Si la question porte sur une personne, son poste ou sa fonction, utilise uniquement les informations d'intervenant fournies dans les sources. "
        "Si plusieurs sources sont pertinentes, synthétise-les et distingue-les uniquement lorsque c'est nécessaire. "
        "Ne produis une description ou un résumé d'une vidéo que si la question le demande. "
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
    response = create_answer_response(client, answer_model, input_messages)
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
    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "answer" if sources else "abstain"
        if not sources:
            if sql_sub_intent == "specific_persons":
                return "Je n'ai trouve aucune video correspondant a cette demande dans la base."
            return (
                "Je n'ai trouve aucun contenu correspondant a cette demande. "
                "Si tu fais reference a une video precise, indique son titre exact ou un mot-cle du titre."
            )
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
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nSources pour répondre :\n\n"
                    + ("\n\n".join(context_blocks) or "Aucun résultat SQL exploitable.")
                ),
            },
        ]
    response = create_answer_response(client, answer_model, input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable pour la route sql.")


def generate_person_clarification_answer(
    client: LLMClientProtocol | None,
    question: str,
    answer_model: str | None,
    person_resolution: dict[str, Any],
    trace: dict[str, str] | None = None,
    prompt_template: str | None = None,
) -> str:
    """Laisse le modèle de réponse formuler l'action face à une personne ambiguë."""
    fallback = (
        str(person_resolution.get("message") or "").strip()
        or "Peux-tu préciser le nom de l'intervenant ?"
    )
    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "clarify"
        return fallback

    input_messages = [
        {
            "role": "system",
            "content": render_answer_system_prompt(
                prompt_template,
                route_instructions=(
                    "Tu réponds à une demande dont la personne visée n'a pas été résolue "
                    "de façon unique. Utilise les candidats fournis pour décider si une "
                    "question de précision est nécessaire. N'invente aucune identité."
                ),
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nRésolution des personnes:\n"
                + json.dumps(person_resolution, ensure_ascii=False)
            ),
        },
    ]
    response = create_answer_response(client, answer_model, input_messages)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de clarification exploitable.")


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
    person_resolution = retrieval.get("person_resolution") or {}
    if route != "direct" and person_resolution.get("ambiguous"):
        return generate_person_clarification_answer(
            client,
            question,
            answer_model,
            person_resolution,
            trace,
            prompt_template,
        )
    if route == "direct":
        if trace is not None:
            trace["action"] = "answer"
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
