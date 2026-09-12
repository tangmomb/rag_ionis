from __future__ import annotations

import json
import re
from typing import Any, Callable

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
    "Quand action vaut abstain, formule uniquement l'impossibilité factuelle de "
    "répondre avec les éléments fournis."
)


SOURCE_SELECTION_INSTRUCTION = (
    "Renseigne `source_indexes` avec les numéros des sources réellement utilisées "
    "pour construire la réponse, par exemple [1, 3]. Utilise une liste vide si aucune "
    "source n'est utilisée. Utilise une liste vide avec action=abstain. N'ajoute aucun "
    "marqueur [S1] ou citation technique dans `answer`."
)


ANSWER_ACTION_INSTRUCTION = (
    "Choisis l'action answer ou abstain. Choisis answer uniquement si "
    "le message répond suffisamment à la question à partir des éléments fournis. Choisis "
    "abstain si la demande est ambiguë ou que les éléments fournis ne permettent pas "
    "d'y répondre fidèlement. Avec answer, retry_query vaut null. Avec abstain, retry_query contient "
    "une question de recherche courte, autonome et plus précise qui pourrait permettre de "
    "répondre. `retry_query` est du texte naturel, jamais du SQL, une commande ou du code. "
    "Le champ `answer` contient uniquement le message final à afficher et "
    "le champ `source_indexes` contient uniquement les numéros des sources utilisées."
)


ANSWER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "action": {
            "type": "string",
            "enum": ["answer", "abstain"],
        },
        "source_indexes": {
            "type": "array",
            # Mistral's json_schema relay rejects numeric constraints and uniqueItems.
            # Index bounds and deduplication are enforced by
            # parse_answer_output below.
            "items": {"type": "integer"},
        },
        # Keep the nullable form consistent with the planner schema.
        "retry_query": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
    },
    "required": ["answer", "action", "source_indexes", "retry_query"],
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
    stream_callback: Callable[[str], None] | None = None,
) -> Any:
    return client.responses.create(
        model=answer_model,
        input=input_messages,
        response_schema=ANSWER_RESPONSE_SCHEMA,
        stream_callback=stream_callback,
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
    if "Choisis l'action answer ou abstain." not in rendered:
        rendered = f"{rendered}\n\n{ANSWER_ACTION_INSTRUCTION}"
    return re.sub(r"\n{3,}", "\n\n", rendered).strip()


def record_answer_trace(
    trace: dict[str, Any] | None,
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


def parse_answer_output(raw_answer: str, trace: dict[str, Any] | None = None) -> str:
    """Extrait la réponse et conserve l'action choisie par le modèle."""
    fallback_action: AnswerAction = "abstain"
    try:
        payload = safe_json_loads(raw_answer)
    except (TypeError, ValueError, json.JSONDecodeError):
        if trace is not None:
            trace["action"] = fallback_action
            trace["source_indexes"] = []
            trace["retry_query"] = None
        return raw_answer

    if not isinstance(payload, dict):
        if trace is not None:
            trace["action"] = fallback_action
            trace["source_indexes"] = []
            trace["retry_query"] = None
        return raw_answer

    action = payload.get("action")
    if action not in {"answer", "abstain"}:
        action = fallback_action
    if trace is not None:
        raw_source_indexes = payload.get("source_indexes")
        source_indexes = list(
            dict.fromkeys(
                index
                for index in raw_source_indexes
                if isinstance(index, int) and not isinstance(index, bool) and index >= 1
            )
        ) if isinstance(raw_source_indexes, list) else []
        trace["source_indexes"] = source_indexes
        raw_retry_query = payload.get("retry_query")
        retry_query = (
            raw_retry_query.strip()
            if isinstance(raw_retry_query, str)
            else ""
        )
        retry_query = (
            retry_query
            if action == "abstain"
            and retry_query
            and not re.match(
                r"^(?:select|with|insert|update|delete|alter|drop|create|merge)\b",
                retry_query,
                flags=re.IGNORECASE,
            )
            else None
        )
        if action == "abstain" and source_indexes:
            trace["action"] = "answer"
            trace["retry_query"] = None
            trace["action_normalization"] = {
                "reason": "abstain_with_cited_sources",
                "from_action": "abstain",
                "to_action": "answer",
                "source_indexes": source_indexes,
            }
        else:
            trace["action"] = action
            trace["retry_query"] = retry_query
    answer = str(payload.get("answer") or "").strip()
    return answer or raw_answer


def select_answer_sources(
    answer: str,
    sources: list[dict[str, Any]],
    source_indexes: list[int] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Sélectionne les sources JSON, avec lecture des anciens marqueurs en secours."""
    selected_indexes = {
        index for index in (source_indexes or []) if 1 <= index <= len(sources)
    }
    selected_indexes.update({
        int(value)
        for value in re.findall(r"\[S(\d+)\]", answer, flags=re.IGNORECASE)
        if 1 <= int(value) <= len(sources)
    })
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


def format_global_analytics_context(sources: list[dict[str, Any]]) -> str | None:
    """Turn flat SQL ranking rows into one readable prompt section."""
    ranking_sources = [
        (index, source)
        for index, source in enumerate(sources, start=1)
        if isinstance(source.get("global_ranking"), dict)
    ]
    if not ranking_sources:
        return None

    first_source = ranking_sources[0][1]
    first_text = str(first_source.get("text") or "")
    population_match = re.search(r"Population analysée : ([^.]+)\.", first_text)
    first_publication_match = re.search(r"Première publication : ([^.]+)\.", first_text)
    last_publication_match = re.search(r"Dernière publication : ([^.]+)\.", first_text)
    overview = ["## Population analysée"]
    if population_match:
        overview.append(f"- Vidéos : {population_match.group(1)}")
    if first_publication_match:
        overview.append(f"- Première publication : {first_publication_match.group(1)}")
    if last_publication_match:
        overview.append(f"- Dernière publication : {last_publication_match.group(1)}")

    labels = {
        ("views", "top"): "Vidéos les plus vues",
        ("views", "bottom"): "Vidéos les moins vues",
        ("likes", "top"): "Vidéos avec le plus de likes",
        ("likes", "bottom"): "Vidéos avec le moins de likes",
        ("comments", "top"): "Vidéos avec le plus de commentaires",
        ("comments", "bottom"): "Vidéos avec le moins de commentaires",
    }
    metric_labels = {"views": "vues", "likes": "likes", "comments": "commentaires"}
    sections = ["\n".join(overview)]
    for key, heading in labels.items():
        rows = []
        for index, source in ranking_sources:
            ranking = source["global_ranking"]
            if (ranking.get("metric"), ranking.get("direction")) != key:
                continue
            rows.append(
                f"{ranking.get('rank')}. {source.get('video_title') or 'Sans titre'} — "
                f"{ranking.get('value')} {metric_labels[key[0]]} "
                f"(source {index})"
            )
        if rows:
            ranks = [int(source["global_ranking"].get("rank") or 0) for _index, source in ranking_sources if (source["global_ranking"].get("metric"), source["global_ranking"].get("direction")) == key]
            rank_label = f"rang {min(ranks)}" if len(ranks) == 1 else f"rangs {min(ranks)} à {max(ranks)}"
            sections.append("## " + heading + f" ({rank_label})\n" + "\n".join(rows))
    return "\n\n".join(sections)


def format_answer_sources(
    sources: list[dict[str, Any]],
    *,
    sql_sub_intent: str | None = None,
) -> str:
    """Use a compact semantic layout for global analytics, otherwise source cards."""
    global_context = format_global_analytics_context(sources)
    if global_context is not None:
        return "Données analytiques globales de la chaîne en question :\n\n" + global_context
    source_cards = "\n\n".join(
        "\n".join(
            [
                f"Source {index} :",
                f"Titre: {source['video_title']}",
                f"URL: {source['video_url']}",
                f"Texte: {source['text']}",
            ]
        )
        for index, source in enumerate(sources, start=1)
    )
    if not source_cards:
        return ""
    if sql_sub_intent != "analytics":
        return source_cards
    return (
        "Résultats SQL vérifiés : les lignes ci-dessous sont les données retournées "
        "par la base et peuvent suffire à répondre directement à la question.\n\n"
        + source_cards
    )


def generate_answer(
    client: LLMClientProtocol | None,
    question: str,
    answer_model: str | None,
    sources: list[dict[str, Any]],
    trace: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    stream_callback: Callable[[str], None] | None = None,
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
                    source_marker_instruction=SOURCE_SELECTION_INSTRUCTION,
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
    response = create_answer_response(client, answer_model, input_messages, stream_callback)
    record_answer_trace(trace, answer_model, input_messages, response)
    answer = getattr(response, "output_text", "").strip()
    if answer:
        return parse_answer_output(answer, trace)
    raise RuntimeError("Le modele n'a pas renvoye de texte exploitable.")


def build_sql_sub_intent_prompt(sql_sub_intent: str | None) -> str:
    if sql_sub_intent == "analytics":
        return (
            "Tu réponds à une demande analytique à partir du résultat SQL fourni. "
            "Respecte exactement l'opération demandée : comptage, agrégation, classement, extremum ou statistiques d'une vidéo. "
            "Présente uniquement les valeurs et entités présentes dans le résultat, sans extrapoler au-delà de son périmètre. "
            "Indique toujours la date de collecte ou du snapshot associée à chaque statistique citée. "
            "Si le contexte contient des données analytiques globales, il est organisé en population puis en six classements. "
            "Utilise seulement le ou les classements nécessaires à la question ; ne récite pas les autres. "
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


def generate_sql_answer(
    client: LLMClientProtocol | None,
    question: str,
    answer_model: str | None,
    sql_sub_intent: str | None,
    sources: list[dict[str, Any]],
    trace: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    stream_callback: Callable[[str], None] | None = None,
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

    task_prompt = build_sql_sub_intent_prompt(sql_sub_intent)
    source_marker_instruction = SOURCE_SELECTION_INSTRUCTION
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
                    + (
                        format_answer_sources(
                            sources,
                            sql_sub_intent=sql_sub_intent,
                        )
                        or "Aucun résultat SQL exploitable."
                    )
                ),
            },
        ]
    response = create_answer_response(client, answer_model, input_messages, stream_callback)
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
    trace: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> str:
    """Laisse le modèle de réponse formuler l'action face à une personne ambiguë."""
    fallback = (
        str(person_resolution.get("message") or "").strip()
        or "Peux-tu préciser le nom de l'intervenant ?"
    )
    if client is None or not answer_model:
        if trace is not None:
            trace["action"] = "abstain"
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
    response = create_answer_response(client, answer_model, input_messages, stream_callback)
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
    trace: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    judge_feedback: str | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> str:
    generation_question = question
    if judge_feedback:
        generation_question = (
            f"{question}\n\nCorrection interne obligatoire : {judge_feedback.strip()}"
        )
    route = retrieval.get("route") or retrieval.get("retrieval_mode")
    person_resolution = retrieval.get("person_resolution") or {}
    if route != "direct" and person_resolution.get("ambiguous"):
        return generate_person_clarification_answer(
            client,
            generation_question,
            answer_model,
            person_resolution,
            trace,
            prompt_template,
            stream_callback,
        )
    if route == "direct":
        if trace is not None:
            trace["action"] = "answer"
        return retrieval.get("direct_answer") or "Je peux repondre directement a cette demande."
    if route == "sql_search":
        return generate_sql_answer(
            client,
            generation_question,
            answer_model,
            "description" if retrieval.get("description_requested") else retrieval.get("sql_sub_intent"),
            sources,
            trace,
            prompt_template,
            stream_callback,
        )
    if route == "vector_search":
        return generate_answer(
            client, generation_question, answer_model, sources, trace, prompt_template, stream_callback
        )
    return generate_answer(
        client, generation_question, answer_model, sources, trace, prompt_template, stream_callback
    )
