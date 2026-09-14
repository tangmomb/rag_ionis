from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline.support.environment import load_project_env

from interface.backend.api import run_rag as rag
from interface.backend.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FINAL_K,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
    DEFAULT_TOP_K,
    RAG_LLM_MODEL_CHOICES,
)
from interface.backend.database import ensure_chat_schema
from interface.backend.generation import DEFAULT_ANSWER_PROMPT_TEMPLATE
from interface.backend.llm_providers import (
    LLM_MAX_RETRIES_ENV,
    LLM_MODEL_CATALOG,
    LLMProviderError,
    LLM_REQUEST_TIMEOUT_ENV,
    OPENAI_SERVICE_TIER_ENV,
    provider_for_model,
    track_llm_costs,
)
from interface.backend.planner import (
    build_planner_prompt,
    build_question_reformulation_prompt,
)
from interface.backend.schemas import RagRequest, RagResponse
from interface.backend.telemetry import configure_telemetry, shutdown_telemetry
from interface.backend.utilities import normalize_model_name


DEFAULT_PHOENIX_BASE_URL = "http://127.0.0.1:6006"
OPENAI_EXPERIMENT_SERVICE_TIERS = ("default", "fast")
OPENAI_SERVICE_TIER_OPTIONS = (
    ("Configuration du projet", None),
    ("Standard", "default"),
    ("Fast", "fast"),
)
OPENAI_SERVICE_TIER_IDS_BY_LABEL = dict(OPENAI_SERVICE_TIER_OPTIONS)
OPENAI_SERVICE_TIER_LABELS_BY_ID = {
    service_tier: label
    for label, service_tier in OPENAI_SERVICE_TIER_OPTIONS
}
LLM_MODEL_OPTIONS = tuple(
    (
        f"{provider_config['label']} - {model_label}",
        model_id,
    )
    for provider_config in LLM_MODEL_CATALOG.values()
    for model_label, model_id in provider_config["models"]
)
LLM_MODEL_IDS_BY_LABEL = dict(LLM_MODEL_OPTIONS)
LLM_MODEL_LABELS_BY_ID = {
    model_id: label for label, model_id in LLM_MODEL_OPTIONS
}
LLM_MODEL_SHORT_NAMES = {
    "gpt-5.6-sol": "sol",
    "gpt-5.6-terra": "terra",
    "gpt-5.6-luna": "luna",
    "mistral-medium-latest": "mistral-medium",
    "mistral-small-latest": "mistral-small",
    "mistral-large-latest": "mistral-large",
    "gemini-3.1-flash-lite": "gemini-3-1-flash-lite",
    "gemini-3.6-flash": "gemini-3-6-flash",
    "gemini-3.5-flash-lite": "gemini-3-5-flash-lite",
}
DEFAULT_REFORMULATION_PROMPT = build_question_reformulation_prompt("", [])[0]
DEFAULT_PLANNER_PROMPT = build_planner_prompt("")[0]


@dataclass(frozen=True)
class RagExperimentSettings:
    question_key: str = "question"
    reformulation_model: str = DEFAULT_REFORMULATION_MODEL
    planner_model: str = DEFAULT_PLANNER_MODEL
    answer_model: str = DEFAULT_GENERATION_MODEL
    reformulation_prompt: str = DEFAULT_REFORMULATION_PROMPT
    planner_prompt: str = DEFAULT_PLANNER_PROMPT
    answer_prompt: str = DEFAULT_ANSWER_PROMPT_TEMPLATE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rerank_model: str | None = None
    use_rerank: bool = True
    top_k: int = DEFAULT_TOP_K
    final_k: int = DEFAULT_FINAL_K
    openai_service_tier: str | None = None
    rerank_delay_seconds: float = 10.0


@dataclass(frozen=True)
class ExperimentSelection:
    dataset: str
    experiment_name: str
    reformulation_model: str = DEFAULT_REFORMULATION_MODEL
    planner_model: str = DEFAULT_PLANNER_MODEL
    answer_model: str = DEFAULT_GENERATION_MODEL
    reformulation_prompt: str = DEFAULT_REFORMULATION_PROMPT
    planner_prompt: str = DEFAULT_PLANNER_PROMPT
    answer_prompt: str = DEFAULT_ANSWER_PROMPT_TEMPLATE
    openai_service_tier: str | None = None


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("la valeur doit etre superieure ou egale a 1")
    return parsed


def non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("la valeur doit etre superieure ou egale a 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("la valeur doit etre positive ou nulle")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute le pipeline RAG IONIS sur toutes les entrees d'un dataset "
            "Phoenix et enregistre les resultats comme une experience."
        )
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Nom ou identifiant du dataset deja present dans Phoenix. "
            "Si omis, une fenetre de selection s'ouvre."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        help="Nom de base de l'experience (defaut: rag-ionis).",
    )
    parser.add_argument(
        "--description",
        default="Evaluation de bout en bout du pipeline RAG IONIS.",
        help="Description affichee dans Phoenix.",
    )
    parser.add_argument(
        "--question-key",
        default="question",
        help="Chemin du champ question dans input, par exemple question ou payload.question.",
    )
    parser.add_argument(
        "--phoenix-base-url",
        default=os.getenv("PHOENIX_BASE_URL", DEFAULT_PHOENIX_BASE_URL),
        help="URL HTTP de Phoenix.",
    )
    parser.add_argument(
        "--reformulation-model",
        default=DEFAULT_REFORMULATION_MODEL,
        help="Modele de reformulation OpenAI, Mistral ou Google.",
    )
    parser.add_argument(
        "--planner-model",
        default=DEFAULT_PLANNER_MODEL,
        help="Modele du planner OpenAI, Mistral ou Google.",
    )
    parser.add_argument(
        "--answer-model",
        default=DEFAULT_GENERATION_MODEL,
        help="Modele de reponse OpenAI, Mistral ou Google.",
    )
    parser.add_argument(
        "--openai-service-tier",
        choices=OPENAI_EXPERIMENT_SERVICE_TIERS,
        default=(
            os.getenv(OPENAI_SERVICE_TIER_ENV, "").strip().lower() or None
        ),
        help=(
            "Tier des appels OpenAI : default ou fast. "
            "Si omis, conserve la configuration du projet OpenAI."
        ),
    )
    parser.add_argument(
        "--reformulation-prompt",
        default=DEFAULT_REFORMULATION_PROMPT,
        help="Prompt systeme de reformulation.",
    )
    parser.add_argument(
        "--planner-prompt",
        default=DEFAULT_PLANNER_PROMPT,
        help="Prompt systeme du planner.",
    )
    parser.add_argument(
        "--answer-prompt",
        default=DEFAULT_ANSWER_PROMPT_TEMPLATE,
        help="Template du prompt systeme de reponse.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model")
    parser.add_argument(
        "--no-rerank",
        dest="use_rerank",
        action="store_false",
        default=True,
        help="Desactive le reranking Cohere.",
    )
    parser.add_argument("--top-k", type=positive_integer, default=DEFAULT_TOP_K)
    parser.add_argument("--final-k", type=positive_integer, default=DEFAULT_FINAL_K)
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const=1,
        type=positive_integer,
        metavar="N",
        help=(
            "Teste N exemples sans enregistrer d'experience. "
            "Sans valeur, teste un exemple."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=positive_integer,
        default=300,
        help="Timeout en secondes pour chaque exemple (defaut: 300).",
    )
    parser.add_argument(
        "--retries",
        type=non_negative_integer,
        default=0,
        help="Nombre de nouvelles tentatives en cas d'echec (defaut: 0).",
    )
    parser.add_argument(
        "--llm-timeout",
        type=positive_integer,
        default=60,
        help="Timeout de chaque appel LLM en secondes (defaut: 60).",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=non_negative_integer,
        default=0,
        help="Reprises internes de chaque client LLM (defaut: 0).",
    )
    parser.add_argument(
        "--rerank-delay-seconds",
        type=non_negative_float,
        default=10.0,
        help=(
            "Delai ajoute apres chaque cas utilisant le rerank Cohere, "
            "pour respecter une limite de debit (defaut: 10)."
        ),
    )
    return parser


def value_at_path(data: Mapping[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            available = ", ".join(sorted(str(key) for key in data)) or "(aucun)"
            raise ValueError(
                f"Champ question introuvable: {path!r}. "
                f"Champs input disponibles: {available}."
            )
        value = value[part]
    return value


def build_rag_request(
    example_input: Mapping[str, Any],
    settings: RagExperimentSettings,
) -> RagRequest:
    question = str(value_at_path(example_input, settings.question_key)).strip()
    if not question:
        raise ValueError(
            f"Le champ {settings.question_key!r} contient une question vide."
        )
    return RagRequest(
        question=question,
        reformulationModel=settings.reformulation_model,
        plannerModel=settings.planner_model,
        answerModel=settings.answer_model,
        reformulationPrompt=settings.reformulation_prompt,
        plannerPrompt=settings.planner_prompt,
        answerPrompt=settings.answer_prompt,
        embeddingModel=settings.embedding_model,
        rerankModel=settings.rerank_model,
        useSql=True,
        useRerank=settings.use_rerank,
        topK=settings.top_k,
        finalK=settings.final_k,
    )


def compact_experiment_output(
    response: RagResponse,
    llm_cost_usd: float | None = None,
    priced_llm_calls: int = 0,
) -> dict[str, Any]:
    retrieval = response.retrieval
    execution_plan = retrieval.get("execution_plan") or {}
    telemetry = retrieval.get("telemetry") or {}
    retrieved_sources = retrieval.get("retrieved_sources")
    if not isinstance(retrieved_sources, list):
        retrieved_sources = response.sources
    return {
        "answer": response.answer,
        "action": response.action,
        "sources": [
            {
                "chunk_id": source.chunk_id,
                "video_title": source.video_title,
                "video_url": source.video_url,
                "cohere_relevance_score": source.cohere_relevance_score,
                "rrf_score": source.rrf_score,
            }
            for source in response.sources
        ],
        "retrieved_sources": [
            compact_retrieved_source(source) for source in retrieved_sources
        ],
        "trace_id": telemetry.get("trace_id"),
        "conversation_id": response.conversation_id,
        "message_id": response.message_id,
        "estimated_llm_cost_usd": llm_cost_usd,
        "priced_llm_calls": priced_llm_calls,
        "diagnostics": {
            "reformulation_model": retrieval.get("reformulation_model"),
            "reformulation_provider": model_provider_name(
                retrieval.get("reformulation_model")
            ),
            "planner_model": retrieval.get("planner_model"),
            "planner_provider": model_provider_name(
                retrieval.get("planner_model")
            ),
            "answer_model": retrieval.get("answer_model"),
            "answer_provider": model_provider_name(
                retrieval.get("answer_model")
            ),
            "route": execution_plan.get("route"),
            "sql_sub_intent": execution_plan.get("sql_sub_intent"),
            "retrieval_mode": retrieval.get("retrieval_mode"),
            "used_rerank": retrieval.get("used_rerank"),
            "source_evaluation": retrieval.get("source_evaluation"),
        },
    }


def source_value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def youtube_video_id_from_url(value: Any) -> str | None:
    url = str(value or "").strip()
    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0] or None
    if host in {"youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [""])[0].strip()
        return video_id or None
    return None


def compact_retrieved_source(source: Any) -> dict[str, Any]:
    video_url = source_value(source, "video_url")
    return {
        "chunk_id": source_value(source, "chunk_id"),
        "youtube_video_id": youtube_video_id_from_url(video_url),
    }


def build_rag_task(settings: RagExperimentSettings):
    def rag_ionis_task(input: Mapping[str, Any]) -> dict[str, Any]:
        request = build_rag_request(input, settings)
        with track_llm_costs() as costs:
            response = rag(request)
        if (
            settings.rerank_delay_seconds
            and response.retrieval.get("used_rerank")
        ):
            time.sleep(settings.rerank_delay_seconds)
        return compact_experiment_output(
            response,
            llm_cost_usd=costs.total_usd if costs.priced_calls else None,
            priced_llm_calls=costs.priced_calls,
        )

    return rag_ionis_task


def response_nonempty(output: Mapping[str, Any] | None) -> bool:
    output = output or {}
    return bool(str(output.get("answer") or "").strip())


def estimated_llm_cost_usd(
    output: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    value = (output or {}).get("estimated_llm_cost_usd")
    if not isinstance(value, (int, float)):
        return None, "unlabeled", "Aucun tarif configure pour les appels LLM."
    return float(value), "estimated", "Estimation USD des appels LLM du cas."


def answer_action(output: Mapping[str, Any] | None) -> dict[str, str]:
    output = output or {}
    action = str(output.get("action") or "unknown")
    return {"label": action}


def answer_action_match(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    output = output or {}
    expected_action = str((expected or {}).get("action") or "").strip()
    actual_action = str(output.get("action") or "unknown")
    if not expected_action:
        return None, "unlabeled", "Le dataset ne fournit pas expected.action."
    matches = actual_action == expected_action
    return (
        float(matches),
        "match" if matches else "mismatch",
        f"attendu={expected_action}; obtenu={actual_action}",
    )


def route_match(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    output = output or {}
    diagnostics = output.get("diagnostics") or {}
    actual_route = (
        str(diagnostics.get("route") or "unknown")
        if isinstance(diagnostics, Mapping)
        else "unknown"
    )
    expected_route = str(
        (expected or {}).get("expected_execution_route") or ""
    ).strip()
    if not expected_route:
        return None, "unlabeled", "Le dataset ne fournit pas expected_execution_route."
    matches = actual_route == expected_route
    return (
        float(matches),
        "match" if matches else "mismatch",
        f"attendu={expected_route}; obtenu={actual_route}",
    )


def expected_identifier_set(
    expected: Mapping[str, Any] | None,
    key: str,
) -> set[str]:
    value = (expected or {}).get(key, [])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def retrieved_identifier_set(
    output: Mapping[str, Any] | None,
    key: str,
) -> set[str]:
    output = output or {}
    sources = output.get("retrieved_sources") or output.get("sources") or []
    if not isinstance(sources, list):
        return set()
    return {
        str(source.get(key)).strip()
        for source in sources
        if isinstance(source, Mapping) and source.get(key) is not None
        and str(source.get(key)).strip()
    }


def retrieval_recall(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    *,
    expected_key: str,
    actual_key: str,
) -> tuple[float | None, str, str]:
    expected_ids = expected_identifier_set(expected, expected_key)
    if not expected_ids:
        return None, "unlabeled", f"Le dataset ne fournit pas {expected_key}."
    actual_ids = retrieved_identifier_set(output, actual_key)
    matched_ids = expected_ids & actual_ids
    score = len(matched_ids) / len(expected_ids)
    return (
        score,
        "measured",
        f"retrouves={len(matched_ids)}/{len(expected_ids)}; retournes={len(actual_ids)}",
    )


def retrieval_precision(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    *,
    expected_key: str,
    actual_key: str,
) -> tuple[float | None, str, str]:
    expected_ids = expected_identifier_set(expected, expected_key)
    if not expected_ids:
        return None, "unlabeled", f"Le dataset ne fournit pas {expected_key}."
    actual_ids = retrieved_identifier_set(output, actual_key)
    if not actual_ids:
        return 0.0, "measured", "Aucun resultat retourne."
    matched_ids = expected_ids & actual_ids
    score = len(matched_ids) / len(actual_ids)
    return (
        score,
        "measured",
        f"pertinents={len(matched_ids)}/{len(actual_ids)}; attendus={len(expected_ids)}",
    )


def youtube_video_recall(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    return retrieval_recall(
        output,
        expected,
        expected_key="youtube_video_ids",
        actual_key="youtube_video_id",
    )


def youtube_video_precision(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    return retrieval_precision(
        output,
        expected,
        expected_key="youtube_video_ids",
        actual_key="youtube_video_id",
    )


def route_scoped_video_metric(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    *,
    route: str,
    metric: Callable[..., tuple[float | None, str, str]],
) -> tuple[float | None, str, str]:
    expected_route = str(
        (expected or {}).get("expected_execution_route") or ""
    ).strip()
    if expected_route != route:
        return None, "unlabeled", f"Metrique reservee aux cas {route}."
    return metric(output, expected)


def sql_youtube_video_recall(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    return route_scoped_video_metric(
        output, expected, route="sql_search", metric=youtube_video_recall
    )


def sql_youtube_video_precision(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    return route_scoped_video_metric(
        output, expected, route="sql_search", metric=youtube_video_precision
    )


def vector_youtube_video_recall(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    return route_scoped_video_metric(
        output, expected, route="vector_search", metric=youtube_video_recall
    )


def vector_youtube_video_precision(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    return route_scoped_video_metric(
        output, expected, route="vector_search", metric=youtube_video_precision
    )


def chunk_recall(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    return retrieval_recall(
        output,
        expected,
        expected_key="relevant_chunk_ids",
        actual_key="chunk_id",
    )


def chunk_precision(
    output: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    return retrieval_precision(
        output,
        expected,
        expected_key="relevant_chunk_ids",
        actual_key="chunk_id",
    )


def default_experiment_name() -> str:
    return "rag-ionis"


def model_option_label(value: str, default: str) -> str:
    model_id = normalize_model_name(value, default)
    return LLM_MODEL_LABELS_BY_ID.get(
        model_id,
        LLM_MODEL_LABELS_BY_ID[default],
    )


def openai_service_tier_option_label(value: str | None) -> str:
    return OPENAI_SERVICE_TIER_LABELS_BY_ID.get(
        value,
        OPENAI_SERVICE_TIER_LABELS_BY_ID[None],
    )


def model_short_name(model_id: str) -> str:
    known_name = LLM_MODEL_SHORT_NAMES.get(model_id)
    if known_name:
        return known_name
    normalized = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return normalized or "modele"


def model_provider_name(model_id: Any) -> str | None:
    if not model_id:
        return None
    try:
        return provider_for_model(str(model_id))
    except LLMProviderError:
        return "unknown"


def experiment_name_with_models(
    base_name: str,
    settings: RagExperimentSettings,
) -> str:
    model_suffix = "_".join(
        [
            model_short_name(settings.reformulation_model),
            model_short_name(settings.planner_model),
            model_short_name(settings.answer_model),
        ]
    )
    cleaned_base_name = base_name.strip() or default_experiment_name()
    suffix = f"_{model_suffix}"
    if cleaned_base_name.casefold().endswith(suffix.casefold()):
        return cleaned_base_name
    return f"{cleaned_base_name}{suffix}"


def list_dataset_names(phoenix_base_url: str, timeout: int) -> list[str]:
    from phoenix.client import Client

    client = Client(
        base_url=phoenix_base_url.rstrip("/"),
        api_key=os.getenv("PHOENIX_API_KEY"),
    )
    datasets = client.datasets.list(timeout=timeout)
    return sorted(
        {
            str(dataset["name"]).strip()
            for dataset in datasets
            if str(dataset.get("name") or "").strip()
        },
        key=str.casefold,
    )


def prompt_experiment_selection(
    phoenix_base_url: str,
    timeout: int,
    initial_experiment_name: str | None = None,
    initial_reformulation_model: str = DEFAULT_REFORMULATION_MODEL,
    initial_planner_model: str = DEFAULT_PLANNER_MODEL,
    initial_answer_model: str = DEFAULT_GENERATION_MODEL,
    initial_reformulation_prompt: str = DEFAULT_REFORMULATION_PROMPT,
    initial_planner_prompt: str = DEFAULT_PLANNER_PROMPT,
    initial_answer_prompt: str = DEFAULT_ANSWER_PROMPT_TEMPLATE,
    initial_openai_service_tier: str | None = None,
) -> ExperimentSelection | None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter est indisponible. Utilise --dataset et --experiment-name."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.title("Nouvelle experience Phoenix")
    root.resizable(True, True)

    try:
        dataset_names = list_dataset_names(phoenix_base_url, timeout)
    except Exception as exc:
        messagebox.showerror(
            "Connexion Phoenix impossible",
            f"Impossible de recuperer les datasets depuis :\n"
            f"{phoenix_base_url}\n\n{exc}",
            parent=root,
        )
        root.destroy()
        return None

    if not dataset_names:
        messagebox.showwarning(
            "Aucun dataset",
            "Aucun dataset n'est disponible dans Phoenix.",
            parent=root,
        )
        root.destroy()
        return None

    dataset_value = tk.StringVar(value=dataset_names[0])
    experiment_value = tk.StringVar(
        value=initial_experiment_name or default_experiment_name()
    )
    reformulation_model_value = tk.StringVar(
        value=model_option_label(
            initial_reformulation_model,
            DEFAULT_REFORMULATION_MODEL,
        )
    )
    planner_model_value = tk.StringVar(
        value=model_option_label(initial_planner_model, DEFAULT_PLANNER_MODEL)
    )
    answer_model_value = tk.StringVar(
        value=model_option_label(initial_answer_model, DEFAULT_GENERATION_MODEL)
    )
    openai_service_tier_value = tk.StringVar(
        value=openai_service_tier_option_label(initial_openai_service_tier)
    )
    selection: ExperimentSelection | None = None

    frame = ttk.Frame(root, padding=18)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=0)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ttk.Label(frame, text="Dataset Phoenix").grid(
        row=0,
        column=0,
        sticky="w",
        pady=(0, 5),
    )
    dataset_box = ttk.Combobox(
        frame,
        textvariable=dataset_value,
        values=dataset_names,
        state="readonly",
        width=48,
    )
    dataset_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))

    ttk.Label(frame, text="Nom de l'experience").grid(
        row=2,
        column=0,
        sticky="w",
        pady=(0, 5),
    )
    experiment_entry = ttk.Entry(
        frame,
        textvariable=experiment_value,
        width=51,
    )
    experiment_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 18))

    ttk.Separator(frame, orient="horizontal").grid(
        row=4,
        column=0,
        columnspan=2,
        sticky="ew",
        pady=(0, 14),
    )

    model_labels = [label for label, _model_id in LLM_MODEL_OPTIONS]

    ttk.Label(frame, text="LLM de reformulation").grid(
        row=5,
        column=0,
        sticky="w",
        pady=(0, 5),
    )
    ttk.Combobox(
        frame,
        textvariable=reformulation_model_value,
        values=model_labels,
        state="readonly",
        width=48,
    ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 12))

    ttk.Label(frame, text="LLM du planner").grid(
        row=7,
        column=0,
        sticky="w",
        pady=(0, 5),
    )
    ttk.Combobox(
        frame,
        textvariable=planner_model_value,
        values=model_labels,
        state="readonly",
        width=48,
    ).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 12))

    ttk.Label(frame, text="LLM de reponse").grid(
        row=9,
        column=0,
        sticky="w",
        pady=(0, 5),
    )
    ttk.Combobox(
        frame,
        textvariable=answer_model_value,
        values=model_labels,
        state="readonly",
        width=48,
    ).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(0, 18))

    ttk.Label(frame, text="Tier de service OpenAI").grid(
        row=11,
        column=0,
        sticky="w",
        pady=(0, 5),
    )
    ttk.Combobox(
        frame,
        textvariable=openai_service_tier_value,
        values=[label for label, _tier in OPENAI_SERVICE_TIER_OPTIONS],
        state="readonly",
        width=48,
    ).grid(row=12, column=0, columnspan=2, sticky="ew", pady=(0, 18))

    ttk.Label(frame, text="Prompts systeme (editables)").grid(
        row=13,
        column=0,
        columnspan=2,
        sticky="w",
        pady=(0, 5),
    )
    prompt_notebook = ttk.Notebook(frame)
    prompt_notebook.grid(
        row=14,
        column=0,
        columnspan=2,
        sticky="nsew",
        pady=(0, 18),
    )
    frame.rowconfigure(14, weight=1)

    def add_prompt_tab(title: str, initial_value: str):
        tab = ttk.Frame(prompt_notebook, padding=6)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        text = tk.Text(
            tab,
            wrap="word",
            undo=True,
            width=78,
            height=10,
        )
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.insert("1.0", initial_value)
        prompt_notebook.add(tab, text=title)
        return text

    reformulation_prompt_text = add_prompt_tab(
        "Reformulation",
        initial_reformulation_prompt,
    )
    planner_prompt_text = add_prompt_tab("Planner", initial_planner_prompt)
    answer_prompt_text = add_prompt_tab("Reponse", initial_answer_prompt)

    def validate_and_close() -> None:
        nonlocal selection
        dataset = dataset_value.get().strip()
        experiment_name = experiment_value.get().strip()
        if not dataset:
            messagebox.showwarning(
                "Dataset requis",
                "Selectionne un dataset.",
                parent=root,
            )
            return
        if not experiment_name:
            messagebox.showwarning(
                "Nom requis",
                "Saisis un nom pour l'experience.",
                parent=root,
            )
            experiment_entry.focus_set()
            return
        prompt_values = {
            "reformulation": reformulation_prompt_text.get("1.0", "end-1c").strip(),
            "planner": planner_prompt_text.get("1.0", "end-1c").strip(),
            "reponse": answer_prompt_text.get("1.0", "end-1c").strip(),
        }
        empty_prompt = next(
            (name for name, value in prompt_values.items() if not value),
            None,
        )
        if empty_prompt:
            messagebox.showwarning(
                "Prompt requis",
                f"Le prompt {empty_prompt} ne peut pas etre vide.",
                parent=root,
            )
            return
        selection = ExperimentSelection(
            dataset=dataset,
            experiment_name=experiment_name,
            reformulation_model=LLM_MODEL_IDS_BY_LABEL[
                reformulation_model_value.get()
            ],
            planner_model=LLM_MODEL_IDS_BY_LABEL[planner_model_value.get()],
            answer_model=LLM_MODEL_IDS_BY_LABEL[answer_model_value.get()],
            reformulation_prompt=prompt_values["reformulation"],
            planner_prompt=prompt_values["planner"],
            answer_prompt=prompt_values["reponse"],
            openai_service_tier=OPENAI_SERVICE_TIER_IDS_BY_LABEL[
                openai_service_tier_value.get()
            ],
        )
        root.destroy()

    def cancel() -> None:
        root.destroy()

    ttk.Button(frame, text="Annuler", command=cancel).grid(
        row=15,
        column=0,
        sticky="e",
        padx=(0, 6),
    )
    ttk.Button(
        frame,
        text="Lancer l'experience",
        command=validate_and_close,
    ).grid(row=15, column=1, sticky="e")

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.bind("<Return>", lambda _event: validate_and_close())
    root.bind("<Escape>", lambda _event: cancel())
    root.deiconify()
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f"+{x}+{y}")
    dataset_box.focus_set()
    root.mainloop()
    return selection


def run(args: argparse.Namespace) -> dict[str, Any]:
    from phoenix.client import Client

    phoenix_base_url = args.phoenix_base_url.rstrip("/")
    os.environ["PHOENIX_BASE_URL"] = phoenix_base_url
    os.environ.setdefault(
        "PHOENIX_COLLECTOR_ENDPOINT",
        f"{phoenix_base_url}/v1/traces",
    )

    client = Client(
        base_url=phoenix_base_url,
        api_key=os.getenv("PHOENIX_API_KEY"),
    )
    dataset = client.datasets.get_dataset(
        dataset=args.dataset,
        timeout=args.timeout,
    )
    settings = RagExperimentSettings(
        question_key=args.question_key,
        reformulation_model=normalize_model_name(
            args.reformulation_model,
            DEFAULT_REFORMULATION_MODEL,
        ),
        planner_model=normalize_model_name(
            args.planner_model,
            DEFAULT_PLANNER_MODEL,
        ),
        answer_model=normalize_model_name(
            args.answer_model,
            DEFAULT_GENERATION_MODEL,
        ),
        reformulation_prompt=args.reformulation_prompt,
        planner_prompt=args.planner_prompt,
        answer_prompt=args.answer_prompt,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
        use_rerank=args.use_rerank,
        top_k=args.top_k,
        final_k=args.final_k,
        openai_service_tier=args.openai_service_tier,
        rerank_delay_seconds=args.rerank_delay_seconds,
    )

    ensure_chat_schema()
    configure_telemetry()
    experiment_name = experiment_name_with_models(
        args.experiment_name or default_experiment_name(),
        settings,
    )
    previous_openai_service_tier = os.environ.get(OPENAI_SERVICE_TIER_ENV)
    previous_llm_timeout = os.environ.get(LLM_REQUEST_TIMEOUT_ENV)
    previous_llm_max_retries = os.environ.get(LLM_MAX_RETRIES_ENV)
    os.environ[LLM_REQUEST_TIMEOUT_ENV] = str(args.llm_timeout)
    os.environ[LLM_MAX_RETRIES_ENV] = str(args.llm_max_retries)
    if settings.openai_service_tier is not None:
        os.environ[OPENAI_SERVICE_TIER_ENV] = settings.openai_service_tier
    try:
        evaluators = {
            "response_nonempty": response_nonempty,
            "estimated_llm_cost_usd": estimated_llm_cost_usd,
            "answer_action": answer_action,
            "answer_action_match": answer_action_match,
            "route_match": route_match,
            "youtube_video_recall": youtube_video_recall,
            "youtube_video_precision": youtube_video_precision,
            "sql_youtube_video_recall": sql_youtube_video_recall,
            "sql_youtube_video_precision": sql_youtube_video_precision,
            "vector_youtube_video_recall": vector_youtube_video_recall,
            "vector_youtube_video_precision": vector_youtube_video_precision,
            "chunk_recall": chunk_recall,
            "chunk_precision": chunk_precision,
        }
        experiment = client.experiments.run_experiment(
            dataset=dataset,
            task=build_rag_task(settings),
            evaluators=evaluators,
            experiment_name=experiment_name,
            experiment_description=args.description,
            experiment_metadata={
                "application": "rag-ionis",
                "reformulation_model": settings.reformulation_model,
                "reformulation_provider": model_provider_name(
                    settings.reformulation_model
                ),
                "planner_model": settings.planner_model,
                "planner_provider": model_provider_name(
                    settings.planner_model
                ),
                "answer_model": settings.answer_model,
                "answer_provider": model_provider_name(
                    settings.answer_model
                ),
                "prompts": {
                    "reformulation": settings.reformulation_prompt,
                    "planner": settings.planner_prompt,
                    "answer": settings.answer_prompt,
                },
                "embedding_model": settings.embedding_model,
                "rerank_model": settings.rerank_model,
                "use_rerank": settings.use_rerank,
                "top_k": settings.top_k,
                "final_k": settings.final_k,
                "openai_service_tier": settings.openai_service_tier,
                "llm_timeout_seconds": args.llm_timeout,
                "llm_max_retries": args.llm_max_retries,
                "rerank_delay_seconds": settings.rerank_delay_seconds,
            },
            dry_run=args.dry_run or False,
            timeout=args.timeout,
            retries=args.retries,
        )
    finally:
        if previous_openai_service_tier is None:
            os.environ.pop(OPENAI_SERVICE_TIER_ENV, None)
        else:
            os.environ[OPENAI_SERVICE_TIER_ENV] = (
                previous_openai_service_tier
            )
        if previous_llm_timeout is None:
            os.environ.pop(LLM_REQUEST_TIMEOUT_ENV, None)
        else:
            os.environ[LLM_REQUEST_TIMEOUT_ENV] = previous_llm_timeout
        if previous_llm_max_retries is None:
            os.environ.pop(LLM_MAX_RETRIES_ENV, None)
        else:
            os.environ[LLM_MAX_RETRIES_ENV] = previous_llm_max_retries
        shutdown_telemetry()

    result = dict(experiment)
    if not args.dry_run:
        result["url"] = client.experiments.get_experiment_url(
            dataset_id=experiment["dataset_id"],
            experiment_id=experiment["experiment_id"],
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(errors="replace")
    load_project_env(PROJECT_DIR, override=True)
    args = build_parser().parse_args(argv)
    if not args.dataset:
        selection = prompt_experiment_selection(
            args.phoenix_base_url,
            args.timeout,
            args.experiment_name,
            args.reformulation_model,
            args.planner_model,
            args.answer_model,
            args.reformulation_prompt,
            args.planner_prompt,
            args.answer_prompt,
            args.openai_service_tier,
        )
        if selection is None:
            print("Lancement annule.")
            return 0
        args.dataset = selection.dataset
        args.experiment_name = selection.experiment_name
        args.reformulation_model = selection.reformulation_model
        args.planner_model = selection.planner_model
        args.answer_model = selection.answer_model
        args.reformulation_prompt = selection.reformulation_prompt
        args.planner_prompt = selection.planner_prompt
        args.answer_prompt = selection.answer_prompt
        args.openai_service_tier = selection.openai_service_tier
    result = run(args)
    if args.dry_run:
        print(f"Dry run termine sur {args.dry_run} exemple(s). Rien n'a ete enregistre.")
    else:
        print(f"Experience terminee: {result['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
