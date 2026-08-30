from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
    LLM_MODEL_CATALOG,
    LLMProviderError,
    OPENAI_SERVICE_TIER_ENV,
    provider_for_model,
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
    RAG_LLM_MODEL_CHOICES[0]: "sol",
    RAG_LLM_MODEL_CHOICES[1]: "terra",
    RAG_LLM_MODEL_CHOICES[2]: "luna",
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
    shadow_evaluation: bool = False


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
    parser.add_argument(
        "--shadow-evaluation",
        action="store_true",
        help=(
            "Active l'evaluateur shadow et publie ses diagnostics comme metriques "
            "de calibration Phoenix, sans corriger les reponses."
        ),
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
    shadow_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval = response.retrieval
    execution_plan = retrieval.get("execution_plan") or {}
    telemetry = retrieval.get("telemetry") or {}
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
        "trace_id": telemetry.get("trace_id"),
        "conversation_id": response.conversation_id,
        "message_id": response.message_id,
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
            "shadow_evaluation": dict(shadow_evaluation or {}),
        },
    }


def build_rag_task(settings: RagExperimentSettings):
    def rag_ionis_task(input: Mapping[str, Any]) -> dict[str, Any]:
        request = build_rag_request(input, settings)
        shadow_diagnostic: dict[str, Any] = {}
        response = rag(
            request,
            shadow_evaluation_enabled_override=settings.shadow_evaluation,
            shadow_evaluation_sink=shadow_diagnostic,
        )
        return compact_experiment_output(response, shadow_diagnostic)

    return rag_ionis_task


def response_nonempty(output: Mapping[str, Any]) -> bool:
    return bool(str(output.get("answer") or "").strip())


def answer_action(output: Mapping[str, Any]) -> dict[str, str]:
    action = str(output.get("action") or "unknown")
    return {"label": action}


def _shadow_diagnostic(output: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = output.get("diagnostics") or {}
    if not isinstance(diagnostics, Mapping):
        return {}
    shadow = diagnostics.get("shadow_evaluation") or {}
    return shadow if isinstance(shadow, Mapping) else {}


def shadow_status(output: Mapping[str, Any]) -> dict[str, str]:
    diagnostic = _shadow_diagnostic(output)
    return {
        "label": str(diagnostic.get("status") or "not_run"),
        "explanation": str(diagnostic.get("reason") or "Diagnostic indisponible."),
    }


def shadow_grounded(output: Mapping[str, Any]) -> tuple[float | None, str, str]:
    diagnostic = _shadow_diagnostic(output)
    grounded = diagnostic.get("answer_grounded")
    if not isinstance(grounded, bool):
        return None, "not_run", "Diagnostic answer_grounded indisponible."
    return (
        float(grounded),
        "grounded" if grounded else "not_grounded",
        str(diagnostic.get("reason") or ""),
    )


def shadow_retrieval_quality(
    output: Mapping[str, Any],
) -> tuple[float | None, str, str]:
    diagnostic = _shadow_diagnostic(output)
    quality = diagnostic.get("retrieval_quality")
    if not isinstance(quality, (int, float)) or isinstance(quality, bool):
        return None, "not_run", "Diagnostic retrieval_quality indisponible."
    score = float(quality)
    return score, "measured", str(diagnostic.get("reason") or "")


def shadow_status_match(
    output: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> tuple[float | None, str, str]:
    expected_status = str((expected or {}).get("shadow_status") or "").strip()
    actual_status = str(_shadow_diagnostic(output).get("status") or "not_run")
    if not expected_status:
        return None, "unlabeled", "Le dataset ne fournit pas expected.shadow_status."
    matches = actual_status == expected_status
    return (
        float(matches),
        "match" if matches else "mismatch",
        f"attendu={expected_status}; obtenu={actual_status}",
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
        shadow_evaluation=args.shadow_evaluation,
    )

    ensure_chat_schema()
    configure_telemetry()
    experiment_name = experiment_name_with_models(
        args.experiment_name or default_experiment_name(),
        settings,
    )
    previous_openai_service_tier = os.environ.get(OPENAI_SERVICE_TIER_ENV)
    if settings.openai_service_tier is not None:
        os.environ[OPENAI_SERVICE_TIER_ENV] = settings.openai_service_tier
    try:
        evaluators = {
            "response_nonempty": response_nonempty,
            "answer_action": answer_action,
        }
        if settings.shadow_evaluation:
            evaluators.update(
                {
                    "shadow_status": shadow_status,
                    "shadow_grounded": shadow_grounded,
                    "shadow_retrieval_quality": shadow_retrieval_quality,
                    "shadow_status_match": shadow_status_match,
                }
            )
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
                "shadow_evaluation": settings.shadow_evaluation,
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
