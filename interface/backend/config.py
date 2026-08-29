from __future__ import annotations

from pathlib import Path

from .environment import load_project_env


PROJECT_DIR = Path(__file__).resolve().parents[2]
INTERFACE_DIR = Path(__file__).resolve().parents[1]
load_project_env(PROJECT_DIR)

DEFAULT_RAG_LLM_MODEL = "mistral-medium-latest"
# Configuration des LLM par étape. Change uniquement ces trois constantes pour
# répartir les étapes entre Mistral, OpenAI ou Google.
DEFAULT_REFORMULATION_MODEL = "gpt-5.6-terra"
DEFAULT_PLANNER_MODEL = DEFAULT_RAG_LLM_MODEL
DEFAULT_ANALYTICS_SQL_MODEL = DEFAULT_RAG_LLM_MODEL
DEFAULT_GENERATION_MODEL = DEFAULT_RAG_LLM_MODEL
RAG_LLM_MODEL_CHOICES = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)
DEFAULT_RERANK_MODEL = "cohere-rerank"
DEFAULT_COHERE_RERANK_MODEL = "rerank-v4.0-fast"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 2000
DEFAULT_TOP_K = 40
DEFAULT_FINAL_K = 5
MAX_TOP_K = 50
MAX_FINAL_K = 20
DEFAULT_PREFILTER_LIMIT = 1000
DEFAULT_FUSION_K = 60
DEFAULT_BM25_LIMIT = 40
DEFAULT_VECTOR_LIMIT = 40
DEFAULT_RRF_TOP_N = 30
