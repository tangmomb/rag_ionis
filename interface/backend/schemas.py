from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from interface.backend.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FINAL_K,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REFORMULATION_MODEL,
    DEFAULT_TOP_K,
    MAX_FINAL_K,
    MAX_TOP_K,
)


PlannerRoute = Literal["direct", "search"]
ExecutionRoute = Literal[
    "person_clarification",
    "direct",
    "sql_search",
    "vector_search",
]
AnswerAction = Literal["answer", "abstain"]


class RagRequest(BaseModel):
    question: str = Field(min_length=1)
    conversationId: int | None = None
    apiUrl: str | None = None
    reformulationModel: str = DEFAULT_REFORMULATION_MODEL
    plannerModel: str = DEFAULT_PLANNER_MODEL
    answerModel: str = DEFAULT_GENERATION_MODEL
    reformulationPrompt: str | None = None
    plannerPrompt: str | None = None
    answerPrompt: str | None = None
    embeddingModel: str = DEFAULT_EMBEDDING_MODEL
    rerankModel: str | None = None
    useSql: bool = True
    useRerank: bool = True
    topK: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    finalK: int = Field(default=DEFAULT_FINAL_K, ge=1, le=MAX_FINAL_K)

    @model_validator(mode="after")
    def _force_mistral_medium_for_rag_inference(self) -> "RagRequest":
        """Keep every RAG inference stage on the configured Mistral model."""
        self.reformulationModel = DEFAULT_REFORMULATION_MODEL
        self.plannerModel = DEFAULT_PLANNER_MODEL
        self.answerModel = DEFAULT_GENERATION_MODEL
        return self


class PlannerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    _output_rejection_reason: str | None = PrivateAttr(default=None)

    route: PlannerRoute = "search"
    analytics_requested: bool = False
    query_text: str
    query_text_bm25: str | None = None
    title_hints: list[str] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None
    description_requested: bool = False
    transcription_requested: bool = False

    @property
    def title_hint(self) -> str | None:
        return self.title_hints[0] if self.title_hints else None

    @property
    def output_rejection_reason(self) -> str | None:
        """Reason the LLM planner output was discarded in favour of a safe plan."""
        return self._output_rejection_reason


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # This is the final, deterministic graph branch.  It deliberately differs
    # from PlannerPlan.route, which is only the LLM's coarse direct/search intent.
    route: ExecutionRoute = "vector_search"
    analytics_requested: bool = False
    raw_question: str
    query_text: str
    query_text_bm25: str
    title_hints: list[str] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None
    description_requested: bool = False
    transcription_requested: bool = False
    top_k: int | None = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    final_k: int | None = Field(default=DEFAULT_FINAL_K, ge=1, le=MAX_FINAL_K)

    @property
    def title_hint(self) -> str | None:
        return self.title_hints[0] if self.title_hints else None


class ChunkSource(BaseModel):
    chunk_id: int
    video_title: str
    video_url: str
    thumbnail_medium_url: str | None = None
    chunk_index: int
    chunk_level: Literal["global", "section", "detail"] | None = None
    chunk_parent_id: int | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    cohere_relevance_score: float | None = None
    rank_sources: dict[str, int] = Field(default_factory=dict)
    text: str
    persons: list[str] = Field(default_factory=list)
    person_details: list[dict[str, Any]] = Field(default_factory=list)
    transcript: str | None = None
    video_description: str | None = None
    video_type: str | None = None
    section_context: dict[str, Any] | None = None
    global_context: dict[str, Any] | None = None


class RagResponse(BaseModel):
    conversation_id: int
    message_id: int
    answer: str
    action: AnswerAction
    sources: list[ChunkSource]
    retrieval: dict[str, Any]
