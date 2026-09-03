from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
SqlSubIntent = Literal["analytics"]
AnalyticsScope = Literal["global", "specific"]
AnalyticsMetric = Literal["all", "views", "likes", "comments"]
AnalyticsOrder = Literal["asc", "desc"]
AnswerAction = Literal["answer", "clarify", "abstain"]


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
    def _force_gpt_luna_for_rag_inference(self) -> "RagRequest":
        """Keep every RAG inference stage on the configured OpenAI model."""
        self.reformulationModel = DEFAULT_REFORMULATION_MODEL
        self.plannerModel = DEFAULT_PLANNER_MODEL
        self.answerModel = DEFAULT_GENERATION_MODEL
        return self


class PlannerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: PlannerRoute = "search"
    sql_sub_intent: SqlSubIntent | None = None
    analytics_scope: AnalyticsScope | None = None
    analytics_metric: AnalyticsMetric | None = None
    analytics_order: AnalyticsOrder | None = None
    analytics_rank_start: int | None = Field(default=None, ge=1, le=100)
    analytics_rank_end: int | None = Field(default=None, ge=1, le=100)
    query_text: str
    query_text_bm25: str | None = None
    title_hints: list[str] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None
    description_requested: bool = False
    use_rag: bool = False
    sql_main_source: bool = False

    @model_validator(mode="after")
    def _validate_analytics_scope(self) -> "PlannerPlan":
        if self.sql_sub_intent == "analytics":
            self.analytics_scope = self.analytics_scope or "specific"
            if self.analytics_scope == "global":
                self.analytics_metric = self.analytics_metric or "all"
                self.analytics_rank_start = self.analytics_rank_start or 1
                self.analytics_rank_end = self.analytics_rank_end or 3
            else:
                self.analytics_metric = None
                self.analytics_order = None
                self.analytics_rank_start = None
                self.analytics_rank_end = None
        else:
            self.analytics_scope = None
            self.analytics_metric = None
            self.analytics_order = None
            self.analytics_rank_start = None
            self.analytics_rank_end = None
        if self.analytics_rank_start and self.analytics_rank_end and self.analytics_rank_start > self.analytics_rank_end:
            raise ValueError("analytics_rank_start doit être inférieur ou égal à analytics_rank_end.")
        return self

    @property
    def title_hint(self) -> str | None:
        return self.title_hints[0] if self.title_hints else None


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: PlannerRoute = "search"
    sql_sub_intent: SqlSubIntent | None = None
    analytics_scope: AnalyticsScope | None = None
    analytics_metric: AnalyticsMetric | None = None
    analytics_order: AnalyticsOrder | None = None
    analytics_rank_start: int | None = Field(default=None, ge=1, le=100)
    analytics_rank_end: int | None = Field(default=None, ge=1, le=100)
    raw_question: str
    query_text: str
    query_text_bm25: str
    title_hints: list[str] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None
    description_requested: bool = False
    use_rag: bool = False
    sql_main_source: bool = False
    top_k: int | None = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    final_k: int | None = Field(default=DEFAULT_FINAL_K, ge=1, le=MAX_FINAL_K)

    @model_validator(mode="after")
    def _validate_analytics_scope(self) -> "ExecutionPlan":
        if self.sql_sub_intent == "analytics":
            self.analytics_scope = self.analytics_scope or "specific"
            if self.analytics_scope == "global":
                self.analytics_metric = self.analytics_metric or "all"
                self.analytics_rank_start = self.analytics_rank_start or 1
                self.analytics_rank_end = self.analytics_rank_end or 3
            else:
                self.analytics_metric = None
                self.analytics_order = None
                self.analytics_rank_start = None
                self.analytics_rank_end = None
        else:
            self.analytics_scope = None
            self.analytics_metric = None
            self.analytics_order = None
            self.analytics_rank_start = None
            self.analytics_rank_end = None
        if self.analytics_rank_start and self.analytics_rank_end and self.analytics_rank_start > self.analytics_rank_end:
            raise ValueError("analytics_rank_start doit être inférieur ou égal à analytics_rank_end.")
        return self

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
