"""Évaluation quotidienne de la qualité du RAG en production.

Lancée chaque matin par systemd (deploy/run-monitoring.sh quality), dans un
conteneur ponctuel construit sur l'image de l'API :

1. Canari : quelques questions du golden dataset passent par le pipeline
   complet. Leurs traces vont dans un projet Phoenix séparé
   (PHOENIX_PROJECT_NAME=rag-ionis-canary) et leurs conversations sont
   supprimées ensuite : ni les métriques ni l'historique de prod ne sont
   pollués, et la courbe de qualité existe même sans trafic.
2. Juge : un LLM d'un autre modèle que celui qui répond note chaque réponse
   de la veille (vrai trafic) et du canari : fidélité aux sources et
   pertinence. Les notes sont publiées comme annotations Phoenix.
3. Rapport : volume, erreurs, latence, coût estimé, votes et notes. La
   dernière ligne imprimée est un JSON {"message", "anomaly"} que le script
   hôte envoie sur Telegram ; le conteneur ne connaît pas le token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time as day_time, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
TRAFFIC_PROJECT = "rag-ionis"
DEFAULT_JUDGE_MODEL = "mistral-large-latest"
DEFAULT_CANARY_COUNT = 10
DEFAULT_CANARY_DELAY_SECONDS = 5.0
DEFAULT_DAILY_BUDGET_USD = 1.0
DEFAULT_MIN_FAITHFULNESS = 0.8
DEFAULT_MAX_CANARY_FAILURES = 2
ANNOTATION_IDENTIFIER = "quality-judge"
MAX_JUDGE_CONTEXT_CHARS = 30_000
JUDGE_MAX_OUTPUT_TOKENS = 4_000
SPAN_LIMIT = 5_000

FAITHFULNESS_SCORES = {"fidele": 1.0, "partiellement_fidele": 0.5, "non_fidele": 0.0}
RELEVANCE_SCORES = {"pertinente": 1.0, "partiellement_pertinente": 0.5, "hors_sujet": 0.0}

JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fidelite": {"type": "string", "enum": list(FAITHFULNESS_SCORES)},
        "pertinence": {"type": "string", "enum": list(RELEVANCE_SCORES)},
        "explication": {"type": "string"},
    },
    "required": ["fidelite", "pertinence", "explication"],
    "additionalProperties": False,
}

JUDGE_SYSTEM_PROMPT = """Tu évalues la réponse d'un assistant RAG sur des vidéos YouTube.
On te donne le message envoyé au modèle qui a répondu (la question et les sources numérotées) puis sa réponse.

fidelite :
- fidele : chaque affirmation factuelle de la réponse est appuyée par les sources.
- partiellement_fidele : l'essentiel est appuyé, mais un détail n'apparaît pas dans les sources.
- non_fidele : une affirmation importante est absente des sources ou les contredit.
Sans sources, une réponse qui n'affirme aucun fait (salutation, refus, demande de précision) est fidele.

pertinence :
- pertinente : la réponse traite la question, ou explique clairement pourquoi elle ne peut pas y répondre.
- partiellement_pertinente : elle n'en traite qu'une partie ou s'en écarte en partie.
- hors_sujet : elle ne répond pas à la question posée.

explication : une ou deux phrases en français justifiant les deux notes."""


@dataclass(frozen=True)
class JudgeInput:
    trace_id: str
    span_id: str
    question: str
    answer: str
    context: str | None


@dataclass(frozen=True)
class JudgeResult:
    faithfulness: str
    relevance: str
    explanation: str

    @property
    def faithfulness_score(self) -> float:
        return FAITHFULNESS_SCORES[self.faithfulness]

    @property
    def relevance_score(self) -> float:
        return RELEVANCE_SCORES[self.relevance]


@dataclass
class JudgeSummary:
    judged: int = 0
    errors: int = 0
    faithfulness: list[float] = field(default_factory=list)
    relevance: list[float] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)


@dataclass
class CanarySummary:
    asked: int = 0
    failures: list[str] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    answer_ready_ms: list[float] = field(default_factory=list)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def paris_day_window(day: date) -> tuple[datetime, datetime]:
    """Return the UTC bounds of one calendar day in Europe/Paris."""
    start = datetime.combine(day, day_time.min, tzinfo=PARIS)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], rank: int) -> float | None:
    """Nearest-rank percentile, enough for a few dozen daily requests."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, -(-rank * len(ordered) // 100) - 1))
    return ordered[index]


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if isinstance(value, str) else None
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _span_id(span: Mapping[str, Any]) -> str:
    return str(span["context"]["span_id"])


def _trace_id(span: Mapping[str, Any]) -> str:
    return str(span["context"]["trace_id"])


def judge_input_from_trace(spans: Sequence[Mapping[str, Any]]) -> JudgeInput | None:
    """Build the judge input from the spans of one RAG request trace.

    The question comes from the root ``request`` span, the answer from its
    output, and the context from the exact user message sent to the answer
    model (the LLM span under ``generation``), i.e. what the answer could rely on.
    """
    spans_by_id = {_span_id(span): span for span in spans}
    root = next(
        (span for span in spans if span.get("name") == "request" and not span.get("parent_id")),
        None,
    )
    if root is None:
        return None
    attributes = root.get("attributes") or {}
    question = str(_json_object(attributes.get("input.value")).get("question") or "").strip()
    answer = str(_json_object(attributes.get("output.value")).get("answer") or "").strip()
    if not question or not answer:
        return None
    context = None
    for span in spans:
        parent = spans_by_id.get(str(span.get("parent_id") or ""))
        if span.get("span_kind") != "LLM" or not parent or parent.get("name") != "generation":
            continue
        llm_attributes = span.get("attributes") or {}
        index = 0
        while f"llm.input_messages.{index}.message.role" in llm_attributes:
            if llm_attributes[f"llm.input_messages.{index}.message.role"] == "user":
                context = str(llm_attributes.get(f"llm.input_messages.{index}.message.content") or "")
            index += 1
    if not context:
        # Description and transcript requests are answered with database text,
        # without an answer model: the returned sources are then the context.
        context = sources_context(
            question, _json_object(attributes.get("output.value")).get("sources")
        )
    return JudgeInput(
        trace_id=_trace_id(root),
        span_id=_span_id(root),
        question=question,
        answer=answer,
        context=context[:MAX_JUDGE_CONTEXT_CHARS] if context else None,
    )


def sources_context(question: str, sources: Any) -> str | None:
    if not isinstance(sources, list) or not sources:
        return None
    blocks = []
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        fields = [
            ("Titre", source.get("video_title")),
            ("URL", source.get("video_url")),
            ("Personnes", ", ".join(map(str, source.get("persons") or []))),
            ("Description", source.get("video_description")),
            ("Extrait", source.get("text")),
            ("Transcription", source.get("transcript")),
        ]
        lines = [f"{label} : {value}" for label, value in fields if value]
        if lines:
            blocks.append(f"[S{index}]\n" + "\n".join(lines))
    if not blocks:
        return None
    return "Sources renvoyées avec la réponse :\n\n" + "\n\n".join(blocks) + f"\n\nQuestion : {question}"


def judge_user_prompt(item: JudgeInput) -> str:
    context = item.context or f"(aucune source)\n\nQuestion : {item.question}"
    return (
        f"=== Message envoyé au modèle ===\n{context}\n\n"
        f"=== Réponse de l'assistant ===\n{item.answer}"
    )


def judge_answer(item: JudgeInput, *, model: str) -> JudgeResult:
    from interface.backend.llm_providers import create_llm_response

    last_error: Exception | None = None
    # mistral-large sometimes needs ~3000 output tokens before closing the JSON
    # object; a truncated answer is retried once.
    for _attempt in range(2):
        try:
            response = create_llm_response(
                model=model,
                input=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_user_prompt(item)},
                ],
                response_schema=JUDGE_RESPONSE_SCHEMA,
                max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            )
            return parse_judge_payload(str(response.output_text or ""))
        except Exception as exc:
            last_error = exc
    raise last_error  # type: ignore[misc]


def parse_judge_payload(raw: str) -> JudgeResult:
    payload = json.loads(raw.strip())
    faithfulness = str(payload["fidelite"])
    relevance = str(payload["pertinence"])
    if faithfulness not in FAITHFULNESS_SCORES or relevance not in RELEVANCE_SCORES:
        raise ValueError(f"Labels hors barème : {faithfulness!r}, {relevance!r}")
    return JudgeResult(
        faithfulness=faithfulness,
        relevance=relevance,
        explanation=str(payload["explication"]).strip(),
    )


def annotate(client: Any, item: JudgeInput, result: JudgeResult, *, model: str) -> None:
    for name, label, score in (
        ("fidelite", result.faithfulness, result.faithfulness_score),
        ("pertinence", result.relevance, result.relevance_score),
    ):
        client.spans.add_span_annotation(
            span_id=item.span_id,
            annotation_name=name,
            annotator_kind="LLM",
            label=label,
            score=score,
            explanation=result.explanation,
            metadata={"judge_model": model},
            # Re-running the same day updates the annotation instead of adding one.
            identifier=ANNOTATION_IDENTIFIER,
            sync=True,
        )


def judge_traces(
    client: Any,
    project: str,
    trace_ids: Iterable[str],
    *,
    model: str,
    judge: Callable[..., JudgeResult] = judge_answer,
) -> JudgeSummary:
    summary = JudgeSummary()
    for trace_id in trace_ids:
        try:
            spans = client.spans.get_spans(
                project_identifier=project, trace_ids=[trace_id], limit=SPAN_LIMIT
            )
            item = judge_input_from_trace(spans)
            if item is None:
                continue
            result = judge(item, model=model)
            annotate(client, item, result, model=model)
        except Exception as exc:  # one bad trace must not stop the report
            summary.errors += 1
            print(f"[quality] judge failed for {trace_id}: {exc}", file=sys.stderr)
            continue
        summary.judged += 1
        summary.faithfulness.append(result.faithfulness_score)
        summary.relevance.append(result.relevance_score)
        if result.faithfulness == "non_fidele" or result.relevance == "hors_sujet":
            summary.flagged.append(f"{item.question[:80]} → {result.explanation[:160]}")
    return summary


def select_canary_questions(questions: Sequence[str], count: int, day: date) -> list[str]:
    """Rotate through the golden questions: a different slice every day."""
    if not questions or count <= 0:
        return []
    count = min(count, len(questions))
    offset = (day.toordinal() * count) % len(questions)
    return [questions[(offset + index) % len(questions)] for index in range(count)]


def delete_conversations(conversation_ids: Sequence[int]) -> None:
    if not conversation_ids:
        return
    from interface.backend.database import connect_database

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chat.conversations WHERE id = ANY(%s)",
                (list(conversation_ids),),
            )
        connection.commit()


def run_canary(questions: Sequence[str], *, delay_seconds: float) -> CanarySummary:
    from interface.backend.api import run_rag
    from interface.backend.llm_providers import track_llm_costs
    from interface.backend.schemas import RagRequest

    summary = CanarySummary()
    conversation_ids: list[int] = []
    try:
        for index, question in enumerate(questions):
            if index and delay_seconds:
                time.sleep(delay_seconds)
            summary.asked += 1
            try:
                with track_llm_costs() as costs:
                    response = run_rag(
                        RagRequest(question=question),
                        stream_callback=lambda _fragment: None,
                    )
            except Exception as exc:
                summary.failures.append(f"{question[:80]} : {exc}")
                continue
            conversation_ids.append(response.conversation_id)
            summary.cost_usd += costs.total_usd
            trace_id = (response.retrieval.get("telemetry") or {}).get("trace_id")
            if trace_id:
                summary.trace_ids.append(str(trace_id))
            ready_ms = response.retrieval.get("rag_answer_ready_ms")
            if isinstance(ready_ms, (int, float)):
                summary.answer_ready_ms.append(float(ready_ms))
    finally:
        delete_conversations(conversation_ids)
    return summary


def estimated_cost_of_llm_spans(spans: Iterable[Mapping[str, Any]]) -> tuple[float, int]:
    """Sum the estimated cost of LLM spans; returns (usd, unpriced span count)."""
    from interface.backend.llm_providers import estimated_llm_cost_usd

    total = 0.0
    unpriced = 0
    for span in spans:
        attributes = span.get("attributes") or {}
        cost = estimated_llm_cost_usd(
            attributes.get("llm.model_name"),
            attributes.get("llm.token_count.prompt"),
            attributes.get("llm.token_count.completion"),
        )
        if cost is None:
            unpriced += 1
            continue
        total += sum(cost)
    return total, unpriced


def feedback_counts(start: datetime, end: datetime) -> tuple[int, int]:
    from interface.backend.database import connect_database

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*) FILTER (WHERE m.feedback),
                       count(*) FILTER (WHERE NOT m.feedback)
                FROM chat.messages m
                JOIN chat.conversations c ON c.id = m.conversation_id
                WHERE c.date >= %s AND c.date < %s
                """,
                (start, end),
            )
            positive, negative = cursor.fetchone()
    return int(positive or 0), int(negative or 0)


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _format_ms(value: float | None) -> str:
    return "—" if value is None else f"{value / 1000:.1f} s"


def build_report(
    *,
    day: date,
    requests: int,
    errors: int,
    answer_ready_ms: Sequence[float],
    traffic_cost_usd: float,
    feedback: tuple[int, int],
    traffic_judge: JudgeSummary,
    canary: CanarySummary | None,
    canary_judge: JudgeSummary | None,
    budget_usd: float,
    min_faithfulness: float,
    max_canary_failures: int,
) -> tuple[str, list[str]]:
    anomalies: list[str] = []
    if errors:
        anomalies.append(f"{errors} requête(s) en erreur")
    if traffic_cost_usd > budget_usd:
        anomalies.append(f"coût {traffic_cost_usd:.2f} $ > budget {budget_usd:.2f} $")
    lines = [
        f"Trafic : {requests} requête(s), {errors} erreur(s)",
        f"Latence réponse prête : p50 {_format_ms(percentile(answer_ready_ms, 50))}, "
        f"p95 {_format_ms(percentile(answer_ready_ms, 95))}",
        f"Coût LLM estimé : {traffic_cost_usd:.3f} $ (budget {budget_usd:.2f} $)",
        f"Votes : {feedback[0]} 👍 / {feedback[1]} 👎",
        f"Juge (trafic) : fidélité {_format_score(mean(traffic_judge.faithfulness))}, "
        f"pertinence {_format_score(mean(traffic_judge.relevance))} "
        f"sur {traffic_judge.judged} réponse(s)",
    ]
    if canary is not None and canary_judge is not None:
        canary_faithfulness = mean(canary_judge.faithfulness)
        if len(canary.failures) > max_canary_failures:
            anomalies.append(f"{len(canary.failures)} question(s) canari en échec")
        if canary_faithfulness is not None and canary_faithfulness < min_faithfulness:
            anomalies.append(
                f"fidélité canari {canary_faithfulness:.0%} < {min_faithfulness:.0%}"
            )
        lines.append(
            f"Canari : {canary.asked - len(canary.failures)}/{canary.asked} réponses, "
            f"fidélité {_format_score(canary_faithfulness)}, "
            f"pertinence {_format_score(mean(canary_judge.relevance))}, "
            f"p50 {_format_ms(percentile(canary.answer_ready_ms, 50))}, "
            f"{canary.cost_usd:.3f} $"
        )
    flagged = traffic_judge.flagged + (canary_judge.flagged if canary_judge else [])
    if flagged:
        lines.append("À relire :")
        lines.extend(f"• {entry}" for entry in flagged[:5])
    judge_errors = traffic_judge.errors + (canary_judge.errors if canary_judge else 0)
    judged = traffic_judge.judged + (canary_judge.judged if canary_judge else 0)
    if judge_errors:
        lines.append(f"{judge_errors} évaluation(s) du juge en échec")
        if not judged:
            anomalies.append("juge LLM indisponible")
    status = "🔴" if anomalies else "🟢"
    header = f"{status} Qualité RAG IONIS · {day.isoformat()}"
    if anomalies:
        header += "\n" + " ; ".join(anomalies)
    return header + "\n\n" + "\n".join(lines), anomalies


def phoenix_client() -> Any:
    from phoenix.client import Client

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    return Client(base_url=endpoint.rstrip("/").removesuffix("/v1/traces"))


def run_daily(args: argparse.Namespace) -> dict[str, Any]:
    from interface.backend.golden_recall import load_golden_cases
    from interface.backend.telemetry import configure_telemetry, shutdown_telemetry

    model = os.getenv("QUALITY_JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL
    day = args.date or (datetime.now(PARIS).date() - timedelta(days=1))
    start, end = paris_day_window(day)
    client = phoenix_client()

    canary = canary_judge = None
    canary_project = os.getenv("PHOENIX_PROJECT_NAME", "rag-ionis-canary")
    if args.canary_count > 0:
        if canary_project == TRAFFIC_PROJECT:
            raise SystemExit("Le canari doit tracer dans un projet Phoenix distinct de la prod.")
        configure_telemetry()
        questions = [case.question for case in load_golden_cases(os.environ["GOLDEN_DATASET_URL"])]
        canary = run_canary(
            select_canary_questions(questions, args.canary_count, datetime.now(PARIS).date()),
            delay_seconds=args.canary_delay_seconds,
        )
        # Flush the canary spans so the judge can read them back from Phoenix.
        shutdown_telemetry()
        time.sleep(5)

    root_spans = client.spans.get_spans(
        project_identifier=TRAFFIC_PROJECT,
        name="request",
        start_time=start,
        end_time=end,
        limit=SPAN_LIMIT,
    )
    root_spans = [span for span in root_spans if not span.get("parent_id")]
    llm_spans = client.spans.get_spans(
        project_identifier=TRAFFIC_PROJECT,
        span_kind="LLM",
        start_time=start,
        end_time=end,
        limit=SPAN_LIMIT,
    )
    traffic_cost_usd, _unpriced = estimated_cost_of_llm_spans(llm_spans)
    answer_ready_ms = [
        float(value)
        for span in root_spans
        if isinstance(value := (span.get("attributes") or {}).get("rag.answer_ready_ms"), (int, float))
    ]
    traffic_judge = judge_traces(
        client, TRAFFIC_PROJECT, [_trace_id(span) for span in root_spans], model=model
    )
    if canary is not None:
        canary_judge = judge_traces(client, canary_project, canary.trace_ids, model=model)

    message, anomalies = build_report(
        day=day,
        requests=len(root_spans),
        errors=sum(1 for span in root_spans if span.get("status_code") == "ERROR"),
        answer_ready_ms=answer_ready_ms,
        traffic_cost_usd=traffic_cost_usd,
        feedback=feedback_counts(start, end),
        traffic_judge=traffic_judge,
        canary=canary,
        canary_judge=canary_judge,
        budget_usd=env_float("QUALITY_DAILY_BUDGET_USD", DEFAULT_DAILY_BUDGET_USD),
        min_faithfulness=env_float("QUALITY_MIN_FAITHFULNESS", DEFAULT_MIN_FAITHFULNESS),
        max_canary_failures=int(
            env_float("QUALITY_MAX_CANARY_FAILURES", DEFAULT_MAX_CANARY_FAILURES)
        ),
    )
    return {"message": message, "anomaly": bool(anomalies), "anomalies": anomalies}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)
    daily = subcommands.add_parser("daily", help="Canari, juge et rapport de la veille.")
    daily.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Jour du trafic à évaluer (AAAA-MM-JJ, heure de Paris). Défaut : la veille.",
    )
    daily.add_argument("--canary-count", type=int, default=DEFAULT_CANARY_COUNT)
    daily.add_argument(
        "--canary-delay-seconds", type=float, default=DEFAULT_CANARY_DELAY_SECONDS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_daily(args)
    # Last line: read by deploy/run-monitoring.sh, which sends the message.
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
