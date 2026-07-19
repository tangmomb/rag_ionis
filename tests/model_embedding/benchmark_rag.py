from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

try:
    from tests.model_embedding.benchmark import (
        DEFAULT_CACHE_DIR,
        DEFAULT_CASES_PATH,
        DEFAULT_CONFIGS_PATH,
        DEFAULT_VIDEO_DIR,
        EmbeddingConfig,
        EvalCase,
        Chunk,
        content_hash,
        discover_chunks,
        load_cached_embeddings,
        load_cases,
        load_configs,
        markdown_table,
        metric_percent,
        normalize_rows,
        validate_references,
    )
except ModuleNotFoundError:
    from benchmark import (
        DEFAULT_CACHE_DIR,
        DEFAULT_CASES_PATH,
        DEFAULT_CONFIGS_PATH,
        DEFAULT_VIDEO_DIR,
        EmbeddingConfig,
        EvalCase,
        Chunk,
        content_hash,
        discover_chunks,
        load_cached_embeddings,
        load_cases,
        load_configs,
        markdown_table,
        metric_percent,
        normalize_rows,
        validate_references,
    )


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
DEFAULT_RAG_CONFIGS_PATH = ROOT / "rag_configs.json"
DEFAULT_RAG_CACHE_DIR = ROOT / "cache" / "rag_rerank"
DEFAULT_RAG_REPORT_PATH = ROOT / "reports" / "rag_report.md"


@dataclass(frozen=True)
class RagSettings:
    configuration_names: tuple[str, ...]
    lexical_limit: int
    vector_limit: int
    rrf_limit: int
    rerank_limit: int
    rrf_rank_constant: int
    rerank_model: str
    rerank_min_interval_seconds: float
    top_k: tuple[int, ...]


def load_rag_settings(path: Path, embedding_configs_path: Path) -> tuple[RagSettings, list[EmbeddingConfig]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    settings = RagSettings(
        configuration_names=tuple(str(value) for value in payload["configurations"]),
        lexical_limit=int(payload.get("lexical_limit", 40)),
        vector_limit=int(payload.get("vector_limit", 40)),
        rrf_limit=int(payload.get("rrf_limit", 30)),
        rerank_limit=int(payload.get("rerank_limit", 5)),
        rrf_rank_constant=int(payload.get("rrf_rank_constant", 60)),
        rerank_model=str(payload.get("rerank_model", "rerank-v4.0-fast")),
        rerank_min_interval_seconds=float(payload.get("rerank_min_interval_seconds", 0)),
        top_k=tuple(sorted({int(value) for value in payload.get("top_k", [1, 5, 10])})),
    )
    numeric_values = (
        settings.lexical_limit,
        settings.vector_limit,
        settings.rrf_limit,
        settings.rerank_limit,
        settings.rrf_rank_constant,
        *settings.top_k,
    )
    if (
        not settings.configuration_names
        or not settings.rerank_model
        or settings.rerank_min_interval_seconds < 0
        or any(value <= 0 for value in numeric_values)
    ):
        raise ValueError(f"Configuration RAG invalide dans {path}")
    available, _ = load_configs(embedding_configs_path)
    by_name = {config.name: config for config in available}
    missing = [name for name in settings.configuration_names if name not in by_name]
    if missing:
        raise ValueError(f"Configurations d'embedding absentes: {', '.join(missing)}")
    return settings, [by_name[name] for name in settings.configuration_names]


def fetch_database_chunks(connection: Any) -> dict[str, dict[str, Any]]:
    sql = """
        SELECT
            c.id,
            v.youtube_video_id,
            c.chunk_index,
            c.content,
            v.title,
            v.url,
            v.speakers
        FROM data.chunks c
        JOIN data.videos v ON v.id = c.video_id
        ORDER BY c.id
    """
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
    return {
        f"{row[1]}:{row[2]}": {
            "chunk_id": int(row[0]),
            "key": f"{row[1]}:{row[2]}",
            "text": str(row[3]),
            "video_title": row[4],
            "video_url": row[5],
            "speakers": row[6] or [],
        }
        for row in rows
    }


def fetch_lexical_results(connection: Any, question: str, limit: int) -> tuple[list[dict[str, Any]], float]:
    sql = """
        SELECT
            c.id,
            v.youtube_video_id,
            c.chunk_index,
            c.content,
            v.title,
            v.url,
            v.speakers,
            ts_rank_cd(
                to_tsvector('french', coalesce(c.content, '')),
                websearch_to_tsquery('french', %s)
            ) AS score
        FROM data.chunks c
        JOIN data.videos v ON v.id = c.video_id
        WHERE to_tsvector('french', coalesce(c.content, ''))
              @@ websearch_to_tsquery('french', %s)
        ORDER BY score DESC, c.id ASC
        LIMIT %s
    """
    started_at = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(sql, (question, question, limit))
        rows = cursor.fetchall()
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return [
        {
            "chunk_id": int(row[0]),
            "key": f"{row[1]}:{row[2]}",
            "text": str(row[3]),
            "video_title": row[4],
            "video_url": row[5],
            "speakers": row[6] or [],
            "bm25_score": float(row[7]),
        }
        for row in rows
    ], elapsed_ms


def vector_results(
    chunks: list[Chunk],
    database_chunks: dict[str, dict[str, Any]],
    normalized_chunk_embeddings: np.ndarray,
    normalized_question_embedding: np.ndarray,
    limit: int,
) -> tuple[list[dict[str, Any]], float]:
    started_at = time.perf_counter()
    scores = normalized_chunk_embeddings @ normalized_question_embedding
    resolved_limit = min(limit, len(chunks))
    candidates = np.argpartition(-scores, resolved_limit - 1)[:resolved_limit]
    ordered = candidates[np.argsort(-scores[candidates])]
    results = []
    for index in ordered:
        chunk = chunks[int(index)]
        item = {**database_chunks[chunk.key]}
        item["vector_score"] = float(scores[int(index)])
        results.append(item)
    return results, (time.perf_counter() - started_at) * 1000


def reciprocal_rank_fusion(
    lexical: list[dict[str, Any]],
    vector: list[dict[str, Any]],
    limit: int,
    rank_constant: int,
) -> tuple[list[dict[str, Any]], float]:
    started_at = time.perf_counter()
    scored: dict[int, dict[str, Any]] = {}
    for source_name, results in (("bm25", lexical), ("vector", vector)):
        for rank, item in enumerate(results, start=1):
            chunk_id = int(item["chunk_id"])
            if chunk_id not in scored:
                scored[chunk_id] = {**item, "rrf_score": 0.0, "rank_sources": {}}
            scored[chunk_id][f"{source_name}_score"] = item.get(f"{source_name}_score")
            scored[chunk_id]["rrf_score"] += 1.0 / (rank_constant + rank)
            scored[chunk_id]["rank_sources"][source_name] = rank
    fused = sorted(
        scored.values(),
        key=lambda item: (-item["rrf_score"], item["chunk_id"]),
    )[:limit]
    return fused, (time.perf_counter() - started_at) * 1000


def rerank_signature(question: str, candidates: list[dict[str, Any]], model: str, limit: int) -> str:
    payload = {
        "question": question,
        "model": model,
        "limit": limit,
        "documents": [(item["key"], item["text"]) for item in candidates],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_rerank_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def save_rerank_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rerank_results(
    client: Any,
    question: str,
    candidates: list[dict[str, Any]],
    model: str,
    limit: int,
    cache: dict[str, Any],
    cache_key: str,
    request_state: dict[str, float] | None = None,
    min_request_interval_seconds: float = 0,
) -> tuple[list[dict[str, Any]], float | None, bool]:
    signature = rerank_signature(question, candidates, model, limit)
    cached = cache.get(cache_key)
    by_key = {item["key"]: item for item in candidates}
    if isinstance(cached, dict) and cached.get("signature") == signature:
        cached_results = cached.get("results", [])
        if all(item.get("key") in by_key for item in cached_results):
            cached_elapsed_ms = (
                float(cached["elapsed_ms"])
                if cached.get("timing_kind") == "api" and cached.get("elapsed_ms") is not None
                else None
            )
            return [
                {**by_key[item["key"]], "cohere_relevance_score": float(item["score"])}
                for item in cached_results
            ], cached_elapsed_ms, True

    request_state = request_state if request_state is not None else {}
    response = None
    elapsed_ms = None
    for attempt in range(12):
        wait_seconds = max(0.0, request_state.get("next_request_at", 0.0) - time.monotonic())
        if wait_seconds:
            time.sleep(wait_seconds)
        try:
            request_started_at = time.perf_counter()
            response = client.rerank(
                model=model,
                query=question,
                documents=[item["text"] for item in candidates],
                top_n=min(limit, len(candidates)),
            )
            elapsed_ms = (time.perf_counter() - request_started_at) * 1000
            request_state["next_request_at"] = time.monotonic() + min_request_interval_seconds
            break
        except Exception as error:
            if getattr(error, "status_code", None) != 429 or attempt == 11:
                raise
            request_state["next_request_at"] = time.monotonic() + max(
                min_request_interval_seconds, 6.2
            )
    if response is None:
        raise RuntimeError("Cohere n'a renvoye aucune reponse")
    results: list[dict[str, Any]] = []
    serialized: list[dict[str, Any]] = []
    for result in list(getattr(response, "results", []) or []):
        index = getattr(result, "index", None)
        if not isinstance(index, int) or not 0 <= index < len(candidates):
            continue
        score = float(getattr(result, "relevance_score", 0) or 0)
        selected = {**candidates[index], "cohere_relevance_score": score}
        results.append(selected)
        serialized.append({"key": selected["key"], "score": score})
    if not results:
        raise RuntimeError("Cohere n'a renvoye aucun resultat de reranking exploitable")
    cache[cache_key] = {
        "signature": signature,
        "elapsed_ms": round(elapsed_ms, 3),
        "timing_kind": "api",
        "results": serialized,
    }
    return results, elapsed_ms, False


def first_relevant_rank(results: list[dict[str, Any]], relevant: tuple[str, ...]) -> int | None:
    relevant_set = set(relevant)
    for rank, item in enumerate(results, start=1):
        if item["key"] in relevant_set:
            return rank
    return None


def stage_record(
    configuration: str,
    stage: str,
    case: EvalCase,
    results: list[dict[str, Any]],
    result_limit: int,
    elapsed_ms: float | None,
) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "stage": stage,
        "case_id": case.id,
        "category": case.category,
        "question": case.question,
        "relevant": list(case.relevant),
        "rank": first_relevant_rank(results, case.relevant),
        "result_limit": result_limit,
        "elapsed_ms": elapsed_ms,
        "results": [
            {
                "key": item["key"],
                "score": item.get("cohere_relevance_score", item.get("rrf_score", item.get("vector_score", item.get("bm25_score")))),
            }
            for item in results
        ],
    }


def summarize_records(records: list[dict[str, Any]], top_k: tuple[int, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str | None], list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(
            (item["configuration"], item["stage"], item.get("category")),
            [],
        ).append(item)
    summaries = []
    for (configuration, stage, category), items in sorted(grouped.items(), key=lambda value: str(value[0])):
        ranks = [item["rank"] for item in items]
        result_limit = min(item["result_limit"] for item in items)
        row: dict[str, Any] = {
            "configuration": configuration,
            "stage": stage,
            "category": category,
            "questions": len(items),
            "mrr": sum(0 if rank is None else 1.0 / rank for rank in ranks) / len(items),
            "result_limit": result_limit,
        }
        latencies = [item["elapsed_ms"] for item in items if item["elapsed_ms"] is not None]
        row["mean_latency_ms"] = float(np.mean(latencies)) if latencies else None
        row["p95_latency_ms"] = float(np.percentile(latencies, 95)) if latencies else None
        for value in top_k:
            row[f"recall_at_{value}"] = (
                None
                if value > result_limit
                else sum(rank is not None and rank <= value for rank in ranks) / len(items)
            )
        summaries.append(row)
    return summaries


def summary_table(rows: list[dict[str, Any]], top_k: tuple[int, ...]) -> str:
    includes_categories = any(row.get("category") is not None for row in rows)
    headers = [
        "Configuration",
        "Étape",
        *(["Catégorie"] if includes_categories else []),
        "Questions",
        *[f"Recall@{value}" for value in top_k],
        "MRR",
        "Latence moyenne",
        "Latence p95",
    ]
    table_rows = [
        [
            row["configuration"],
            row["stage"],
            *([row.get("category") or "—"] if includes_categories else []),
            row["questions"],
            *[metric_percent(row.get(f"recall_at_{value}")) for value in top_k],
            metric_percent(row["mrr"]),
            "—" if row["mean_latency_ms"] is None else f"{row['mean_latency_ms']:.2f} ms",
            "—" if row["p95_latency_ms"] is None else f"{row['p95_latency_ms']:.2f} ms",
        ]
        for row in rows
    ]
    return markdown_table(headers, table_rows)


def write_rag_report(
    path: Path,
    settings: RagSettings,
    cases: list[EvalCase],
    records: list[dict[str, Any]],
    cache_hits: int,
    cache_misses: int,
) -> None:
    global_rows = [row for row in summarize_records(records, settings.top_k) if row["category"] is None]
    # Les enregistrements portent toujours une categorie ; on reconstruit donc aussi l'agregat global.
    global_records = [{**item, "category": None} for item in records]
    global_rows = summarize_records(global_records, settings.top_k)
    category_rows = summarize_records(records, settings.top_k)

    final_records = [item for item in records if item["stage"] == "rerank"]
    final_by_config = {
        name: {item["case_id"]: item for item in final_records if item["configuration"] == name}
        for name in settings.configuration_names
    }
    best_or_equal = {name: 0 for name in settings.configuration_names}
    unanimous_equal = 0
    for case in cases:
        ranks = {
            name: final_by_config[name][case.id]["rank"] or math.inf
            for name in settings.configuration_names
        }
        best_rank = min(ranks.values())
        for name, rank in ranks.items():
            if rank == best_rank:
                best_or_equal[name] += 1
        unanimous_equal += int(len(set(ranks.values())) == 1)
    comparison_lines = [
        *[
            f"- **{name} meilleur ou ex æquo après rerank :** {best_or_equal[name]} questions"
            for name in settings.configuration_names
        ],
        f"- **Même rang pour toutes les configurations :** {unanimous_equal} questions",
    ]

    failures = [item for item in final_records if item["rank"] is None]
    failure_lines = ["Aucun échec dans le top final."] if not failures else []
    for item in failures:
        failure_lines.extend(
            [
                f"### {item['configuration']} — {item['case_id']}",
                "",
                item["question"],
                "",
                f"- Attendu : {', '.join(item['relevant'])}",
                f"- Top final : {', '.join(result['key'] for result in item['results'])}",
                "",
            ]
        )

    detail_lines = []
    records_by_case: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        records_by_case.setdefault(item["case_id"], []).append(item)
    for case in cases:
        rows = []
        for item in records_by_case[case.id]:
            rows.append(
                [
                    item["configuration"],
                    item["stage"],
                    item["rank"] or "—",
                    ", ".join(result["key"] for result in item["results"][:5]),
                ]
            )
        detail_lines.extend(
            [
                "<details>",
                f"<summary>{case.id} — {case.question}</summary>",
                "",
                f"**Catégorie :** {case.category or '—'}  ",
                f"**Attendu :** {', '.join(case.relevant)}",
                "",
                markdown_table(["Configuration", "Étape", "Rang", "Top 5"], rows),
                "",
                "</details>",
                "",
            ]
        )

    report = "\n".join(
        [
            "# Rapport du benchmark RAG hybride",
            "",
            "## 1. Configuration",
            "",
            f"- Recherche lexicale PostgreSQL (`ts_rank_cd`) : top {settings.lexical_limit}",
            f"- Recherche vectorielle : top {settings.vector_limit}",
            "- Le classement vectoriel est calculé exactement en mémoire depuis le cache ; sa latence ne représente pas une requête pgvector/HNSW.",
            f"- Fusion RRF (`k={settings.rrf_rank_constant}`) : top {settings.rrf_limit}",
            f"- Reranker Cohere `{settings.rerank_model}` : top {settings.rerank_limit}",
            f"- Intervalle minimal entre appels Cohere : {settings.rerank_min_interval_seconds:g} s",
            f"- Configurations : {', '.join(settings.configuration_names)}",
            f"- Cache rerank : {cache_hits} hits, {cache_misses} nouveaux appels",
            "- Les questions brutes sont utilisées pour isoler l'effet du retrieval ; planner, préfiltres et LLM de réponse sont exclus.",
            "",
            "## 2. Résultats globaux",
            "",
            summary_table(global_rows, settings.top_k),
            "",
            "## 3. Résultats par catégorie",
            "",
            summary_table(category_rows, settings.top_k),
            "",
            "## 4. Comparaison après reranking",
            "",
            *comparison_lines,
            "",
            "## 5. Définition des métriques",
            "",
            "- **Recall@K** : proportion de questions avec un chunk attendu dans les K premiers résultats.",
            "- **MRR** : moyenne de l'inverse du rang du premier chunk attendu.",
            "- **RRF** : fusion des rangs lexical et vectoriel ; un document bien classé par les deux reçoit un score supérieur.",
            "- **Rerank** : reclassement sémantique par Cohere des candidats issus de RRF.",
            "- **—** : métrique non calculable au-delà du nombre de résultats conservés par l'étape.",
            "- Les latences Cohere mesurent uniquement l'appel HTTP, sans l'attente de quota. Les anciens caches sans mesure compatible affichent `—`.",
            "",
            "## 6. Échecs après reranking",
            "",
            *failure_lines,
            "",
            "## 7. Détail par question",
            "",
            *detail_lines,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark hybride lexical + vectoriel + RRF + Cohere.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--embedding-configs", type=Path, default=DEFAULT_CONFIGS_PATH)
    parser.add_argument("--rag-configs", type=Path, default=DEFAULT_RAG_CONFIGS_PATH)
    parser.add_argument("--embedding-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rerank-cache-dir", type=Path, default=DEFAULT_RAG_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_RAG_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_DIR / ".env", override=True)
    settings, embedding_configs = load_rag_settings(args.rag_configs, args.embedding_configs)
    chunks = discover_chunks(args.video_dir)
    cases = load_cases(args.cases)
    validate_references(cases, {chunk.key for chunk in chunks})
    chunk_items = [(chunk.key, chunk.content) for chunk in chunks]
    question_items = [(case.id, case.question) for case in cases]

    normalized_embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for config in embedding_configs:
        chunk_cache = load_cached_embeddings(
            args.embedding_cache_dir,
            config,
            "chunks",
            content_hash(chunk_items),
        )
        question_cache = load_cached_embeddings(
            args.embedding_cache_dir,
            config,
            "questions",
            content_hash(question_items),
        )
        if chunk_cache is None or question_cache is None:
            raise RuntimeError(
                f"Cache embedding manquant pour {config.name}. Lance d'abord benchmark.py."
            )
        chunk_keys, chunk_vectors, _ = chunk_cache
        question_keys, question_vectors, _ = question_cache
        if chunk_keys != [chunk.key for chunk in chunks] or question_keys != [case.id for case in cases]:
            raise ValueError(f"Ordre de cache invalide pour {config.name}")
        normalized_embeddings[config.name] = (
            normalize_rows(chunk_vectors),
            normalize_rows(question_vectors),
        )

    try:
        import cohere
    except ImportError as error:
        raise RuntimeError("Le SDK Cohere est requis pour benchmark_rag.py") from error
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY manquante dans .env")
    cohere_client = cohere.ClientV2(api_key)

    try:
        from interface.backend.database import connect_database
    except ImportError as error:
        raise RuntimeError("Impossible de charger la connexion PostgreSQL du projet") from error

    cache_hits = cache_misses = 0
    rerank_request_state: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    rerank_caches = {
        config.name: load_rerank_cache(args.rerank_cache_dir / f"{config.name}.json")
        for config in embedding_configs
    }
    with connect_database() as connection:
        database_chunks = fetch_database_chunks(connection)
        missing_database_keys = sorted({chunk.key for chunk in chunks} - set(database_chunks))
        if missing_database_keys:
            raise ValueError(f"Chunks absents de PostgreSQL: {', '.join(missing_database_keys[:10])}")

        lexical_by_case: dict[str, tuple[list[dict[str, Any]], float]] = {}
        for case in cases:
            lexical_by_case[case.id] = fetch_lexical_results(
                connection, case.question, settings.lexical_limit
            )
            lexical_results, lexical_ms = lexical_by_case[case.id]
            records.append(
                stage_record(
                    "shared",
                    "lexical",
                    case,
                    lexical_results,
                    settings.lexical_limit,
                    lexical_ms,
                )
            )

        for config in embedding_configs:
            chunk_matrix, question_matrix = normalized_embeddings[config.name]
            rerank_cache = rerank_caches[config.name]
            for case_index, case in enumerate(cases):
                lexical_results, _ = lexical_by_case[case.id]
                vector, vector_ms = vector_results(
                    chunks,
                    database_chunks,
                    chunk_matrix,
                    question_matrix[case_index],
                    settings.vector_limit,
                )
                records.append(
                    stage_record(
                        config.name,
                        "vector",
                        case,
                        vector,
                        settings.vector_limit,
                        vector_ms,
                    )
                )
                fused, rrf_ms = reciprocal_rank_fusion(
                    lexical_results,
                    vector,
                    settings.rrf_limit,
                    settings.rrf_rank_constant,
                )
                records.append(
                    stage_record(
                        config.name,
                        "rrf",
                        case,
                        fused,
                        settings.rrf_limit,
                        rrf_ms,
                    )
                )
                reranked, rerank_ms, cached = rerank_results(
                    cohere_client,
                    case.question,
                    fused,
                    settings.rerank_model,
                    settings.rerank_limit,
                    rerank_cache,
                    case.id,
                    rerank_request_state,
                    settings.rerank_min_interval_seconds,
                )
                cache_hits += int(cached)
                cache_misses += int(not cached)
                records.append(
                    stage_record(
                        config.name,
                        "rerank",
                        case,
                        reranked,
                        settings.rerank_limit,
                        rerank_ms,
                    )
                )
                if not cached:
                    save_rerank_cache(args.rerank_cache_dir / f"{config.name}.json", rerank_cache)
                print(f"[{config.name}] {case_index + 1}/{len(cases)}", flush=True)

    write_rag_report(args.report, settings, cases, records, cache_hits, cache_misses)
    print(f"[ok] rapport RAG: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
