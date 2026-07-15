from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai import OpenAI


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT.parents[1]
DEFAULT_VIDEO_DIR = PROJECT_DIR / "downloads" / "youtube"
DEFAULT_CONFIGS_PATH = ROOT / "configs.json"
DEFAULT_CASES_PATH = ROOT / "cases.jsonl"
DEFAULT_CACHE_DIR = ROOT / "cache"
DEFAULT_REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class EmbeddingConfig:
    name: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class Chunk:
    key: str
    video_key: str
    chunk_index: str
    content: str
    source: str
    speakers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    relevant: tuple[str, ...]
    category: str | None = None
    answerable: bool = True


def normalized_index(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def chunk_key(video_key: Any, chunk_index: Any) -> str:
    video = str(video_key).strip()
    index = normalized_index(chunk_index)
    if not video or not index:
        raise ValueError("video_key et chunk_index sont obligatoires")
    return f"{video}:{index}"


def load_configs(path: Path) -> tuple[list[EmbeddingConfig], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_configs = payload.get("configurations")
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError(f"Aucune configuration dans {path}")

    configs: list[EmbeddingConfig] = []
    names: set[str] = set()
    for item in raw_configs:
        config = EmbeddingConfig(
            name=str(item["name"]).strip(),
            model=str(item["model"]).strip(),
            dimensions=int(item["dimensions"]),
        )
        if not config.name or not config.model or config.dimensions <= 0:
            raise ValueError(f"Configuration invalide: {item!r}")
        if config.name in names:
            raise ValueError(f"Nom de configuration duplique: {config.name}")
        names.add(config.name)
        configs.append(config)

    top_k = sorted({int(value) for value in payload.get("top_k", [1, 5, 10])})
    if not top_k or top_k[0] <= 0:
        raise ValueError("top_k doit contenir des entiers strictement positifs")
    return configs, top_k


def relevant_key(value: Any) -> str:
    if isinstance(value, str):
        if ":" not in value:
            raise ValueError(f"Cle pertinente invalide: {value!r}")
        return value.strip()
    if isinstance(value, dict):
        return chunk_key(value.get("video_key"), value.get("chunk_index"))
    raise ValueError(f"Reference de chunk invalide: {value!r}")


def load_cases(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(
            f"Jeu d'evaluation absent: {path}. Copie cases.example.jsonl vers cases.jsonl puis adapte-le."
        )

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON invalide dans {path}, ligne {line_number}: {error}") from error

        case_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        relevant = tuple(dict.fromkeys(relevant_key(value) for value in item.get("relevant", [])))
        answerable = bool(item.get("answerable", True))
        if not case_id or not question:
            raise ValueError(f"id et question obligatoires dans {path}, ligne {line_number}")
        if case_id in seen_ids:
            raise ValueError(f"Identifiant de question duplique: {case_id}")
        if answerable and not relevant:
            raise ValueError(f"La question {case_id} est answerable mais n'a aucun chunk pertinent")
        if not answerable and relevant:
            raise ValueError(f"La question {case_id} est non answerable mais possede des chunks pertinents")
        seen_ids.add(case_id)
        cases.append(
            EvalCase(
                id=case_id,
                question=question,
                relevant=relevant,
                category=str(item.get("category") or "").strip() or None,
                answerable=answerable,
            )
        )

    if not cases:
        raise ValueError(f"Aucune question dans {path}")
    return cases


def chunk_source_priority(path: Path) -> int | None:
    name = path.name
    if name == "transcript_chunks_speaker_validated.json":
        return 0
    if name.endswith("_chunks_corrected.json"):
        return 1
    if name == "transcript_chunks.json":
        return 2
    if name.endswith("_chunks.json") and not name.endswith("_embedding.json"):
        return 3
    return None


def video_base_for_chunk_file(path: Path) -> Path:
    chunks_dir = path.parent
    if chunks_dir.name == "chunks" and chunks_dir.parent.name == "outputs":
        return chunks_dir.parent.parent
    if chunks_dir.name == "chunks":
        return chunks_dir.parent
    return chunks_dir


def discover_chunk_sources(video_dir: Path) -> list[tuple[Path, Path]]:
    selected: dict[Path, tuple[int, Path]] = {}
    for path in video_dir.rglob("*.json"):
        priority = chunk_source_priority(path)
        if priority is None:
            continue
        video_base = video_base_for_chunk_file(path).resolve()
        current = selected.get(video_base)
        if current is None or priority < current[0]:
            selected[video_base] = (priority, path)
    return [(video_base, value[1]) for video_base, value in sorted(selected.items(), key=lambda item: str(item[0]))]


def discover_chunks(video_dir: Path) -> list[Chunk]:
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Dossier de videos introuvable: {video_dir}")

    chunks: list[Chunk] = []
    seen_keys: dict[str, str] = {}
    for video_base, source in discover_chunk_sources(video_dir):
        payload = json.loads(source.read_text(encoding="utf-8"))
        raw_chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        if not isinstance(raw_chunks, list):
            continue
        video_key = video_base.name
        for item in raw_chunks:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            index = normalized_index(item.get("chunk_index"))
            if not content or not index or index == "None":
                continue
            key = chunk_key(video_key, index)
            if key in seen_keys:
                raise ValueError(f"Cle de chunk dupliquee {key}: {seen_keys[key]} et {source}")
            seen_keys[key] = str(source)
            meta_data = item.get("meta_data") if isinstance(item.get("meta_data"), dict) else {}
            raw_speakers = meta_data.get("speakers") or item.get("speakers") or []
            speakers = tuple(
                str(value).strip()
                for value in raw_speakers
                if str(value).strip()
            ) if isinstance(raw_speakers, list) else ()
            chunks.append(
                Chunk(
                    key=key,
                    video_key=video_key,
                    chunk_index=index,
                    content=content,
                    source=str(source),
                    speakers=speakers,
                )
            )
    if not chunks:
        raise ValueError(f"Aucun transcript_chunks*.json exploitable dans {video_dir}")
    return chunks


def content_hash(items: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for key, text in sorted(items):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Un embedding nul ne peut pas etre normalise")
    return matrix / norms


def cache_paths(cache_dir: Path, config: EmbeddingConfig, kind: str) -> tuple[Path, Path]:
    config_dir = cache_dir / config.name
    return config_dir / f"{kind}.npz", config_dir / f"{kind}.manifest.json"


def load_cached_embeddings(
    cache_dir: Path,
    config: EmbeddingConfig,
    kind: str,
    expected_hash: str,
) -> tuple[list[str], np.ndarray, dict[str, Any]] | None:
    data_path, manifest_path = cache_paths(cache_dir, config, kind)
    if not data_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model": config.model,
        "dimensions": config.dimensions,
        "content_hash": expected_hash,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    with np.load(data_path, allow_pickle=False) as payload:
        keys = [str(value) for value in payload["keys"].tolist()]
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.shape != (len(keys), config.dimensions):
        return None
    return keys, embeddings, manifest


def embed_and_cache(
    client: "OpenAI",
    cache_dir: Path,
    config: EmbeddingConfig,
    kind: str,
    items: list[tuple[str, str]],
    batch_size: int,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    expected_hash = content_hash(items)
    cached = load_cached_embeddings(cache_dir, config, kind, expected_hash)
    if cached is not None:
        print(f"[cache] {config.name}/{kind}: {len(cached[0])} embeddings")
        return cached

    keys = [key for key, _ in items]
    texts = [text for _, text in items]
    vectors: list[list[float]] = []
    prompt_tokens = 0
    started_at = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=config.model,
            dimensions=config.dimensions,
            input=batch,
        )
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        prompt_tokens += int(getattr(response.usage, "prompt_tokens", 0) or 0)
        print(f"[embed] {config.name}/{kind}: {min(start + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.shape != (len(keys), config.dimensions):
        raise ValueError(
            f"Dimensions inattendues pour {config.name}/{kind}: {embeddings.shape}, "
            f"attendu {(len(keys), config.dimensions)}"
        )

    data_path, manifest_path = cache_paths(cache_dir, config, kind)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, keys=np.asarray(keys), embeddings=embeddings)
    manifest = {
        "model": config.model,
        "dimensions": config.dimensions,
        "content_hash": expected_hash,
        "count": len(keys),
        "prompt_tokens": prompt_tokens,
        "embedding_seconds": round(time.perf_counter() - started_at, 3),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return keys, embeddings, manifest


def validate_references(cases: list[EvalCase], chunk_keys: set[str]) -> None:
    missing = sorted({key for case in cases for key in case.relevant if key not in chunk_keys})
    if missing:
        preview = "\n".join(f"- {key}" for key in missing[:20])
        suffix = f"\n... et {len(missing) - 20} autres" if len(missing) > 20 else ""
        raise ValueError(f"Chunks pertinents absents du corpus:\n{preview}{suffix}")


def evaluate_configuration(
    config: EmbeddingConfig,
    chunks: list[Chunk],
    cases: list[EvalCase],
    chunk_embeddings: np.ndarray,
    question_embeddings: np.ndarray,
    top_k: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_chunks = normalize_rows(chunk_embeddings)
    normalized_questions = normalize_rows(question_embeddings)
    key_to_index = {chunk.key: index for index, chunk in enumerate(chunks)}
    max_k = min(max(top_k), len(chunks))
    sums = {value: 0 for value in top_k}
    reciprocal_rank_sum = 0.0
    evaluated = 0
    details: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases):
        scores = normalized_chunks @ normalized_questions[case_index]
        if max_k == len(scores):
            top_indices = np.argsort(-scores)[:max_k]
        else:
            candidates = np.argpartition(-scores, max_k - 1)[:max_k]
            top_indices = candidates[np.argsort(-scores[candidates])]
        top_results = [
            {"key": chunks[int(index)].key, "score": round(float(scores[int(index)]), 8)}
            for index in top_indices
        ]

        first_relevant_rank: int | None = None
        if case.answerable:
            evaluated += 1
            best_relevant_score = max(float(scores[key_to_index[key]]) for key in case.relevant)
            first_relevant_rank = 1 + int(np.count_nonzero(scores > best_relevant_score))
            reciprocal_rank_sum += 1.0 / first_relevant_rank
            top_keys = [item["key"] for item in top_results]
            for value in top_k:
                if set(top_keys[:value]).intersection(case.relevant):
                    sums[value] += 1

        details.append(
            {
                "configuration": config.name,
                "case_id": case.id,
                "question": case.question,
                "category": case.category,
                "answerable": case.answerable,
                "relevant": list(case.relevant),
                "first_relevant_rank": first_relevant_rank,
                "top_results": top_results,
            }
        )

    summary: dict[str, Any] = {
        "configuration": config.name,
        "model": config.model,
        "dimensions": config.dimensions,
        "evaluated_questions": evaluated,
        "mrr": round(reciprocal_rank_sum / evaluated, 6) if evaluated else None,
        "bytes_per_vector": 4 * config.dimensions + 8,
    }
    for value in top_k:
        summary[f"recall_at_{value}"] = round(sums[value] / evaluated, 6) if evaluated else None
    return summary, details


def write_catalog(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["key", "video_key", "chunk_index", "speakers", "content", "source"],
        )
        writer.writeheader()
        for chunk in chunks:
            writer.writerow(
                {
                    "key": chunk.key,
                    "video_key": chunk.video_key,
                    "chunk_index": chunk.chunk_index,
                    "speakers": " | ".join(chunk.speakers),
                    "content": chunk.content,
                    "source": chunk.source,
                }
            )


def markdown_cell(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def metric_percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f} %"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(markdown_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(markdown_cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


METRICS_DEFINITIONS_MARKDOWN = """## Définition des métriques

- **Recall@K** : proportion de questions pour lesquelles au moins un chunk attendu apparaît parmi les K premiers résultats. Par exemple, Recall@5 mesure la présence d'un bon chunk dans les cinq premiers résultats. Plus la valeur est proche de 100 %, mieux c'est.
- **MRR (Mean Reciprocal Rank)** : moyenne de l'inverse du rang du premier chunk pertinent. Un bon chunk classé 1er vaut `1`, classé 2e vaut `0,5`, classé 3e vaut `0,333`. Cette métrique récompense les bons résultats placés très haut.
- **Questions évaluées** : nombre de questions répondables incluses dans le calcul. Les éventuels cas `answerable: false` sont exclus de Recall@K et MRR.
- **Octets/vecteur** : estimation du stockage brut d'un embedding pgvector de type `vector`, calculée avec `4 × dimensions + 8`. Elle n'inclut pas les autres colonnes ni l'index HNSW.
- **Tokens chunks** : nombre de tokens envoyés à l'API pour calculer les embeddings du corpus. Le cache évite de les recalculer aux lancements suivants.
- **Tokens questions** : nombre de tokens envoyés pour calculer les embeddings des questions du test.
- **Temps embeddings** : durée cumulée des appels de création d'embeddings lors de leur génération initiale. Une valeur issue du cache peut provenir du premier calcul.
"""


def write_reports(
    reports_dir: Path,
    summaries: list[dict[str, Any]],
    details: list[dict[str, Any]],
    top_k: list[int],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_headers = [
        "Configuration",
        "Modèle",
        "Dimensions",
        "Questions",
        *[f"Recall@{value}" for value in top_k],
        "MRR",
        "Octets/vecteur",
        "Tokens chunks",
        "Tokens questions",
        "Temps embeddings",
    ]
    summary_rows = [
        [
            item["configuration"],
            item["model"],
            item["dimensions"],
            item["evaluated_questions"],
            *[metric_percent(item.get(f"recall_at_{value}")) for value in top_k],
            metric_percent(item.get("mrr")),
            item["bytes_per_vector"],
            item.get("chunk_prompt_tokens", 0),
            item.get("question_prompt_tokens", 0),
            f"{float(item.get('embedding_seconds', 0)):.3f} s",
        ]
        for item in summaries
    ]
    summary_section = (
        "## 1. Résumé global\n\n"
        + markdown_table(summary_headers, summary_rows)
        + "\n\n"
        + "Les valeurs Recall@K et MRR sont affichées en pourcentage.\n"
    )

    category_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in details:
        if item["answerable"] and item.get("category"):
            grouped.setdefault((item["configuration"], item["category"]), []).append(item)
    for (configuration, category), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "configuration": configuration,
            "category": category,
            "evaluated_questions": len(items),
            "mrr": round(
                sum(1.0 / item["first_relevant_rank"] for item in items) / len(items),
                6,
            ),
        }
        for value in top_k:
            row[f"recall_at_{value}"] = round(
                sum(item["first_relevant_rank"] <= value for item in items) / len(items),
                6,
            )
        category_rows.append(row)
    category_section = "## 2. Résultats par catégorie\n\nAucune catégorie renseignée.\n"
    if category_rows:
        category_headers = [
            "Configuration",
            "Catégorie",
            "Questions",
            *[f"Recall@{value}" for value in top_k],
            "MRR",
        ]
        category_markdown_rows = [
            [
                item["configuration"],
                item["category"],
                item["evaluated_questions"],
                *[metric_percent(item.get(f"recall_at_{value}")) for value in top_k],
                metric_percent(item.get("mrr")),
            ]
            for item in category_rows
        ]
        category_section = (
            "## 2. Résultats par catégorie\n\n"
            + markdown_table(category_headers, category_markdown_rows)
            + "\n"
        )

    question_lines = [
        "## 5. Détail par question",
        "",
        "- **Rang du premier chunk pertinent** : position du premier chunk attendu dans le classement ; 1 est le meilleur résultat.",
        "- **Score cosinus** : similarité entre la question et le chunk après normalisation. Une valeur plus élevée indique une plus grande proximité sémantique, mais les scores bruts ne doivent pas être comparés comme un seuil universel entre modèles.",
        "",
    ]
    for item in details:
        question_lines.extend(
            [
                f"### {item['configuration']} — {item['case_id']}",
                "",
                f"**Catégorie :** {item.get('category') or '—'}  ",
                f"**Question :** {item['question']}  ",
                f"**Chunks attendus :** {', '.join(item['relevant']) or '—'}  ",
                f"**Rang du premier chunk pertinent :** {item['first_relevant_rank'] or '—'}",
                "",
                markdown_table(
                    ["Rang", "Chunk", "Score cosinus"],
                    [
                        [rank, result["key"], result["score"]]
                        for rank, result in enumerate(item["top_results"], start=1)
                    ],
                ),
                "",
            ]
        )
    max_k = max(top_k)
    failures = [
        item
        for item in details
        if item["answerable"] and (item["first_relevant_rank"] or math.inf) > max_k
    ]
    lines = [f"## 4. Échecs au Recall@{max_k}", ""]
    if not failures:
        lines.append("Aucun échec.")
    for item in failures:
        lines.extend(
            [
                f"### {item['configuration']} — {item['case_id']}",
                "",
                item["question"],
                "",
                f"- Rang du premier chunk pertinent : {item['first_relevant_rank']}",
                f"- Attendu : {', '.join(item['relevant'])}",
                f"- Top obtenu : {', '.join(result['key'] for result in item['top_results'])}",
                "",
            ]
        )
    report = "\n".join(
        [
            "# Rapport du benchmark des embeddings",
            "",
            "## Sommaire",
            "",
            "1. Résumé global",
            "2. Résultats par catégorie",
            "3. Définition des métriques",
            f"4. Échecs au Recall@{max_k}",
            "5. Détail par question",
            "",
            summary_section,
            "",
            category_section,
            "",
            METRICS_DEFINITIONS_MARKDOWN.replace(
                "## Définition des métriques", "## 3. Définition des métriques"
            ).rstrip(),
            "",
            "\n".join(lines),
            "",
            "\n".join(question_lines),
            "",
        ]
    )
    (reports_dir / "report.md").write_text(report, encoding="utf-8")

    for legacy_name in (
        "summary.md",
        "summary_by_category.md",
        "per_question.md",
        "failures.md",
        "summary.csv",
        "summary_by_category.csv",
        "per_question.json",
    ):
        legacy_path = reports_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare plusieurs modeles et dimensions d'embeddings.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"Racine contenant les dossiers de videos. Defaut: {DEFAULT_VIDEO_DIR}",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Exporte uniquement le catalogue des chunks, sans appeler OpenAI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size doit etre strictement positif")

    chunks = discover_chunks(args.video_dir)
    print(f"[corpus] {len(chunks)} chunks decouverts")
    if args.catalog_only:
        catalog_path = args.reports_dir / "chunk_catalog.csv"
        write_catalog(chunks, catalog_path)
        print(f"[ok] catalogue: {catalog_path}")
        return 0

    configs, top_k = load_configs(args.configs)
    cases = load_cases(args.cases)
    validate_references(cases, {chunk.key for chunk in chunks})
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Le SDK openai est requis pour calculer les embeddings. "
            "Active l'environnement virtuel du projet ou installe requirements.txt."
        ) from error
    client = OpenAI()

    chunk_items = [(chunk.key, chunk.content) for chunk in chunks]
    question_items = [(case.id, case.question) for case in cases]
    summaries: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    for config in configs:
        chunk_keys, chunk_vectors, chunk_manifest = embed_and_cache(
            client, args.cache_dir, config, "chunks", chunk_items, args.batch_size
        )
        question_keys, question_vectors, question_manifest = embed_and_cache(
            client, args.cache_dir, config, "questions", question_items, args.batch_size
        )
        if chunk_keys != [chunk.key for chunk in chunks]:
            raise ValueError(f"Ordre du cache chunks invalide pour {config.name}")
        if question_keys != [case.id for case in cases]:
            raise ValueError(f"Ordre du cache questions invalide pour {config.name}")

        summary, details = evaluate_configuration(
            config, chunks, cases, chunk_vectors, question_vectors, top_k
        )
        summary["chunk_prompt_tokens"] = int(chunk_manifest.get("prompt_tokens", 0))
        summary["question_prompt_tokens"] = int(question_manifest.get("prompt_tokens", 0))
        summary["embedding_seconds"] = round(
            float(chunk_manifest.get("embedding_seconds", 0))
            + float(question_manifest.get("embedding_seconds", 0)),
            3,
        )
        summaries.append(summary)
        all_details.extend(details)
        recalls = ", ".join(f"R@{value}={summary[f'recall_at_{value}']:.3f}" for value in top_k)
        print(f"[resultat] {config.name}: {recalls}, MRR={summary['mrr']:.3f}")

    write_reports(args.reports_dir, summaries, all_details, top_k)
    print(f"[ok] rapports: {args.reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
