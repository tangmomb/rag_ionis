import json
from collections import Counter
from pathlib import Path
from typing import Any

import dagster as dg
from dagster import AssetCheckExecutionContext, AssetExecutionContext, SensorEvaluationContext


PROJECT_DIR = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = PROJECT_DIR / "downloads" / "youtube"
EXPLORER_URL = "http://127.0.0.1:8003/videos"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_PARTITIONS = dg.DynamicPartitionsDefinition(name="youtube_videos")
GROUP_NAME = "rag_ionis_video"


def read_json(path: Path | None) -> Any:
    if path is None or not path.is_file() or path.stat().st_size > 50 * 1024 * 1024:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def read_text(path: Path | None, limit: int = 8_000) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except (OSError, UnicodeDecodeError):
        return ""


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def discover_local_videos() -> dict[str, Path]:
    """Return the newest local directory for every YouTube video id."""
    discovered: dict[str, Path] = {}
    if not DOWNLOAD_ROOT.is_dir():
        return discovered
    for run_dir in sorted(DOWNLOAD_ROOT.glob("*_init*"), reverse=True):
        if not run_dir.is_dir():
            continue
        for video_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            video_file = next(
                (path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
                None,
            )
            if video_file is None:
                continue
            metadata = read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
            video_id = str(metadata.get("youtube_video_id") or video_dir.name)
            discovered.setdefault(video_id, video_dir)
    return discovered


def selected_video(context: AssetExecutionContext | AssetCheckExecutionContext) -> tuple[str, Path]:
    video_id = context.partition_key
    video_dir = discover_local_videos().get(video_id)
    if video_dir is None:
        raise dg.Failure(
            description=f"La vidéo {video_id} n'existe plus dans {DOWNLOAD_ROOT}. Actualise les partitions Dagster."
        )
    return video_id, video_dir


def output_paths(video_dir: Path) -> dict[str, Path | None]:
    transcript_dirs = sorted(path for path in (video_dir / "outputs").glob("transcripts*") if path.is_dir())
    return {
        "analysis": first_existing(
            [video_dir / "metadata" / "pipeline_analysis.json", video_dir / "outputs" / "metadata" / "pipeline_analysis.json"]
        ),
        "summary": first_existing([path / "video_summary.md" for path in transcript_dirs]),
        "transcript": first_existing([path / "plain_transcript.txt" for path in transcript_dirs]),
        "ocr": first_existing(
            [
                video_dir / "outputs" / "ocr" / "03_reviewed_ocr_overlays.json",
                video_dir / "outputs" / "ocr" / "02_filtered_ocr_overlays.json",
                video_dir / "outputs" / "ocr" / "01_processed_ocr_items.json",
            ]
        ),
        "chunks": first_existing(
            [
                video_dir / "outputs" / "chunks" / "transcript_chunks_speaker_validated.json",
                video_dir / "outputs" / "chunks" / "transcript_chunks.json",
            ]
        ),
    }


def chunk_values(path: Path | None) -> list[dict[str, Any]]:
    payload = read_json(path)
    values = payload.get("chunks") if isinstance(payload, dict) else payload
    return [value for value in values if isinstance(value, dict)] if isinstance(values, list) else []


def ocr_values(path: Path | None) -> dict[str, Any]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("kinds"), dict):
        return payload["kinds"]
    if isinstance(payload, dict):
        return {"items": payload.get("items", payload)}
    if isinstance(payload, list):
        return {"items": payload}
    return {}


def count_ocr(groups: dict[str, Any]) -> int:
    return sum(len(value) for value in groups.values() if isinstance(value, (dict, list)))


def explorer_metadata(video_id: str) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "explorateur": dg.MetadataValue.url(f"{EXPLORER_URL}?video={video_id}"),
    }


@dg.asset(
    partitions_def=VIDEO_PARTITIONS,
    group_name=GROUP_NAME,
    kinds={"youtube", "video"},
    description="Fichier vidéo source et métadonnées YouTube disponibles localement.",
)
def video_source(context: AssetExecutionContext) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    metadata = read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
    video_file = next(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    return dg.MaterializeResult(
        metadata={
            **explorer_metadata(video_id),
            "titre": str(metadata.get("title") or video_id),
            "durée_secondes": int(metadata.get("duration_seconds") or metadata.get("duration") or 0),
            "fichier": dg.MetadataValue.path(str(video_file)),
            "taille_mo": round(video_file.stat().st_size / 1024 / 1024, 2),
            "youtube": dg.MetadataValue.url(str(metadata.get("url") or f"https://www.youtube.com/watch?v={video_id}")),
        }
    )


@dg.asset(
    partitions_def=VIDEO_PARTITIONS,
    deps=[video_source],
    group_name=GROUP_NAME,
    kinds={"images", "computer-vision"},
    description="Frames extraites et classées par type visuel.",
)
def extracted_images(context: AssetExecutionContext) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    images_dir = video_dir / "outputs" / "images"
    images = [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    categories = Counter(path.parent.name for path in images)
    return dg.MaterializeResult(
        metadata={
            **explorer_metadata(video_id),
            "nombre_images": len(images),
            "répartition": dict(sorted(categories.items())),
            "dossier": dg.MetadataValue.path(str(images_dir)),
        }
    )


@dg.asset(
    partitions_def=VIDEO_PARTITIONS,
    deps=[extracted_images],
    group_name=GROUP_NAME,
    kinds={"ocr", "json"},
    description="Textes OCR consolidés, séparés entre sous-titres, graphiques et autres textes.",
)
def ocr_texts(context: AssetExecutionContext) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    path = output_paths(video_dir)["ocr"]
    groups = ocr_values(path)
    preview_lines: list[str] = []
    for kind, values in groups.items():
        entries = list(values.items()) if isinstance(values, dict) else list(enumerate(values)) if isinstance(values, list) else []
        if entries:
            preview_lines.append(f"### {kind}")
            preview_lines.extend(f"- `{key}` — {str(value)[:180]}" for key, value in entries[:5])
    return dg.MaterializeResult(
        metadata={
            **explorer_metadata(video_id),
            "nombre_textes": count_ocr(groups),
            "catégories": {key: len(value) for key, value in groups.items() if isinstance(value, (dict, list))},
            "aperçu": dg.MetadataValue.md("\n".join(preview_lines) or "Aucun texte OCR."),
            "fichier": dg.MetadataValue.path(str(path)) if path else "absent",
        }
    )


@dg.asset(
    partitions_def=VIDEO_PARTITIONS,
    deps=[ocr_texts],
    group_name=GROUP_NAME,
    kinds={"text", "transcript"},
    description="Transcript textuel unifié utilisé pour le résumé et le chunking.",
)
def plain_transcript(context: AssetExecutionContext) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    paths = output_paths(video_dir)
    transcript = read_text(paths["transcript"], limit=20_000)
    summary = read_text(paths["summary"], limit=8_000)
    return dg.MaterializeResult(
        metadata={
            **explorer_metadata(video_id),
            "caractères": len(transcript),
            "mots": len(transcript.split()),
            "aperçu_transcript": dg.MetadataValue.md(transcript[:2_500] or "Transcript absent."),
            "résumé": dg.MetadataValue.md(summary or "Résumé absent."),
            "fichier": dg.MetadataValue.path(str(paths["transcript"])) if paths["transcript"] else "absent",
        }
    )


@dg.asset(
    partitions_def=VIDEO_PARTITIONS,
    deps=[plain_transcript],
    group_name=GROUP_NAME,
    kinds={"json", "chunks"},
    description="Segments textuels enrichis avec les locuteurs validés.",
)
def transcript_chunks(context: AssetExecutionContext) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    path = output_paths(video_dir)["chunks"]
    chunks = chunk_values(path)
    speakers = sorted(
        {
            str(speaker)
            for chunk in chunks
            for speaker in (chunk.get("meta_data") or {}).get("speakers", [])
            if speaker
        }
    )
    preview = "\n\n".join(
        f"### Chunk {chunk.get('chunk_index', index + 1)}\n{str(chunk.get('content') or chunk.get('text') or '')[:800]}"
        for index, chunk in enumerate(chunks[:3])
    )
    return dg.MaterializeResult(
        metadata={
            **explorer_metadata(video_id),
            "nombre_chunks": len(chunks),
            "locuteurs": speakers,
            "aperçu": dg.MetadataValue.md(preview or "Aucun chunk."),
            "fichier": dg.MetadataValue.path(str(path)) if path else "absent",
        }
    )


@dg.asset(
    partitions_def=VIDEO_PARTITIONS,
    deps=[transcript_chunks],
    group_name=GROUP_NAME,
    kinds={"embeddings", "openai"},
    description="Vecteurs d'embedding associés aux chunks et prêts pour le RAG.",
)
def chunk_embeddings(context: AssetExecutionContext) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    chunks_dir = video_dir / "outputs" / "chunks"
    embeddings = sorted(chunks_dir.glob("*_embedding.json"))
    chunks = chunk_values(output_paths(video_dir)["chunks"])
    return dg.MaterializeResult(
        metadata={
            **explorer_metadata(video_id),
            "nombre_embeddings": len(embeddings),
            "nombre_chunks": len(chunks),
            "couverture_pct": round(len(embeddings) / len(chunks) * 100, 1) if chunks else 0.0,
            "dossier": dg.MetadataValue.path(str(chunks_dir)),
        }
    )


@dg.asset_check(asset=video_source, description="Le fichier vidéo source existe et n'est pas vide.")
def source_video_exists(context: AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    files = [path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    size = sum(path.stat().st_size for path in files)
    return dg.AssetCheckResult(passed=bool(files) and size > 0, metadata={"fichiers": len(files), "octets": size})


@dg.asset_check(asset=extracted_images, description="Au moins une image a été extraite.")
def images_are_not_empty(context: AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    count = sum(1 for path in (video_dir / "outputs" / "images").rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return dg.AssetCheckResult(passed=count > 0, metadata={"nombre_images": count})


@dg.asset_check(asset=ocr_texts, description="La sortie OCR contient du texte exploitable.")
def ocr_is_not_empty(context: AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    count = count_ocr(ocr_values(output_paths(video_dir)["ocr"]))
    return dg.AssetCheckResult(passed=count > 0, metadata={"nombre_textes": count})


@dg.asset_check(asset=plain_transcript, description="Le transcript contient au moins 100 caractères.")
def transcript_is_not_empty(context: AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    text = read_text(output_paths(video_dir)["transcript"], limit=1_000_000)
    return dg.AssetCheckResult(passed=len(text.strip()) >= 100, metadata={"caractères": len(text)})


@dg.asset_check(asset=transcript_chunks, description="Le transcript a produit au moins un chunk.")
def chunks_are_not_empty(context: AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    count = len(chunk_values(output_paths(video_dir)["chunks"]))
    return dg.AssetCheckResult(passed=count > 0, metadata={"nombre_chunks": count})


@dg.asset_check(asset=chunk_embeddings, description="Chaque chunk possède exactement un embedding.")
def embeddings_cover_chunks(context: AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    chunk_count = len(chunk_values(output_paths(video_dir)["chunks"]))
    embedding_count = len(list((video_dir / "outputs" / "chunks").glob("*_embedding.json")))
    return dg.AssetCheckResult(
        passed=chunk_count > 0 and embedding_count == chunk_count,
        metadata={"nombre_chunks": chunk_count, "nombre_embeddings": embedding_count},
    )


@dg.sensor(minimum_interval_seconds=15, default_status=dg.DefaultSensorStatus.RUNNING)
def discover_video_partitions(context: SensorEvaluationContext) -> dg.SensorResult:
    discovered = set(discover_local_videos())
    existing = set(context.instance.get_dynamic_partitions(VIDEO_PARTITIONS.name))
    new_partitions = sorted(discovered - existing)
    if not new_partitions:
        return dg.SensorResult(skip_reason="Aucune nouvelle vidéo locale.")
    return dg.SensorResult(
        dynamic_partitions_requests=[VIDEO_PARTITIONS.build_add_request(new_partitions)],
    )


catalog_job = dg.define_asset_job(
    name="cataloguer_video",
    selection=dg.AssetSelection.groups(GROUP_NAME),
    partitions_def=VIDEO_PARTITIONS,
    description="Catalogue et contrôle les sorties existantes d'une vidéo sans relancer le pipeline.",
)


defs = dg.Definitions(
    assets=[video_source, extracted_images, ocr_texts, plain_transcript, transcript_chunks, chunk_embeddings],
    asset_checks=[
        source_video_exists,
        images_are_not_empty,
        ocr_is_not_empty,
        transcript_is_not_empty,
        chunks_are_not_empty,
        embeddings_cover_chunks,
    ],
    sensors=[discover_video_partitions],
    jobs=[catalog_job],
)
