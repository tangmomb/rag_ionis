import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import quote

import dagster as dg
from pydantic import Field


PROJECT_DIR = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = PROJECT_DIR / "downloads" / "youtube"
PIPELINE_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
EXPLORER_URL = "http://127.0.0.1:8001/videos"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_PARTITIONS = dg.DynamicPartitionsDefinition(name="youtube_videos")


class PipelineSettings(dg.ConfigurableResource):
    """Options modifiables dans le Launchpad Dagster pour les traitements vidéo."""

    pipeline_python: str = Field(
        default=str(PIPELINE_PYTHON),
        description="Saisie libre : chemin de l'interpréteur Python utilisé pour lancer les scripts. Laisser la valeur par défaut sauf si l'environnement Python est ailleurs.",
    )
    force: bool = Field(
        default=False,
        description="Choix : false ou true. Si true, régénère les sorties même si elles existent déjà et peut écraser les résultats précédents.",
    )
    openai_mode: Literal["normal", "batch"] = Field(
        default="normal",
        description="normal : appels OpenAI immédiats. batch : utilise la Batch API, généralement moins chère mais asynchrone.",
    )
    review_scope: Literal["duo", "all"] = Field(
        default="duo",
        description="duo : revue d'une seule image par vidéo. all : revue de toutes les images candidates.",
    )
    correction_mode: Literal["conservative", "balanced", "aggressive"] = Field(
        default="balanced",
        description="Intensité de correction des timecodes et des noms propres : conservative, balanced ou aggressive.",
    )
    chunk_speaker_validation_model: str = Field(
        default="gpt-5.4-nano",
        description="Saisie libre : nom ou alias du modèle OpenAI utilisé pour valider les locuteurs des segments audio. Par défaut : gpt-5.4-nano.",
    )
    image_review_model: str = Field(
        default="gpt-5.6-luna",
        description="Saisie libre : modèle OpenAI utilisé pour revoir les images candidates et décider si le texte est ajouté au montage. Par défaut : gpt-5.6-luna.",
    )
    dry_run_upload: bool = Field(
        default=False,
        description="Choix : false ou true. Si true, simule la publication S3 et liste les fichiers à envoyer sans rien téléverser.",
    )
    dry_run_sql: bool = Field(
        default=False,
        description="Choix : false ou true. Si true, simule la mise à jour SQL/pgvector sans modifier la base.",
    )


def read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def read_text(path: Path | None, limit: int = 20_000) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def discover_local_videos(download_root: Path | None = None) -> dict[str, Path]:
    """Retourne le dossier local le plus récent pour chaque identifiant YouTube."""

    root = download_root or DOWNLOAD_ROOT
    discovered: dict[str, Path] = {}
    if not root.is_dir():
        return discovered
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        for video_dir in sorted((path for path in run_dir.iterdir() if path.is_dir())):
            video_file = next(
                (
                    path
                    for path in video_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
                ),
                None,
            )
            if video_file is None:
                continue
            metadata = read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
            video_id = str(metadata.get("youtube_video_id") or video_dir.name)
            discovered.setdefault(video_id, video_dir)
    return discovered


def selected_video(context: dg.AssetExecutionContext | dg.AssetCheckExecutionContext) -> tuple[str, Path]:
    video_id = context.partition_key
    video_dir = discover_local_videos().get(video_id)
    if video_dir is None:
        raise dg.Failure(
            description=(
                f"La vidéo {video_id} n'existe plus dans {DOWNLOAD_ROOT}. "
                "Relance l'import ou synchronise les partitions Dagster."
            )
        )
    return video_id, video_dir


def video_branch(video_dir: Path) -> str:
    candidates = (
        video_dir / "metadata" / "pipeline_analysis.json",
        video_dir / "metadata" / "analysed_infos.json",
        video_dir / "analysed_infos.json",
    )
    payload = read_json(first_existing(candidates)) or {}
    has_subtitles = payload.get("has_subtitles")
    if has_subtitles is True:
        return "has_sub"
    if has_subtitles is False:
        return "no_sub"
    raise dg.Failure(
        description=(
            "La route has_sub/no_sub est inconnue. Matérialise d'abord "
            "step_09_detect_ocr_subtitles pour cette vidéo."
        )
    )


def output_paths(video_dir: Path) -> dict[str, Path | None]:
    outputs = video_dir / "outputs"
    transcript_dirs = sorted(path for path in outputs.glob("transcripts*") if path.is_dir())
    return {
        "analysis": first_existing(
            [video_dir / "metadata" / "pipeline_analysis.json", outputs / "metadata" / "pipeline_analysis.json"]
        ),
        "summary": first_existing(path / "video_summary.md" for path in transcript_dirs),
        "transcript": first_existing(path / "plain_transcript.txt" for path in transcript_dirs),
        "ocr": first_existing(
            [
                outputs / "ocr" / "03_reviewed_ocr_overlays.json",
                outputs / "ocr" / "02_filtered_ocr_overlays.json",
                outputs / "ocr" / "01_processed_ocr_items.json",
            ]
        ),
        "chunks": first_existing(
            [
                outputs / "chunks" / "transcript_chunks_speaker_validated.json",
                outputs / "chunks" / "transcript_chunks.json",
            ]
        ),
    }


def chunk_values(path: Path | None) -> list[dict[str, Any]]:
    payload = read_json(path)
    values = payload.get("chunks") if isinstance(payload, dict) else payload
    return [value for value in values if isinstance(value, dict)] if isinstance(values, list) else []


def ocr_count(path: Path | None) -> int:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("kinds"), dict):
        return sum(len(value) for value in payload["kinds"].values() if isinstance(value, (dict, list)))
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return len(payload["items"])
    if isinstance(payload, list):
        return len(payload)
    return 0


def explorer_metadata(video_id: str, video_dir: Path | None = None) -> dict[str, Any]:
    explorer_url = f"{EXPLORER_URL}?video={quote(video_id)}"
    if video_dir is None:
        return {
            "video_id": video_id,
            "ouvrir_dans_explorateur": dg.MetadataValue.url(explorer_url),
        }

    source = read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
    title = str(source.get("title") or video_id)
    youtube_url = str(source.get("url") or f"https://www.youtube.com/watch?v={video_id}")
    images_dir = video_dir / "outputs" / "images"
    images = sorted(
        (
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: (path.name, path.as_posix()),
    ) if images_dir.is_dir() else []
    if images:
        preview_path = images[0].relative_to(video_dir).as_posix()
        run_name = video_dir.parent.name
        media_root = EXPLORER_URL.removesuffix("/videos")
        preview_url = (
            f"{media_root}/api/videos/{quote(run_name)}/{quote(video_id)}/media"
            f"?path={quote(preview_path, safe='/')}"
        )
        preview_label = "Première frame extraite"
    else:
        preview_url = str(source.get("thumbnail_medium_url") or "").strip()
        preview_label = "Miniature YouTube"

    title_markdown = title.replace("[", "\\[").replace("]", "\\]")
    preview_markdown = f"\n\n![{preview_label}]({preview_url})" if preview_url else ""
    return {
        "titre_video": title,
        "apercu_video": dg.MetadataValue.md(
            f"### [{title_markdown}]({youtube_url}){preview_markdown}\n\n`{video_id}`"
        ),
        "image_apercu": dg.MetadataValue.url(preview_url) if preview_url else "indisponible",
        "video_id": video_id,
        "ouvrir_dans_explorateur": dg.MetadataValue.url(explorer_url),
    }


def output_snapshot(video_id: str, video_dir: Path) -> dict[str, Any]:
    paths = output_paths(video_dir)
    images_dir = video_dir / "outputs" / "images"
    chunks_dir = video_dir / "outputs" / "chunks"
    images = sum(
        1
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if images_dir.is_dir() else 0
    chunks = chunk_values(paths["chunks"])
    embeddings = len(list(chunks_dir.glob("*_embedding.json"))) if chunks_dir.is_dir() else 0
    transcript = read_text(paths["transcript"], limit=1_000_000)
    return {
        **explorer_metadata(video_id, video_dir),
        "dossier_video": dg.MetadataValue.path(str(video_dir)),
        "images": images,
        "textes_ocr": ocr_count(paths["ocr"]),
        "caracteres_transcript": len(transcript),
        "chunks": len(chunks),
        "embeddings": embeddings,
    }


def build_video_command(
    relative_script: str,
    video_dir: Path,
    settings: PipelineSettings,
    extra_args: Iterable[str] = (),
) -> list[str]:
    python = Path(settings.pipeline_python)
    if not python.is_file():
        raise dg.Failure(description=f"Python GPU introuvable : {python}")
    command = [str(python), str(PROJECT_DIR / "scripts" / "init" / relative_script), "--video-dir", str(video_dir)]
    command.extend(map(str, extra_args))
    if settings.force:
        command.append("--force")
    return command


def execute_command(context: dg.AssetExecutionContext | dg.OpExecutionContext, command: list[str], env: dict[str, str] | None = None) -> float:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    if env:
        environment.update(env)
    context.log.info("Commande : %s", subprocess.list2cmdline(command))
    started_at = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        cleaned = line.rstrip()
        if cleaned:
            context.log.info(cleaned)
    return_code = process.wait()
    elapsed = time.perf_counter() - started_at
    if return_code:
        raise dg.Failure(
            description=f"Commande terminée avec le code {return_code}",
            metadata={"commande": subprocess.list2cmdline(command), "code_retour": return_code},
        )
    context.log.info("Étape terminée en %.1f s", elapsed)
    return elapsed
