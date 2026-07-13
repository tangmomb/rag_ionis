import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import dagster as dg

from dagster_pipeline.runtime import (
    DOWNLOAD_ROOT,
    PIPELINE_PYTHON,
    PROJECT_DIR,
    VIDEO_PARTITIONS,
    discover_local_videos,
    execute_command,
)


class ImportSettings(dg.ConfigurableResource):
    """Paramètres visibles dans le Launchpad du job importer_videos."""

    videos: str = "3"
    cookies_from_browser: str = "edge"
    force_download: bool = False
    min_delay_seconds: float = 15.0
    max_delay_seconds: float = 45.0
    reset_before_import: bool = True
    download_dir: str = str(DOWNLOAD_ROOT)
    pipeline_python: str = str(PIPELINE_PYTHON)


def _selection_args(value: str) -> list[str]:
    selection = value.strip()
    if selection.lower() == "all":
        return []
    try:
        count = int(selection)
    except ValueError:
        parsed = urlparse(selection if "://" in selection else f"https://{selection}")
        host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
        is_youtube = host == "youtu.be" or host.endswith("youtube.com")
        if host == "youtu.be":
            video_id = parsed.path.strip("/").split("/", 1)[0]
        elif parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = parts[1] if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"} else ""
        if not is_youtube or len(video_id) != 11:
            raise dg.Failure(description="videos doit être un entier, 'all' ou une URL YouTube valide.")
        return ["--video-url", selection]
    if count <= 0:
        raise dg.Failure(description="Le nombre de vidéos doit être supérieur à zéro.")
    return ["--limit", str(count)]


def _python(settings: ImportSettings) -> Path:
    python = Path(settings.pipeline_python)
    if not python.is_file():
        raise dg.Failure(description=f"Python GPU introuvable : {python}")
    return python


def _download_root(settings: ImportSettings) -> Path:
    root = Path(settings.download_dir)
    return root if root.is_absolute() else PROJECT_DIR / root


@dg.op(description="Prépare l'import. Le nettoyage destructif n'a lieu que si reset_before_import=true.")
def prepare_import(context: dg.OpExecutionContext, import_settings: ImportSettings) -> str:
    root = _download_root(import_settings).resolve()
    project = PROJECT_DIR.resolve()
    if project not in root.parents:
        raise dg.Failure(description=f"Dossier d'import refusé hors du projet : {root}")
    if import_settings.reset_before_import:
        context.log.warning("Réinitialisation demandée : suppression de %s et vidage de la base SQL.", root)
        if root.exists():
            shutil.rmtree(root)
        command = [str(_python(import_settings)), str(PROJECT_DIR / "utils" / "clear_database.py")]
        execute_command(context, command)
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


@dg.op(description="Step 01 — Collecter les métadonnées YouTube dans PostgreSQL.")
def get_youtube_data(context: dg.OpExecutionContext, download_root: str, import_settings: ImportSettings) -> str:
    command = [
        str(_python(import_settings)),
        str(PROJECT_DIR / "scripts" / "init" / "01_get_data.py"),
        "--skip-transcripts",
        "--download-dir",
        download_root,
        *_selection_args(import_settings.videos),
    ]
    execute_command(context, command)
    return download_root


@dg.op(description="Step 02 — Télécharger les vidéos puis créer leurs partitions Dagster.")
def download_videos(context: dg.OpExecutionContext, download_root: str, import_settings: ImportSettings) -> list[str]:
    command = [
        str(_python(import_settings)),
        str(PROJECT_DIR / "scripts" / "init" / "02_download_videos.py"),
        "--download-dir",
        download_root,
        *_selection_args(import_settings.videos),
    ]
    if import_settings.force_download:
        command.append("--force")
    command.extend(
        [
            "--min-delay",
            str(import_settings.min_delay_seconds),
            "--max-delay",
            str(import_settings.max_delay_seconds),
        ]
    )
    if import_settings.cookies_from_browser:
        command.extend(["--cookies-from-browser", import_settings.cookies_from_browser])
    execute_command(context, command)

    discovered = sorted(discover_local_videos(Path(download_root)))
    if not discovered:
        raise dg.Failure(description="Le téléchargement n'a créé aucune partition vidéo.")
    existing = set(context.instance.get_dynamic_partitions(VIDEO_PARTITIONS.name))
    new_partitions = [video_id for video_id in discovered if video_id not in existing]
    if new_partitions:
        context.instance.add_dynamic_partitions(VIDEO_PARTITIONS.name, new_partitions)
    context.add_output_metadata(
        {
            "videos_detectees": len(discovered),
            "partitions_ajoutees": len(new_partitions),
            "ids": discovered,
        }
    )
    return discovered


@dg.job(
    resource_defs={"import_settings": ImportSettings()},
    description="Importer de nouvelles vidéos puis les rendre sélectionnables comme partitions Dagster.",
)
def importer_videos() -> None:
    download_videos(get_youtube_data(prepare_import()))
