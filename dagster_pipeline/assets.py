import subprocess
from dataclasses import dataclass
from pathlib import Path
import dagster as dg

from dagster_pipeline.runtime import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    VIDEO_PARTITIONS,
    PipelineSettings,
    build_video_command,
    chunk_values,
    execute_command,
    explorer_metadata,
    ocr_count,
    output_paths,
    output_snapshot,
    read_json,
    read_text,
    selected_video,
    video_branch,
)


PROCESSING_GROUP = "01_traitement_video"
PUBLICATION_GROUP = "02_publication"


@dataclass(frozen=True)
class StepSpec:
    name: str
    description: str
    script: str
    dependency: str
    branch: str | None = None
    extra_args: str | None = None
    kinds: tuple[str, ...] = ("python",)
    retries: int = 0


def _extra_args(spec: StepSpec, settings: PipelineSettings) -> list[str]:
    if spec.extra_args == "review_scope" and settings.review_scope == "duo":
        return ["--limit-images", "1"]
    if spec.extra_args == "correction_mode":
        return ["--mode", settings.correction_mode]
    if spec.extra_args == "speaker_model":
        return ["--model", settings.chunk_speaker_validation_model]
    return []


def make_step_asset(spec: StepSpec) -> dg.AssetsDefinition:
    @dg.asset(
        name=spec.name,
        deps=[dg.AssetKey(spec.dependency)],
        partitions_def=VIDEO_PARTITIONS,
        group_name=PROCESSING_GROUP,
        kinds=set(spec.kinds),
        description=spec.description,
        retry_policy=dg.RetryPolicy(max_retries=spec.retries),
    )
    def step_asset(context: dg.AssetExecutionContext, pipeline_settings: PipelineSettings) -> dg.MaterializeResult:
        video_id, video_dir = selected_video(context)
        branch = video_branch(video_dir) if (spec.branch or "{branch}" in spec.script) else None
        if spec.branch and branch != spec.branch:
            context.log.info("Étape non applicable : route vidéo=%s, branche asset=%s", branch, spec.branch)
            return dg.MaterializeResult(
                metadata={
                    **explorer_metadata(video_id),
                    "statut": "non applicable",
                    "route_video": branch or "inconnue",
                    "branche_asset": spec.branch,
                }
            )

        relative_script = spec.script.format(branch=branch)
        command = build_video_command(relative_script, video_dir, pipeline_settings, _extra_args(spec, pipeline_settings))
        elapsed = execute_command(
            context,
            command,
            env={"PIPELINE_OPENAI_MODE": pipeline_settings.openai_mode},
        )
        return dg.MaterializeResult(
            metadata={
                **output_snapshot(video_id, video_dir),
                "statut": "exécuté",
                "route_video": branch or "commune",
                "duree_secondes": round(elapsed, 2),
                "commande": dg.MetadataValue.text(subprocess.list2cmdline(command)),
            }
        )

    return step_asset


@dg.asset(
    partitions_def=VIDEO_PARTITIONS,
    group_name=PROCESSING_GROUP,
    kinds={"youtube", "video"},
    description="Vidéo source téléchargée. Point d'entrée de la partition Dagster.",
)
def video_source(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    metadata = read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
    video_file = next(
        path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return dg.MaterializeResult(
        metadata={
            **explorer_metadata(video_id),
            "titre": str(metadata.get("title") or video_id),
            "fichier": dg.MetadataValue.path(str(video_file)),
            "taille_mo": round(video_file.stat().st_size / 1024 / 1024, 2),
            "youtube": dg.MetadataValue.url(
                str(metadata.get("url") or f"https://www.youtube.com/watch?v={video_id}")
            ),
        }
    )


COMMON_SPECS = [
    StepSpec("step_03_extract_images", "03 — Extraire les frames de la vidéo.", "03_extract_images.py", "video_source", kinds=("images", "opencv"), retries=1),
    StepSpec("step_04_classify_images", "04 — Classer les frames par type visuel.", "04_classify_images.py", "step_03_extract_images", kinds=("images", "ml"), retries=1),
    StepSpec("step_05_detect_interviews", "05 — Détecter les séquences d'interview.", "05_detect_interviews.py", "step_04_classify_images", kinds=("computer_vision",)),
    StepSpec("step_06_infer_video_type", "06 — Déduire le type global de la vidéo.", "06_infer_video_type.py", "step_05_detect_interviews", kinds=("metadata",)),
    StepSpec("step_07_extract_raw_ocr", "07 — Extraire l'OCR brut.", "07_extract_raw_ocr.py", "step_06_infer_video_type", kinds=("ocr",), retries=1),
    StepSpec("step_08_extract_ocr_boxes", "08 — Construire les boîtes OCR.", "08_extract_ocr_boxes.py", "step_07_extract_raw_ocr", kinds=("ocr", "geometry"), retries=1),
    StepSpec("step_09_detect_ocr_subtitles", "09 — Détecter les sous-titres et choisir la route has_sub/no_sub.", "09_detect_ocr_subtitles.py", "step_08_extract_ocr_boxes", kinds=("ocr", "routing")),
    StepSpec("step_10_build_processed_ocr", "10 — Consolider l'OCR selon la route vidéo.", "{branch}/10_OCR_build_processed_ocr.py", "step_09_detect_ocr_subtitles", kinds=("ocr",)),
    StepSpec("step_11_filter_processed_ocr", "11 — Filtrer l'OCR consolidé.", "{branch}/11_OCR_filter_processed_ocr.py", "step_10_build_processed_ocr", kinds=("ocr",)),
    StepSpec("step_12_extract_review_candidates", "12 — Extraire les textes à revoir.", "{branch}/12_OCR_extract_other_text_review_candidates.py", "step_11_filter_processed_ocr", kinds=("ocr", "review")),
    StepSpec("step_13_review_candidates", "13 — Revoir les candidats avec OpenAI.", "{branch}/13_OCR_review_other_text_candidates.py", "step_12_extract_review_candidates", extra_args="review_scope", kinds=("openai", "review"), retries=2),
    StepSpec("step_14_apply_review", "14 — Appliquer la revue aux textes OCR.", "{branch}/14_OCR_apply_other_text_review.py", "step_13_review_candidates", kinds=("ocr", "review")),
]


HAS_SUB_SPECS = [
    StepSpec("step_15_has_sub_extract_subtitles", "15 — Extraire les sous-titres OCR.", "has_sub/15_OCR_extract_ocr_subtitles.py", "step_14_apply_review", branch="has_sub", kinds=("ocr", "subtitles")),
    StepSpec("step_16_has_sub_fix_spacing", "16 — Corriger l'espacement des sous-titres.", "has_sub/16_OCR_correct_ocr_subtitle_spacing.py", "step_15_has_sub_extract_subtitles", branch="has_sub", kinds=("text",)),
    StepSpec("step_17_has_sub_normalize", "17 — Normaliser le vocabulaire IONIS-STM.", "has_sub/17_OCR_normalize_ionis_stm.py", "step_16_has_sub_fix_spacing", branch="has_sub", kinds=("text",)),
    StepSpec("step_18_has_sub_plain_transcript", "18 — Créer le transcript texte.", "has_sub/18_OCR_create_plain_transcript.py", "step_17_has_sub_normalize", branch="has_sub", kinds=("transcript",)),
    StepSpec("step_19_has_sub_enrich_timecodes", "19 — Enrichir les timecodes.", "has_sub/19_OCR_enrich_transcripts.py", "step_18_has_sub_plain_transcript", branch="has_sub", kinds=("transcript",)),
    StepSpec("step_20_has_sub_summary", "20 — Générer le résumé vidéo.", "has_sub/20_OCR_generate_video_summary.py", "step_19_has_sub_enrich_timecodes", branch="has_sub", kinds=("openai", "summary"), retries=2),
    StepSpec("step_21_has_sub_chunks", "21 — Créer les chunks du transcript.", "has_sub/21_CHUNK_create_transcript_chunks.py", "step_20_has_sub_summary", branch="has_sub", kinds=("chunks",)),
    StepSpec("step_22_has_sub_validate_speakers", "22 — Valider les locuteurs des chunks.", "has_sub/22_CHUNK_validate_chunk_speakers.py", "step_21_has_sub_chunks", branch="has_sub", extra_args="speaker_model", kinds=("openai", "chunks"), retries=2),
    StepSpec("step_23_has_sub_embeddings", "23 — Créer les embeddings des chunks.", "has_sub/23_CHUNK_create_chunk_embeddings.py", "step_22_has_sub_validate_speakers", branch="has_sub", kinds=("embeddings", "openai"), retries=2),
]


NO_SUB_SPECS = [
    StepSpec("step_16_no_sub_whisper", "16 — Transcrire l'audio avec WhisperX.", "no_sub/16_WHISPER_transcribe_with_whisper.py", "step_14_apply_review", branch="no_sub", kinds=("whisper", "transcript"), retries=1),
    StepSpec("step_17_no_sub_correct_timecodes", "17 — Corriger les timecodes et noms propres.", "no_sub/17_WHISPER_correct_transcript_timecodes.py", "step_16_no_sub_whisper", branch="no_sub", extra_args="correction_mode", kinds=("transcript", "openai"), retries=2),
    StepSpec("step_18_no_sub_enrich_timecodes", "18 — Enrichir les timecodes.", "no_sub/18_WHISPER_enrich_transcripts.py", "step_17_no_sub_correct_timecodes", branch="no_sub", kinds=("transcript",)),
    StepSpec("step_19_no_sub_summary", "19 — Générer le résumé vidéo.", "no_sub/19_WHISPER_generate_video_summary.py", "step_18_no_sub_enrich_timecodes", branch="no_sub", kinds=("openai", "summary"), retries=2),
    StepSpec("step_20_no_sub_plain_transcript", "20 — Créer le transcript texte.", "no_sub/20_WHISPER_create_plain_transcript.py", "step_19_no_sub_summary", branch="no_sub", kinds=("transcript",)),
    StepSpec("step_21_no_sub_chunks", "21 — Créer les chunks du transcript.", "no_sub/21_CHUNK_create_transcript_chunks.py", "step_20_no_sub_plain_transcript", branch="no_sub", kinds=("chunks",)),
    StepSpec("step_22_no_sub_validate_speakers", "22 — Valider les locuteurs des chunks.", "no_sub/22_CHUNK_validate_chunk_speakers.py", "step_21_no_sub_chunks", branch="no_sub", extra_args="speaker_model", kinds=("openai", "chunks"), retries=2),
    StepSpec("step_24_no_sub_embeddings", "24 — Créer les embeddings des chunks.", "no_sub/24_CHUNK_create_chunk_embeddings.py", "step_22_no_sub_validate_speakers", branch="no_sub", kinds=("embeddings", "openai"), retries=2),
]


STEP_ASSETS = [make_step_asset(spec) for spec in COMMON_SPECS + HAS_SUB_SPECS + NO_SUB_SPECS]


@dg.asset(
    deps=[dg.AssetKey("step_23_has_sub_embeddings"), dg.AssetKey("step_24_no_sub_embeddings")],
    partitions_def=VIDEO_PARTITIONS,
    group_name=PROCESSING_GROUP,
    kinds={"dataset", "rag"},
    description="Vue consolidée des résultats utilisables par le RAG et l'explorateur vidéo.",
)
def pipeline_outputs(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    paths = output_paths(video_dir)
    transcript = read_text(paths["transcript"], limit=20_000)
    summary = read_text(paths["summary"], limit=8_000)
    metadata = output_snapshot(video_id, video_dir)
    metadata.update(
        {
            "route_video": video_branch(video_dir),
            "apercu_transcript": dg.MetadataValue.md(transcript[:2_500] or "Transcript absent."),
            "resume": dg.MetadataValue.md(summary or "Résumé absent."),
        }
    )
    return dg.MaterializeResult(metadata=metadata)


def _publication_command(script: str, video_dir: Path, settings: PipelineSettings, extra: list[str]) -> list[str]:
    python = Path(settings.pipeline_python)
    if not python.is_file():
        raise dg.Failure(description=f"Python GPU introuvable : {python}")
    command = [str(python), str(Path(__file__).resolve().parents[1] / "scripts" / "init" / script), "--video-dir", str(video_dir)]
    command.extend(extra)
    return command


@dg.asset(
    deps=[pipeline_outputs],
    partitions_def=VIDEO_PARTITIONS,
    group_name=PUBLICATION_GROUP,
    kinds={"s3"},
    description="24/25 — Publier les sorties de cette vidéo dans S3.",
    retry_policy=dg.RetryPolicy(max_retries=2),
)
def upload_outputs_to_s3(context: dg.AssetExecutionContext, pipeline_settings: PipelineSettings) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    prefix = f"youtube/{video_dir.parent.name}/{video_dir.name}"
    extra = ["--prefix", prefix]
    if pipeline_settings.force:
        extra.append("--force")
    if pipeline_settings.dry_run_upload:
        extra.append("--dry-run")
    elapsed = execute_command(context, _publication_command("final_01_upload_outputs_to_s3.py", video_dir, pipeline_settings, extra))
    return dg.MaterializeResult(
        metadata={**explorer_metadata(video_id), "prefixe_s3": prefix, "duree_secondes": round(elapsed, 2)}
    )


@dg.asset(
    deps=[upload_outputs_to_s3],
    partitions_def=VIDEO_PARTITIONS,
    group_name=PUBLICATION_GROUP,
    kinds={"postgresql", "pgvector"},
    description="25/26 — Mettre à jour les assets SQL/pgvector de cette vidéo.",
    retry_policy=dg.RetryPolicy(max_retries=2),
)
def update_sql_assets(context: dg.AssetExecutionContext, pipeline_settings: PipelineSettings) -> dg.MaterializeResult:
    video_id, video_dir = selected_video(context)
    prefix = f"youtube/{video_dir.parent.name}/{video_dir.name}"
    extra = ["--prefix", prefix]
    if pipeline_settings.dry_run_sql:
        extra.append("--dry-run")
    elapsed = execute_command(context, _publication_command("final_02_update_sql_assets.py", video_dir, pipeline_settings, extra))
    return dg.MaterializeResult(
        metadata={**explorer_metadata(video_id), "prefixe_s3": prefix, "duree_secondes": round(elapsed, 2)}
    )


@dg.asset_check(asset=video_source, description="Le fichier vidéo source existe et n'est pas vide.")
def source_video_exists(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    files = [path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    size = sum(path.stat().st_size for path in files)
    return dg.AssetCheckResult(passed=bool(files) and size > 0, metadata={"fichiers": len(files), "octets": size})


@dg.asset_check(asset=dg.AssetKey("step_03_extract_images"), description="Au moins une image a été extraite.")
def images_are_not_empty(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    root = video_dir / "outputs" / "images"
    count = sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS) if root.is_dir() else 0
    return dg.AssetCheckResult(passed=count > 0, metadata={"nombre_images": count})


@dg.asset_check(asset=dg.AssetKey("step_09_detect_ocr_subtitles"), description="La sortie OCR contient du texte exploitable.")
def ocr_is_not_empty(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    count = ocr_count(output_paths(video_dir)["ocr"])
    return dg.AssetCheckResult(passed=count > 0, metadata={"nombre_textes": count})


@dg.asset_check(asset=pipeline_outputs, description="Le transcript contient au moins 100 caractères.")
def transcript_is_not_empty(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    text = read_text(output_paths(video_dir)["transcript"], limit=1_000_000)
    return dg.AssetCheckResult(passed=len(text.strip()) >= 100, metadata={"caracteres": len(text)})


@dg.asset_check(asset=pipeline_outputs, description="Le transcript a produit au moins un chunk.")
def chunks_are_not_empty(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    count = len(chunk_values(output_paths(video_dir)["chunks"]))
    return dg.AssetCheckResult(passed=count > 0, metadata={"nombre_chunks": count})


@dg.asset_check(asset=pipeline_outputs, description="Chaque chunk possède exactement un embedding.")
def embeddings_cover_chunks(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    _video_id, video_dir = selected_video(context)
    chunk_count = len(chunk_values(output_paths(video_dir)["chunks"]))
    embedding_count = len(list((video_dir / "outputs" / "chunks").glob("*_embedding.json")))
    return dg.AssetCheckResult(
        passed=chunk_count > 0 and embedding_count == chunk_count,
        metadata={"nombre_chunks": chunk_count, "nombre_embeddings": embedding_count},
    )


ALL_ASSETS = [video_source, *STEP_ASSETS, pipeline_outputs, upload_outputs_to_s3, update_sql_assets]
ALL_CHECKS = [
    source_video_exists,
    images_are_not_empty,
    ocr_is_not_empty,
    transcript_is_not_empty,
    chunks_are_not_empty,
    embeddings_cover_chunks,
]
