from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from .context import PipelineContext
from .contracts import TaskResult, VideoType


ArtifactFingerprint = dict[str, tuple[int, int]]


def _paths_from(value: object) -> list[Path]:
    paths: list[Path] = []

    def collect(item: object) -> None:
        if isinstance(item, Path):
            paths.append(item)
        elif isinstance(item, Mapping):
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                collect(nested)

    collect(value)
    return paths


def _existing_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    unique: dict[str, Path] = {}
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            unique.setdefault(candidate.resolve().as_posix(), candidate)
    return tuple(unique.values())


def _snapshot(paths: Iterable[Path]) -> ArtifactFingerprint:
    snapshot: ArtifactFingerprint = {}
    for path in _existing_paths(paths):
        stat = path.stat()
        snapshot[path.resolve().as_posix()] = (
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
    return snapshot


def _artifact_result(
    context: PipelineContext,
    value: object,
    *,
    artifacts: Iterable[Path],
    state_paths: Iterable[Path],
    before: ArtifactFingerprint,
    success_reason: str,
    cached_reason: str,
    missing_reason: str,
    missing_is_skip: bool = False,
) -> TaskResult:
    existing_artifacts = _existing_paths(artifacts)
    if not existing_artifacts:
        return (
            TaskResult.skipped(missing_reason)
            if missing_is_skip
            else TaskResult.blocked(missing_reason)
        )

    after = _snapshot(state_paths)
    if not after:
        return (
            TaskResult.skipped(missing_reason)
            if missing_is_skip
            else TaskResult.blocked(missing_reason)
        )
    if not context.force_rebuild and before and before == after:
        return TaskResult.cached(
            artifacts=existing_artifacts,
            reason=cached_reason,
            value=value,
        )
    return TaskResult.succeeded(
        value,
        artifacts=existing_artifacts,
        reason=success_reason,
    )


def extract_frames(context: PipelineContext) -> TaskResult:
    from pipeline.steps.inspection.extract_frames import extract_images
    from pipeline.support.paths import images_dir

    target = images_dir(context.video_path)
    before_images = sorted(target.glob("*.jpg")) if target.exists() else []
    before = _snapshot(before_images)
    result = extract_images(
        context.video_path,
        context.options.frame_interval_seconds,
        force=context.force_rebuild,
    )
    images = sorted(target.glob("*.jpg")) if target.exists() else []
    return _artifact_result(
        context,
        result,
        artifacts=(target,),
        state_paths=images,
        before=before,
        success_reason=f"{len(images)} frame(s) extraite(s).",
        cached_reason=f"{len(images)} frame(s) deja extraite(s).",
        missing_reason="L'extraction n'a produit aucune frame.",
    )


def classify_frames(context: PipelineContext) -> TaskResult:
    from pipeline.steps.inspection.classify_frames import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EMBEDDING_CACHE_DIRNAME,
        DEFAULT_MODEL_PATH,
        classify_video,
        existing_manifest_path,
    )
    from pipeline.support.paths import existing_images_dir

    image_directory = existing_images_dir(context.video_path)
    target_before = existing_manifest_path(image_directory)
    before = _snapshot((target_before,))
    result = classify_video(
        context.video_path,
        model=DEFAULT_MODEL_PATH,
        batch_size=DEFAULT_BATCH_SIZE,
        device=None,
        cache_dir=image_directory / DEFAULT_EMBEDDING_CACHE_DIRNAME,
        force=context.force_rebuild,
    )
    target_after = existing_manifest_path(image_directory)
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target_after),
        state_paths=(target_after,),
        before=before,
        success_reason="Frames classifiees.",
        cached_reason="Classification des frames deja a jour.",
        missing_reason="Aucune classification produite; les frames sont absentes.",
    )


def detect_interview(context: PipelineContext) -> TaskResult:
    from pipeline.steps.inspection.detect_interviews import (
        DEFAULT_MAX_INTERVIEW_SEQUENCES,
        LEGACY_MANIFEST_NAME,
        MANIFEST_NAME,
        SOURCE_DIR_NAMES,
        detect_video,
    )
    from pipeline.support.paths import existing_interview_dir

    output_directory = existing_interview_dir(context.video_path)
    candidates = (
        output_directory / MANIFEST_NAME,
        output_directory / LEGACY_MANIFEST_NAME,
    )
    before = _snapshot(candidates)
    result = detect_video(
        context.video_path,
        source_dirs=SOURCE_DIR_NAMES,
        phash_similar_max=6,
        phash_ambiguous_max=14,
        ssim_min=0.92,
        min_run_frames=6,
        max_gap_pairs=1,
        max_interview_sequences=DEFAULT_MAX_INTERVIEW_SEQUENCES,
        force=context.force_rebuild,
    )
    artifacts = (*_paths_from(result), *candidates)
    return _artifact_result(
        context,
        result,
        artifacts=artifacts,
        state_paths=candidates,
        before=before,
        success_reason="Detection d'interview calculee.",
        cached_reason="Detection d'interview deja a jour.",
        missing_reason="Detection impossible; aucune frame candidate.",
    )


def infer_video_type(context: PipelineContext) -> TaskResult:
    from pipeline.steps.inspection.infer_video_type import infer_for_video

    previous = context.video_type
    video_type = infer_for_video(
        context.video_path,
        force=context.force_rebuild,
    )
    normalized = VideoType.from_value(video_type)
    if normalized is None:
        return TaskResult.blocked(
            "Type de video non infere; classification des frames absente."
        )
    context.routing_facts = replace(
        context.routing_facts,
        video_type=normalized,
    )
    context.artifacts.by_task.pop("video.infer_type", None)
    reason = f"Type de video infere: {normalized.value}."
    if previous == normalized.value and not context.force_rebuild:
        return TaskResult.cached(reason=reason, value=normalized.value)
    return TaskResult.succeeded(normalized.value, reason=reason)


def extract_raw_ocr(context: PipelineContext) -> TaskResult:
    from pipeline.steps.inspection.extract_raw_ocr import (
        IMAGE_GROUPS,
        existing_raw_ocr_path,
        extract_for_video_isolated,
        output_ocr_raw_dir,
    )

    output_directory = output_ocr_raw_dir(context.video_path)
    expected_before = tuple(
        existing_raw_ocr_path(output_directory, group)
        for group in IMAGE_GROUPS
    )
    before = _snapshot(expected_before)
    result = extract_for_video_isolated(
        context.video_path,
        force=context.force_rebuild,
    )
    expected_after = tuple(
        existing_raw_ocr_path(output_directory, group)
        for group in IMAGE_GROUPS
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), *expected_after),
        state_paths=expected_after,
        before=before,
        success_reason="OCR brut extrait.",
        cached_reason="OCR brut deja a jour.",
        missing_reason="Aucun fichier OCR brut n'a ete produit.",
    )


def extract_ocr_boxes(context: PipelineContext) -> TaskResult:
    from pipeline.steps.inspection.extract_ocr_boxes import (
        extract_for_video,
        location_path,
    )
    from pipeline.support.paths import existing_ocr_dir

    target = location_path(existing_ocr_dir(context.video_path))
    before = _snapshot((target,))
    result = extract_for_video(
        context.video_path,
        force=context.force_rebuild,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Positions OCR extraites.",
        cached_reason="Positions OCR deja a jour.",
        missing_reason="Positions OCR non produites; OCR brut absent.",
    )


def detect_subtitles(context: PipelineContext) -> TaskResult:
    from pipeline.steps.inspection.detect_subtitles import detect_for_video

    previous = context.has_subtitles
    details = detect_for_video(
        context.video_path,
        force=context.force_rebuild,
    )
    has_subtitles = (
        details.get("has_subtitles")
        if isinstance(details, Mapping)
        else None
    )
    if not isinstance(has_subtitles, bool):
        return TaskResult.blocked(
            "Detection des sous-titres impossible; positions OCR absentes."
        )
    context.routing_facts = replace(
        context.routing_facts,
        has_subtitles=has_subtitles,
        has_subtitles_details=dict(details),
    )
    context.artifacts.by_task.pop("video.detect_subtitles", None)
    reason = f"Detection des sous-titres calculee: {has_subtitles}."
    if previous is has_subtitles and not context.force_rebuild:
        return TaskResult.cached(reason=reason, value=has_subtitles)
    return TaskResult.succeeded(has_subtitles, reason=reason)


def build_processed_ocr(context: PipelineContext) -> TaskResult:
    from pipeline.steps.ocr.build_processed_ocr import process_video, processed_path
    from pipeline.support.paths import existing_ocr_dir

    target = processed_path(existing_ocr_dir(context.video_path))
    before = _snapshot((target,))
    result = process_video(
        context.video_path,
        force=context.force_rebuild,
        strip_subtitles=context.transcript_strategy == "whisper",
    )
    return _artifact_result(
        context,
        result,
        artifacts=(target,),
        state_paths=(target,),
        before=before,
        success_reason="OCR traite construit.",
        cached_reason="OCR traite deja a jour.",
        missing_reason="OCR traite non produit; sorties OCR brutes absentes.",
    )


def filter_ocr_overlays(context: PipelineContext) -> TaskResult:
    from pipeline.steps.ocr.filter_processed_ocr import filter_processed_ocr
    from pipeline.support.ocr_filtering import filtered_ocr_path

    target = filtered_ocr_path(context.video_path)
    before = _snapshot((target,))
    result = filter_processed_ocr(
        context.video_path,
        force=context.force_rebuild,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Overlays OCR filtres.",
        cached_reason="Overlays OCR deja filtres.",
        missing_reason="Filtrage OCR impossible; OCR traite absent.",
    )


def extract_review_candidates(context: PipelineContext) -> TaskResult:
    from pipeline.steps.ocr.extract_other_text_candidates import (
        extract_for_video,
        manifest_path,
    )

    target = manifest_path(context.video_path)
    before = _snapshot((target,))
    result = extract_for_video(
        context.video_path,
        force=context.force_rebuild,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Candidats de revue OCR extraits.",
        cached_reason="Candidats de revue OCR deja a jour.",
        missing_reason="Candidats non produits; OCR filtre absent.",
    )


def review_other_text(context: PipelineContext) -> TaskResult:
    from pipeline.steps.ocr.review_other_text_candidates import (
        review_video,
        source_manifest_path,
        summary_path,
    )

    source = source_manifest_path(context.video_path)
    if not source.exists():
        return TaskResult.blocked(
            "Revue OCR impossible; manifeste des candidats absent."
        )
    target = summary_path(context.video_path)
    before = _snapshot((target,))
    mode = "batch" if context.options.openai_mode == "batch" else "live"
    result = review_video(
        context.video_path,
        context.options.image_review_model,
        mode=mode,
        force=context.force_rebuild,
        limit_images=1 if context.options.review_scope == "duo" else None,
        wait=True,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Textes OCR secondaires revus.",
        cached_reason="Revue des textes OCR deja a jour.",
        missing_reason="La revue OCR n'a produit aucun resume.",
    )


def apply_ocr_review(context: PipelineContext) -> TaskResult:
    from pipeline.steps.ocr.apply_other_text_review import apply_review, output_path

    target = output_path(context.video_path)
    before = _snapshot((target,))
    result = apply_review(
        context.video_path,
        force=context.force_rebuild,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Revue OCR appliquee.",
        cached_reason="Revue OCR deja appliquee.",
        missing_reason="Application impossible; OCR filtre ou revue absent.",
    )


def extract_ocr_transcript(context: PipelineContext) -> TaskResult:
    from pipeline.steps.transcripts.extract_ocr_subtitles import (
        extract_for_video,
        subtitle_timecodes_path,
    )

    target = subtitle_timecodes_path(
        context.video_path,
        transcripts_dir_name=context.transcripts_dir_name,
    )
    before = _snapshot((target,))
    result = extract_for_video(
        context.video_path,
        force=context.force_rebuild,
        transcripts_dir_name=context.transcripts_dir_name,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Transcript OCR extrait.",
        cached_reason="Transcript OCR deja a jour.",
        missing_reason="Transcript OCR non produit; sous-titres OCR absents.",
    )


def correct_ocr_spacing(context: PipelineContext) -> TaskResult:
    from pipeline.steps.transcripts.correct_ocr_subtitle_spacing import (
        openai_client,
        process_video_batch,
        process_video_live,
        subtitle_target_path,
    )

    target = subtitle_target_path(
        context.video_path,
        transcripts_dir_name=context.transcripts_dir_name,
    )
    before = _snapshot((target,))
    if context.options.openai_mode == "batch":
        result = process_video_batch(
            context.options.speaker_validation_model,
            context.video_path,
            force=context.force_rebuild,
            wait=True,
            transcripts_dir_name=context.transcripts_dir_name,
        )
    else:
        result = process_video_live(
            openai_client(),
            context.options.speaker_validation_model,
            context.video_path,
            force=context.force_rebuild,
            transcripts_dir_name=context.transcripts_dir_name,
        )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Espacement du transcript OCR corrige.",
        cached_reason="Espacement du transcript OCR deja corrige.",
        missing_reason="Correction impossible; transcript OCR timecode absent.",
    )


def normalize_brand(context: PipelineContext) -> TaskResult:
    from pipeline.steps.transcripts.normalize_ionis_stm import (
        process_video,
        transcript_path,
    )

    target = transcript_path(context.video_path)
    before = _snapshot((target,))
    result = process_video(
        context.video_path,
        force=context.force_rebuild,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(target,),
        state_paths=(target,),
        before=before,
        success_reason="Marque Ionis-STM normalisee.",
        cached_reason="Normalisation Ionis-STM deja satisfaite.",
        missing_reason="Normalisation impossible; transcript OCR corrige absent.",
    )


def transcribe_whisper(context: PipelineContext) -> TaskResult:
    from pipeline.steps.transcripts.transcribe_with_whisper import (
        DEFAULT_MAX_SPEAKERS,
        DEFAULT_MIN_SPEAKERS,
        load_diarization_pipeline,
        load_whisperx_model,
        transcript_path,
        transcribe_video,
    )
    from pipeline.support.paths import transcripts_dir

    transcript_directory = transcripts_dir(
        context.video_path,
        name=context.transcripts_dir_name,
    )
    target = transcript_path(transcript_directory, context.video_path)
    before = _snapshot((target,))
    if target.exists() and not context.force_rebuild:
        return TaskResult.cached(
            artifacts=(target,),
            reason="Transcript Whisper deja a jour.",
            value=target,
        )

    whisperx, model, device = load_whisperx_model()
    diarization_pipeline, diarization_device = load_diarization_pipeline(device)
    audio_directory = transcript_directory / "audio"
    transcript_directory.mkdir(parents=True, exist_ok=True)
    audio_directory.mkdir(parents=True, exist_ok=True)
    try:
        result = transcribe_video(
            whisperx,
            model,
            context.video_path,
            transcript_directory,
            audio_directory,
            device,
            diarization_pipeline=diarization_pipeline,
            diarization_device=diarization_device,
            min_speakers=DEFAULT_MIN_SPEAKERS,
            max_speakers=DEFAULT_MAX_SPEAKERS,
            force=context.force_rebuild,
        )
    finally:
        shutil.rmtree(audio_directory, ignore_errors=True)
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Transcript Whisper produit.",
        cached_reason="Transcript Whisper deja a jour.",
        missing_reason="Whisper n'a produit aucun transcript.",
    )


def propose_speakers(context: PipelineContext) -> TaskResult:
    from pipeline.steps.speakers.propose_speakers import (
        candidates_path,
        propose_for_video,
    )

    target = candidates_path(context.video_path)
    before = _snapshot((target,))
    result = propose_for_video(
        context.video_path,
        force=context.force_rebuild,
        transcripts_dir_name=context.transcripts_dir_name,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Speakers candidats proposes.",
        cached_reason="Speakers candidats deja a jour.",
        missing_reason="Proposition impossible; transcript source absent ou vide.",
    )


def validate_speakers(context: PipelineContext) -> TaskResult:
    from pipeline.steps.speakers.validate_speakers import (
        normalize_model_name,
        openai_client,
        validate_file_batch,
        validate_file_live,
        validated_path,
    )

    target = validated_path(context.video_path)
    before = _snapshot((target,))
    model = normalize_model_name(context.options.speaker_validation_model)
    if context.options.openai_mode == "batch":
        result = validate_file_batch(
            model,
            context.video_path,
            force=context.force_rebuild,
            wait=True,
        )
    else:
        result = validate_file_live(
            openai_client(),
            model,
            context.video_path,
            force=context.force_rebuild,
        )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Speakers valides.",
        cached_reason="Validation des speakers deja a jour.",
        missing_reason="Validation impossible; candidats speakers absents.",
    )


def assign_ocr_speakers(context: PipelineContext) -> TaskResult:
    from pipeline.steps.speakers.assign_ocr_speakers import (
        assign_ocr_speakers as assign,
        diarization_path,
    )

    target = diarization_path(context.video_path)
    before = _snapshot((target,))
    result = assign(
        context.video_path,
        force=context.force_rebuild,
        transcripts_dir_name=context.transcripts_dir_name,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Speakers attribues au transcript OCR.",
        cached_reason="Attribution des speakers OCR deja a jour.",
        missing_reason="Attribution impossible; transcript ou speakers valides absents.",
    )


def correct_whisper_transcript(context: PipelineContext) -> TaskResult:
    from pipeline.steps.transcripts.correct_whisper_transcript import (
        correct_file,
        corrected_path,
        corrected_words_path,
        timecodes_source_path,
    )

    try:
        source = timecodes_source_path(
            context.video_path,
            transcripts_dir_name=context.transcripts_dir_name,
        )
    except FileNotFoundError:
        return TaskResult.blocked(
            "Correction Whisper impossible; transcript timecode absent."
        )
    target = corrected_path(source)
    words_target = corrected_words_path(source)
    state_paths = (target, words_target)
    before = _snapshot(state_paths)
    result = correct_file(
        context.video_path,
        force=context.force_rebuild,
        mode=context.options.correction_mode,
        transcripts_dir_name=context.transcripts_dir_name,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target, words_target),
        state_paths=state_paths,
        before=before,
        success_reason="Transcript Whisper corrige.",
        cached_reason="Transcript Whisper corrige deja a jour.",
        missing_reason="Correction Whisper impossible; OCR d'analyse absent.",
    )


def enrich_transcript(context: PipelineContext) -> TaskResult:
    from pipeline.steps.transcripts.enrich_transcripts import (
        enrich_transcript as enrich,
        enriched_path,
        timecodes_path,
    )

    try:
        source = timecodes_path(
            context.video_path,
            transcripts_dir_name=context.transcripts_dir_name,
        )
    except FileNotFoundError:
        return TaskResult.blocked(
            "Enrichissement impossible; transcript corrige absent."
        )
    target = enriched_path(source)
    before = _snapshot((target,))
    result = enrich(
        context.video_path,
        force=context.force_rebuild,
        transcripts_dir_name=context.transcripts_dir_name,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason="Transcript enrichi avec les textes visuels.",
        cached_reason="Transcript enrichi deja a jour.",
        missing_reason="Enrichissement impossible; OCR filtre absent.",
    )


def create_plain_transcript(context: PipelineContext) -> TaskResult:
    from pipeline.steps.transcripts.create_plain_transcript import (
        convert_file,
        output_path,
        timecoded_inputs,
    )
    from pipeline.support.paths import existing_transcripts_dir

    sources = timecoded_inputs(
        existing_transcripts_dir(
            context.video_path,
            name=context.transcripts_dir_name,
        )
    )
    if not sources:
        return TaskResult.blocked(
            "Aucun transcript timecode corrige n'est disponible."
        )
    targets = [output_path(source) for source in sources]
    before = _snapshot(targets)
    results = [
        convert_file(
            context.video_path,
            source,
            force=context.force_rebuild,
            transcripts_dir_name=context.transcripts_dir_name,
        )
        for source in sources
    ]
    return _artifact_result(
        context,
        results,
        artifacts=(*_paths_from(results), *targets),
        state_paths=targets,
        before=before,
        success_reason="Transcript sans timecodes cree.",
        cached_reason="Transcript sans timecodes deja a jour.",
        missing_reason="Conversion sans timecodes n'a produit aucun fichier.",
    )


def create_chunks(context: PipelineContext) -> TaskResult:
    from pipeline.steps.chunks.create_transcript_chunks import (
        chunks_path,
        create_chunks as create,
    )

    target = chunks_path(context.video_path)
    before = _snapshot((target,))
    result = create(
        context.video_path,
        force=context.force_rebuild,
        profile=context.chunk_strategy,
        transcripts_dir_name=context.transcripts_dir_name,
    )
    return _artifact_result(
        context,
        result,
        artifacts=(*_paths_from(result), target),
        state_paths=(target,),
        before=before,
        success_reason=f"Chunks {context.chunk_strategy} crees.",
        cached_reason=f"Chunks {context.chunk_strategy} deja a jour.",
        missing_reason="Chunks non produits; transcript ou speakers valides absents.",
    )


def summarize_sections(context: PipelineContext) -> TaskResult:
    from pipeline.steps.chunks.hierarchical_chunks import (
        chunks_at_level,
        load_chunks,
        summarize_sections as summarize,
    )

    try:
        _payload, target = load_chunks(context.video_path)
    except FileNotFoundError:
        return TaskResult.blocked(
            "Resume des sections impossible; chunks absents."
        )
    before = _snapshot((target,))
    result = summarize(
        context.video_path,
        force=context.force_rebuild,
        details_per_section=context.options.details_per_section,
    )
    payload, target = load_chunks(context.video_path)
    artifacts = (target,) if chunks_at_level(payload, "section") else ()
    return _artifact_result(
        context,
        result,
        artifacts=artifacts,
        state_paths=(target,),
        before=before,
        success_reason="Resumes de sections crees.",
        cached_reason="Resumes de sections deja a jour.",
        missing_reason="Aucun resume de section produit; chunks detail invalides.",
    )


def summarize_video(context: PipelineContext) -> TaskResult:
    from pipeline.steps.chunks.hierarchical_chunks import (
        chunks_at_level,
        load_chunks,
        summarize_video as summarize,
    )

    try:
        _payload, target = load_chunks(context.video_path)
    except FileNotFoundError:
        return TaskResult.blocked(
            "Resume global impossible; chunks absents."
        )
    before = _snapshot((target,))
    result = summarize(
        context.video_path,
        force=context.force_rebuild,
    )
    payload, target = load_chunks(context.video_path)
    artifacts = (target,) if chunks_at_level(payload, "global") else ()
    return _artifact_result(
        context,
        result,
        artifacts=artifacts,
        state_paths=(target,),
        before=before,
        success_reason="Resume global cree.",
        cached_reason="Resume global deja a jour.",
        missing_reason="Aucun resume global produit; sections absentes.",
    )


def create_embeddings(context: PipelineContext) -> TaskResult:
    from openai import OpenAI
    from pipeline.steps.embeddings.create_chunk_embeddings import (
        DEFAULT_EMBEDDING_DIMENSIONS,
        DEFAULT_EMBEDDING_MODEL,
        chunks_dir,
        create_embeddings as create,
    )

    target_directory = chunks_dir(context.video_path)
    before_paths = sorted(target_directory.glob("*_embedding.json"))
    before = _snapshot(before_paths)
    result = create(
        OpenAI(),
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_EMBEDDING_DIMENSIONS,
        context.video_path,
        force=context.force_rebuild,
    )
    embedding_paths = sorted(target_directory.glob("*_embedding.json"))
    return _artifact_result(
        context,
        result,
        artifacts=embedding_paths,
        state_paths=embedding_paths,
        before=before,
        success_reason=f"{len(embedding_paths)} embedding(s) cree(s).",
        cached_reason=f"{len(embedding_paths)} embedding(s) deja a jour.",
        missing_reason="Aucun embedding produit; chunks sources absents ou vides.",
    )
