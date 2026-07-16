from __future__ import annotations

import shutil
from types import SimpleNamespace
from typing import Any

from .context import PipelineContext


def _finish(context: PipelineContext, task_id: str, result: Any) -> Any:
    context.record_artifacts(task_id, result)
    return result


def extract_frames(context: PipelineContext) -> Any:
    from pipeline.steps.inspection.extract_frames import extract_images
    from pipeline.support.paths import images_dir

    result = extract_images(
        context.video_path,
        context.options.frame_interval_seconds,
        force=context.options.force,
    )
    return _finish(context, "frames.extract", [images_dir(context.video_path), result])


def classify_frames(context: PipelineContext) -> Any:
    from pipeline.steps.inspection.classify_frames import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EMBEDDING_CACHE_DIRNAME,
        DEFAULT_MODEL_PATH,
        classify_video_images,
    )
    from pipeline.support.paths import existing_images_dir

    image_directory = existing_images_dir(context.video_path)
    args = SimpleNamespace(
        model=str(DEFAULT_MODEL_PATH),
        batch_size=DEFAULT_BATCH_SIZE,
        device=None,
        cache_dir=str(image_directory / DEFAULT_EMBEDDING_CACHE_DIRNAME),
        force=context.options.force,
    )
    return _finish(
        context,
        "frames.classify",
        classify_video_images(context.video_path, args),
    )


def detect_interview(context: PipelineContext) -> Any:
    from pipeline.steps.inspection.detect_interviews import (
        DEFAULT_MAX_INTERVIEW_SEQUENCES,
        SOURCE_DIR_NAMES,
        detect_for_video,
    )

    args = SimpleNamespace(
        source_dirs=list(SOURCE_DIR_NAMES),
        phash_similar_max=6,
        phash_ambiguous_max=14,
        ssim_min=0.92,
        min_run_frames=6,
        max_gap_pairs=1,
        max_interview_sequences=DEFAULT_MAX_INTERVIEW_SEQUENCES,
        force=context.options.force,
    )
    return _finish(
        context,
        "video.detect_interview",
        detect_for_video(context.video_path, args),
    )


def infer_video_type(context: PipelineContext) -> Any:
    from pipeline.steps.inspection.infer_video_type import infer_for_video

    return _finish(
        context,
        "video.infer_type",
        infer_for_video(context.video_path, force=context.options.force),
    )


def extract_raw_ocr(context: PipelineContext) -> Any:
    from pipeline.steps.inspection.extract_raw_ocr import extract_for_video

    return _finish(
        context,
        "ocr.extract_raw",
        extract_for_video(context.video_path, force=context.options.force),
    )


def extract_ocr_boxes(context: PipelineContext) -> Any:
    from pipeline.steps.inspection.extract_ocr_boxes import extract_for_video

    return _finish(
        context,
        "ocr.extract_boxes",
        extract_for_video(context.video_path, force=context.options.force),
    )


def detect_subtitles(context: PipelineContext) -> Any:
    from pipeline.steps.inspection.detect_subtitles import detect_for_video

    return _finish(
        context,
        "video.detect_subtitles",
        detect_for_video(context.video_path, force=context.options.force),
    )


def build_processed_ocr(context: PipelineContext) -> Any:
    from pipeline.steps.ocr.build_processed_ocr import process_video, processed_path
    from pipeline.support.paths import existing_ocr_dir

    process_video(
        context.video_path,
        force=context.options.force,
        strip_subtitles=context.transcript_strategy == "whisper",
    )
    return _finish(
        context,
        "ocr.build_processed",
        processed_path(existing_ocr_dir(context.video_path)),
    )


def filter_ocr_overlays(context: PipelineContext) -> Any:
    from pipeline.steps.ocr.filter_processed_ocr import filter_processed_ocr

    return _finish(
        context,
        "ocr.filter_overlays",
        filter_processed_ocr(context.video_path, force=context.options.force),
    )


def extract_review_candidates(context: PipelineContext) -> Any:
    from pipeline.steps.ocr.extract_other_text_candidates import extract_for_video

    return _finish(
        context,
        "ocr.extract_review_candidates",
        extract_for_video(context.video_path, force=context.options.force),
    )


def review_other_text(context: PipelineContext) -> Any:
    from pipeline.steps.ocr.review_other_text_candidates import review_video

    mode = "batch" if context.options.openai_mode == "batch" else "live"
    result = review_video(
        context.video_path,
        context.options.image_review_model,
        mode=mode,
        force=context.options.force,
        limit_images=1 if context.options.review_scope == "duo" else None,
        wait=True,
    )
    return _finish(context, "ocr.review_other_text", result)


def apply_ocr_review(context: PipelineContext) -> Any:
    from pipeline.steps.ocr.apply_other_text_review import apply_review

    return _finish(
        context,
        "ocr.apply_review",
        apply_review(context.video_path, force=context.options.force),
    )


def extract_ocr_transcript(context: PipelineContext) -> Any:
    from pipeline.steps.transcripts.extract_ocr_subtitles import extract_for_video

    return _finish(
        context,
        "transcript.extract_ocr",
        extract_for_video(context.video_path, force=context.options.force),
    )


def correct_ocr_spacing(context: PipelineContext) -> Any:
    from pipeline.steps.transcripts.correct_ocr_subtitle_spacing import (
        openai_client,
        process_video_batch,
        process_video_live,
    )

    if context.options.openai_mode == "batch":
        result = process_video_batch(
            context.options.speaker_validation_model,
            context.video_path,
            force=context.options.force,
            wait=True,
        )
    else:
        result = process_video_live(
            openai_client(),
            context.options.speaker_validation_model,
            context.video_path,
            force=context.options.force,
        )
    return _finish(context, "transcript.correct_ocr_spacing", result)


def normalize_brand(context: PipelineContext) -> Any:
    from pipeline.steps.transcripts.normalize_ionis_stm import (
        process_video,
        transcript_path,
    )

    process_video(context.video_path, force=context.options.force)
    return _finish(
        context,
        "transcript.normalize_brand",
        transcript_path(context.video_path),
    )


def transcribe_whisper(context: PipelineContext) -> Any:
    from pipeline.steps.transcripts.transcribe_with_whisper import (
        DEFAULT_MAX_SPEAKERS,
        DEFAULT_MIN_SPEAKERS,
        load_diarization_pipeline,
        load_whisperx_model,
        transcribe_video,
    )
    from pipeline.support.paths import transcripts_dir

    whisperx, model, device = load_whisperx_model()
    diarization_pipeline, diarization_device = load_diarization_pipeline(device)
    transcript_directory = transcripts_dir(context.video_path)
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
            force=context.options.force,
        )
    finally:
        shutil.rmtree(audio_directory, ignore_errors=True)
    return _finish(context, "transcript.whisper", result)


def propose_speakers(context: PipelineContext) -> Any:
    from pipeline.steps.speakers.propose_speakers import propose_for_video

    return _finish(
        context,
        "speakers.propose",
        propose_for_video(context.video_path, force=context.options.force),
    )


def validate_speakers(context: PipelineContext) -> Any:
    from pipeline.steps.speakers.validate_speakers import (
        normalize_model_name,
        openai_client,
        validate_file_batch,
        validate_file_live,
    )

    model = normalize_model_name(context.options.speaker_validation_model)
    if context.options.openai_mode == "batch":
        result = validate_file_batch(
            model,
            context.video_path,
            force=context.options.force,
            wait=True,
        )
    else:
        result = validate_file_live(
            openai_client(),
            model,
            context.video_path,
            force=context.options.force,
        )
    return _finish(context, "speakers.validate", result)


def assign_ocr_speakers(context: PipelineContext) -> Any:
    from pipeline.steps.speakers.assign_ocr_speakers import assign_ocr_speakers

    return _finish(
        context,
        "speakers.assign_ocr",
        assign_ocr_speakers(context.video_path, force=context.options.force),
    )


def correct_whisper_transcript(context: PipelineContext) -> Any:
    from pipeline.steps.transcripts.correct_whisper_transcript import correct_file

    return _finish(
        context,
        "transcript.correct_whisper",
        correct_file(
            context.video_path,
            force=context.options.force,
            mode=context.options.correction_mode,
        ),
    )


def enrich_transcript(context: PipelineContext) -> Any:
    from pipeline.steps.transcripts.enrich_transcripts import enrich_transcript

    return _finish(
        context,
        "transcript.enrich",
        enrich_transcript(context.video_path, force=context.options.force),
    )


def create_plain_transcript(context: PipelineContext) -> Any:
    from pipeline.steps.transcripts.create_plain_transcript import (
        convert_file,
        timecoded_inputs,
    )
    from pipeline.support.paths import existing_transcripts_dir

    results = [
        convert_file(context.video_path, source, force=context.options.force)
        for source in timecoded_inputs(existing_transcripts_dir(context.video_path))
    ]
    return _finish(context, "transcript.create_plain", results)


def create_chunks(context: PipelineContext) -> Any:
    from pipeline.steps.chunks.create_transcript_chunks import create_chunks

    return _finish(
        context,
        "chunks.create",
        create_chunks(
            context.video_path,
            force=context.options.force,
            profile=context.chunk_strategy,
        ),
    )


def summarize_sections(context: PipelineContext) -> Any:
    from pipeline.steps.chunks.hierarchical_chunks import summarize_sections

    return _finish(
        context,
        "chunks.summarize_sections",
        summarize_sections(
            context.video_path,
            force=context.options.force,
            details_per_section=context.options.details_per_section,
        ),
    )


def summarize_video(context: PipelineContext) -> Any:
    from pipeline.steps.chunks.hierarchical_chunks import summarize_video

    return _finish(
        context,
        "chunks.summarize_video",
        summarize_video(context.video_path, force=context.options.force),
    )


def create_embeddings(context: PipelineContext) -> Any:
    from openai import OpenAI
    from pipeline.steps.embeddings.create_chunk_embeddings import (
        DEFAULT_EMBEDDING_DIMENSIONS,
        DEFAULT_EMBEDDING_MODEL,
        chunks_dir,
        create_embeddings,
    )

    create_embeddings(
        OpenAI(),
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_EMBEDDING_DIMENSIONS,
        context.video_path,
        force=context.options.force,
    )
    embedding_paths = sorted(
        chunks_dir(context.video_path).glob("*_embedding.json")
    )
    return _finish(context, "embeddings.create", embedding_paths)
