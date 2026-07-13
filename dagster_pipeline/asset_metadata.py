from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import dagster as dg

from dagster_pipeline.runtime import IMAGE_EXTENSIONS, explorer_metadata, read_json, read_text


_CLASS_NAMES = ("footage", "graphic", "mixture")
_IGNORED_STATE_PARTS = {".embedding_cache", "__pycache__"}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _value_count(value: Any) -> int:
    if isinstance(value, (dict, list)):
        return len(value)
    return 0


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _analysis_path(video_dir: Path) -> Path | None:
    return _first_existing(
        (
            video_dir / "metadata" / "pipeline_analysis.json",
            video_dir / "outputs" / "metadata" / "pipeline_analysis.json",
            video_dir / "metadata" / "analysed_infos.json",
            video_dir / "analysed_infos.json",
        )
    )


def _transcript_dirs(video_dir: Path) -> list[Path]:
    outputs = video_dir / "outputs"
    preferred = [outputs / "transcripts_ocr", outputs / "transcripts_whisper", outputs / "transcripts"]
    existing = [path for path in preferred if path.is_dir()]
    existing.extend(
        path
        for path in sorted(outputs.glob("transcripts*"))
        if path.is_dir() and path not in existing
    )
    return existing


def _transcript_file(video_dir: Path, names: Iterable[str], patterns: Iterable[str] = ()) -> Path | None:
    directories = _transcript_dirs(video_dir)
    direct = _first_existing(directory / name for directory in directories for name in names)
    if direct is not None:
        return direct
    return next(
        (
            path
            for directory in directories
            for pattern in patterns
            for path in sorted(directory.glob(pattern))
            if path.is_file()
        ),
        None,
    )


def _relative(path: Path, video_dir: Path) -> str:
    try:
        return path.relative_to(video_dir).as_posix()
    except ValueError:
        return str(path)


def _format_size(size: int) -> str:
    if size < 1_024:
        return f"{size} o"
    if size < 1_024 * 1_024:
        return f"{size / 1_024:.1f} Ko"
    return f"{size / 1_024 / 1_024:.1f} Mo"


def _files_metadata(video_dir: Path, paths: Iterable[Path | None]) -> dict[str, Any]:
    files: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path is None or not path.is_file() or path in seen:
            continue
        seen.add(path)
        files.append(path)
    if not files:
        return {}

    rows = [f"- `{_relative(path, video_dir)}` — {_format_size(path.stat().st_size)}" for path in files]
    return {
        "fichier_principal": dg.MetadataValue.path(str(files[0])),
        "fichiers_resultat": dg.MetadataValue.md("\n".join(rows)),
    }


def _text_stats(path: Path | None) -> dict[str, Any]:
    text = read_text(path, limit=1_000_000)
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "caracteres": len(text.strip()),
        "mots": len(text.split()),
        "lignes_non_vides": len(lines),
    }


def _text_preview(path: Path | None, limit: int = 2_500) -> dg.MetadataValue:
    text = read_text(path, limit=limit).strip()
    if not text:
        return dg.MetadataValue.md("_Aucun contenu textuel disponible._")
    suffix = "\n\n_… aperçu tronqué_" if path and path.stat().st_size > len(text.encode("utf-8")) else ""
    return dg.MetadataValue.md(text + suffix)


def _kind_counts(payload: dict[str, Any]) -> dict[str, int]:
    kinds = _dict(payload.get("kinds"))
    return {str(name): _value_count(value) for name, value in kinds.items()}


def _raw_ocr_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    raw_dir = video_dir / "outputs" / "ocr" / "raw"
    paths = [raw_dir / f"raw_ocr_{name}_frames.json" for name in _CLASS_NAMES]
    counts: dict[str, int] = {}
    for name, path in zip(_CLASS_NAMES, paths):
        counts[name] = len(_items(_dict(read_json(path)).get("items")))
    total = sum(counts.values())
    return (
        {
            "resultat": f"OCR brut extrait pour {total} image(s)",
            "images_ocrisees": total,
            "repartition_images": dg.MetadataValue.json(counts),
        },
        paths,
    )


def _classification_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    images_dir = video_dir / "outputs" / "images"
    manifest = images_dir / "frame_classification_manifest.json"
    features = images_dir / "frame_classification_features.json"
    payload = _dict(read_json(manifest))
    counts = _dict(payload.get("class_counts"))
    if not counts:
        counts = {name: int(payload.get(f"{name}_count", 0) or 0) for name in _CLASS_NAMES}
    total = sum(int(value or 0) for value in counts.values())
    return (
        {
            "resultat": f"{total} image(s) classée(s) en footage / graphic / mixture",
            "images_classees": total,
            "repartition_classes": dg.MetadataValue.json(counts),
            "methode": str(payload.get("identification_strategy") or payload.get("method") or "inconnue"),
            "modele": str(payload.get("model") or "inconnu"),
        },
        [manifest, features],
    )


def _interview_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest = video_dir / "outputs" / "interview" / "interview_detection_manifest.json"
    payload = _dict(read_json(manifest))
    thresholds = _dict(payload.get("thresholds"))
    sequence_count = int(payload.get("sequence_count", 0) or 0)
    max_sequences = int(thresholds.get("max_interview_sequences", 0) or 0)
    blocked_by_graphics = bool(payload.get("blocked_by_graphic_majority"))
    is_interview = bool(payload.get("is_interview"))
    if blocked_by_graphics:
        reason = "majorité d’images graphiques"
    elif is_interview:
        reason = f"{sequence_count} séquence(s), sous le seuil strict de {max_sequences}"
    else:
        reason = f"{sequence_count} séquence(s), seuil strict attendu : moins de {max_sequences}"
    sequences = [
        {
            "id": sequence.get("sequence_id"),
            "debut_s": sequence.get("start_second"),
            "fin_s": sequence.get("end_second"),
            "duree_s": sequence.get("duration_seconds"),
            "frames": sequence.get("frame_count"),
        }
        for sequence in _items(payload.get("sequences"))[:20]
        if isinstance(sequence, dict)
    ]
    return (
        {
            "resultat": "Interview détectée" if is_interview else "Pas une interview",
            "is_interview": is_interview,
            "raison": reason,
            "sequences_detectees": sequence_count,
            "images_candidates": int(payload.get("frame_count", 0) or 0),
            "images_dans_sequences": int(payload.get("selected_frame_count", 0) or 0),
            "bloque_par_majorite_graphique": blocked_by_graphics,
            "seuils": dg.MetadataValue.json(thresholds),
            "apercu_sequences": dg.MetadataValue.json(sequences),
        },
        [manifest],
    )


def _video_type_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    analysis_path = _analysis_path(video_dir)
    analysis = _dict(read_json(analysis_path))
    interview_path = video_dir / "outputs" / "interview" / "interview_detection_manifest.json"
    interview = _dict(read_json(interview_path))
    video_type = str(analysis.get("video_type") or "inconnu")
    if bool(interview.get("is_interview")):
        reason = "décision issue de la détection d’interview"
    elif video_type == "motion_design":
        reason = "proportion de footage insuffisante dans la classification des images"
    elif video_type == "video_recording":
        reason = "présence majoritaire de footage et interview non détectée"
    else:
        reason = "type non encore déterminé"
    return (
        {
            "resultat": f"Type de vidéo : {video_type}",
            "video_type": video_type,
            "raison": reason,
        },
        [analysis_path, interview_path],
    )


def _ocr_boxes_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    path = video_dir / "outputs" / "ocr" / "ocr_box_locations.json"
    payload = _dict(read_json(path))
    items = _items(payload.get("items"))
    boxes = sum(len(_items(item.get("boxes"))) for item in items if isinstance(item, dict))
    return (
        {
            "resultat": f"{boxes} zone(s) OCR localisée(s) sur {len(items)} image(s)",
            "images_analysees": len(items),
            "zones_ocr": boxes,
            "sources": dg.MetadataValue.json(_items(payload.get("sources"))),
        },
        [path],
    )


def _subtitle_route_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    path = _analysis_path(video_dir)
    payload = _dict(read_json(path))
    details = _dict(payload.get("has_subtitles_details"))
    has_subtitles = bool(payload.get("has_subtitles"))
    reason_names = {
        "stable_anchor_across_continuous_seconds": "ancrage de sous-titres stable et continu",
        "not_enough_continuous_matching_seconds": "durée continue insuffisante",
        "not_enough_anchor_candidates": "pas assez de candidats d’ancrage",
    }
    reason = reason_names.get(str(details.get("reason")), str(details.get("reason") or "inconnue"))
    route = "has_sub" if has_subtitles else "no_sub"
    compact_details = {
        "raison_technique": details.get("reason"),
        "ancrages": details.get("anchor_count"),
        "secondes_correspondantes": details.get("total_matching_seconds_count"),
        "plus_longue_sequence_secondes": details.get("longest_continuous_seconds_duration"),
        "minimum_requis_secondes": details.get("required_continuous_seconds"),
        "ancrage_median": details.get("anchor"),
    }
    return (
        {
            "resultat": "Sous-titres détectés" if has_subtitles else "Aucun sous-titre stable détecté",
            "has_subtitles": has_subtitles,
            "route_choisie": route,
            "raison": reason,
            "duree_continue_secondes": float(details.get("longest_continuous_seconds_duration", 0) or 0),
            "duree_minimum_secondes": float(details.get("required_continuous_seconds", 0) or 0),
            "details_detection": dg.MetadataValue.json(compact_details),
        },
        [path],
    )


def _processed_ocr_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    path = video_dir / "outputs" / "ocr" / "01_processed_ocr_items.json"
    payload = _dict(read_json(path))
    items = _items(payload.get("items"))
    return (
        {
            "resultat": f"{len(items)} élément(s) OCR consolidé(s)",
            "elements_ocr": len(items),
            "confiance_minimum": float(payload.get("min_confidence", 0) or 0),
            "sources": dg.MetadataValue.json(_items(payload.get("sources"))),
        },
        [path],
    )


def _filtered_ocr_metadata(video_dir: Path, reviewed: bool = False) -> tuple[dict[str, Any], list[Path]]:
    name = "03_reviewed_ocr_overlays.json" if reviewed else "02_filtered_ocr_overlays.json"
    path = video_dir / "outputs" / "ocr" / name
    payload = _dict(read_json(path))
    counts = _kind_counts(payload)
    metadata: dict[str, Any] = {
        "resultat": f"{sum(counts.values())} overlay(s) OCR retenu(s)",
        "elements_source": int(payload.get("source_item_count", 0) or 0),
        "elements_filtres": int(payload.get("filtered_item_count", 0) or 0),
        "repartition_par_type": dg.MetadataValue.json(counts),
    }
    if reviewed:
        review = _dict(payload.get("others_review"))
        metadata.update(
            {
                "revus": int(review.get("reviewed_count", 0) or 0),
                "conserves": int(review.get("kept_count", 0) or 0),
                "corriges": int(review.get("corrected_count", 0) or 0),
                "supprimes": int(review.get("removed_count", 0) or 0),
            }
        )
    return metadata, [path]


def _review_candidates_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    path = video_dir / "outputs" / "ocr" / "other_text_review_candidates" / "review_candidates_manifest.json"
    payload = _dict(read_json(path))
    items = _items(payload.get("items"))
    return (
        {
            "resultat": f"{len(items)} candidat(s) OCR à réviser",
            "candidats": len(items),
            "images_rendues": int(payload.get("image_count", 0) or 0),
            "type_ocr": str(payload.get("kind") or "inconnu"),
        },
        [path],
    )


def _review_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    root = video_dir / "outputs" / "ocr" / "other_text_gpt_review"
    path = root / "review_summary.json"
    payload = _dict(read_json(path))
    reviewed = int(payload.get("reviewed_count", 0) or 0)
    metadata = {
        "resultat": f"{reviewed} candidat(s) révisé(s) par le modèle",
        "modele": str(payload.get("model") or "inconnu"),
        "revus": reviewed,
        "ajoutes_ou_corriges": int(payload.get("added_in_edit_count", 0) or 0),
        "non_ajoutes": int(payload.get("not_added_count", 0) or 0),
    }
    decisions = sorted(root.glob("*/review_decision.json"))[:20] if root.is_dir() else []
    return metadata, [path, *decisions]


def _text_output_metadata(path: Path | None, label: str) -> tuple[dict[str, Any], list[Path | None]]:
    stats = _text_stats(path)
    metadata: dict[str, Any] = {
        "resultat": f"{label} : {stats['mots']} mot(s), {stats['lignes_non_vides']} ligne(s)",
        **stats,
        "apercu": _text_preview(path),
    }
    return metadata, [path]


def _correction_metadata(video_dir: Path, source: Path | None, target: Path | None, label: str) -> tuple[dict[str, Any], list[Path | None]]:
    metadata, paths = _text_output_metadata(target, label)
    source_lines = read_text(source, limit=1_000_000).splitlines()
    target_lines = read_text(target, limit=1_000_000).splitlines()
    changed = sum(left != right for left, right in zip(source_lines, target_lines)) + abs(len(source_lines) - len(target_lines))
    metadata["lignes_modifiees"] = changed
    return metadata, [target, source]


def _chunks_metadata(video_dir: Path, validated: bool = False) -> tuple[dict[str, Any], list[Path]]:
    chunks_dir = video_dir / "outputs" / "chunks"
    name = "transcript_chunks_speaker_validated.json" if validated else "transcript_chunks.json"
    path = chunks_dir / name
    payload = _dict(read_json(path))
    chunks = [item for item in _items(payload.get("chunks")) if isinstance(item, dict)]
    total_chars = sum(len(str(item.get("content") or item.get("text") or "")) for item in chunks)
    metadata: dict[str, Any] = {
        "resultat": f"{len(chunks)} chunk(s), {total_chars} caractère(s)",
        "chunks": len(chunks),
        "caracteres_cumules": total_chars,
        "source": str(payload.get("source") or "inconnue"),
        "parametres_chunking": dg.MetadataValue.json(_dict(payload.get("chunking"))),
    }
    paths: list[Path] = [path]
    if validated:
        log_path = chunks_dir / "speaker_validation_log.json"
        log = _dict(read_json(log_path))
        validation = _dict(payload.get("speaker_validation"))
        metadata.update(
            {
                "modele_validation": str(log.get("model") or validation.get("model") or "inconnu"),
                "locuteurs_revus": int(log.get("reviewed", validation.get("reviewed", 0)) or 0),
                "locuteurs_conserves": int(log.get("kept", validation.get("kept", 0)) or 0),
                "locuteurs_rejetes": int(log.get("rejected", validation.get("rejected", 0)) or 0),
                "locuteurs_valides": dg.MetadataValue.json(_items(validation.get("valid_speakers"))),
            }
        )
        paths.append(log_path)
    return metadata, paths


def _embeddings_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    chunks_dir = video_dir / "outputs" / "chunks"
    paths = sorted(chunks_dir.glob("*_embedding.json")) if chunks_dir.is_dir() else []
    first = _dict(read_json(paths[0])) if paths else {}
    expected = int(first.get("chunk_count", 0) or 0)
    dimension = len(_items(first.get("embedding")))
    complete = expected > 0 and len(paths) == expected
    return (
        {
            "resultat": f"{len(paths)}/{expected or '?'} embedding(s) généré(s)",
            "couverture_complete": complete,
            "embeddings": len(paths),
            "chunks_attendus": expected,
            "modele": str(first.get("model") or "inconnu"),
            "dimensions": dimension,
        },
        paths[:20],
    )


def _image_extraction_metadata(video_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    images_dir = video_dir / "outputs" / "images"
    counts: dict[str, int] = {}
    if images_dir.is_dir():
        root_count = sum(1 for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if root_count:
            counts["non_classees"] = root_count
        for name in _CLASS_NAMES:
            class_dir = images_dir / name
            counts[name] = sum(1 for path in class_dir.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS) if class_dir.is_dir() else 0
    total = sum(counts.values())
    return (
        {
            "resultat": f"{total} image(s) extraite(s)",
            "images_extraites": total,
            "repartition_actuelle": dg.MetadataValue.json(counts),
            "dossier_images": dg.MetadataValue.path(str(images_dir)),
        },
        [],
    )


def snapshot_result_files(video_dir: Path) -> dict[str, tuple[int, int]]:
    """Capture légère des sorties afin d'indiquer si l'étape a écrit ou réutilisé ses résultats."""

    roots = (video_dir / "outputs", video_dir / "metadata")
    state: dict[str, tuple[int, int]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or any(part in _IGNORED_STATE_PARTS for part in path.parts):
                continue
            stat = path.stat()
            state[_relative(path, video_dir)] = (stat.st_size, stat.st_mtime_ns)
    return state


def changed_result_files(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    return sorted(path for path, signature in after.items() if before.get(path) != signature)


def step_output_metadata(step_name: str, video_id: str, video_dir: Path) -> dict[str, Any]:
    outputs = video_dir / "outputs"
    ocr = outputs / "ocr"
    transcript_timecoded = _transcript_file(
        video_dir,
        ("ocr_subtitles_timecoded.txt", "whisper_transcript_timecoded.txt"),
        ("*_timecodes.txt", "*_timecoded.txt"),
    )
    corrected_timecoded = _transcript_file(
        video_dir,
        ("ocr_subtitles_timecoded_corrected.txt", "whisper_transcript_timecoded_corrected.txt"),
        ("*_corrected.txt",),
    )
    enriched = _transcript_file(video_dir, (), ("*_enriched.txt", "*_enrichi.txt"))
    plain = _transcript_file(video_dir, ("plain_transcript.txt",), ("*_transcript.txt",))
    summary = _transcript_file(video_dir, ("video_summary.md",), ("*_summary.md",))

    if step_name == "step_03_extract_images":
        metadata, files = _image_extraction_metadata(video_dir)
    elif step_name == "step_04_classify_images":
        metadata, files = _classification_metadata(video_dir)
    elif step_name == "step_05_detect_interviews":
        metadata, files = _interview_metadata(video_dir)
    elif step_name == "step_06_infer_video_type":
        metadata, files = _video_type_metadata(video_dir)
    elif step_name == "step_07_extract_raw_ocr":
        metadata, files = _raw_ocr_metadata(video_dir)
    elif step_name == "step_08_extract_ocr_boxes":
        metadata, files = _ocr_boxes_metadata(video_dir)
    elif step_name == "step_09_detect_ocr_subtitles":
        metadata, files = _subtitle_route_metadata(video_dir)
    elif "step_10_build_processed_ocr" in step_name:
        metadata, files = _processed_ocr_metadata(video_dir)
    elif "step_11_filter_processed_ocr" in step_name:
        metadata, files = _filtered_ocr_metadata(video_dir)
    elif "step_12_extract_review_candidates" in step_name:
        metadata, files = _review_candidates_metadata(video_dir)
    elif "step_13_review_candidates" in step_name:
        metadata, files = _review_metadata(video_dir)
    elif "step_14_apply_review" in step_name:
        metadata, files = _filtered_ocr_metadata(video_dir, reviewed=True)
    elif "step_15_has_sub_extract_subtitles" in step_name:
        metadata, files = _text_output_metadata(transcript_timecoded, "Sous-titres OCR extraits")
    elif "step_16_has_sub_fix_spacing" in step_name:
        metadata, files = _correction_metadata(video_dir, transcript_timecoded, corrected_timecoded, "Sous-titres corrigés")
    elif "step_17_has_sub_normalize" in step_name:
        metadata, files = _text_output_metadata(corrected_timecoded, "Vocabulaire IONIS-STM normalisé")
    elif "step_16_no_sub_whisper" in step_name:
        metadata, files = _text_output_metadata(transcript_timecoded, "Transcription WhisperX")
    elif "step_17_no_sub_correct_timecodes" in step_name:
        metadata, files = _correction_metadata(video_dir, transcript_timecoded, corrected_timecoded, "Transcript corrigé")
    elif "enrich_timecodes" in step_name:
        metadata, files = _text_output_metadata(enriched, "Transcript enrichi")
    elif "plain_transcript" in step_name:
        metadata, files = _text_output_metadata(plain, "Transcript final")
    elif step_name.endswith("_summary"):
        metadata, files = _text_output_metadata(summary, "Résumé vidéo")
    elif step_name.endswith("_chunks"):
        metadata, files = _chunks_metadata(video_dir)
    elif step_name.endswith("_validate_speakers"):
        metadata, files = _chunks_metadata(video_dir, validated=True)
    elif step_name.endswith("_embeddings"):
        metadata, files = _embeddings_metadata(video_dir)
    else:
        metadata = {"resultat": "Étape terminée"}
        files = []

    result = metadata.pop("resultat", "Étape terminée")
    return {
        "resultat": result,
        **explorer_metadata(video_id, video_dir),
        **metadata,
        **_files_metadata(video_dir, files),
        "dossier_video": dg.MetadataValue.path(str(video_dir)),
    }
