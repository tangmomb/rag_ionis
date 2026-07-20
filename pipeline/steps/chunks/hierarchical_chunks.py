from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from pipeline.support.json_io import read_json, write_json, write_jsonl
from pipeline.support.openai_batch import (
    COMPLETED_BATCH_STATUS,
    batch_request_fingerprint,
    batch_state_matches,
    download_batch_files,
    is_terminal_batch_status,
    load_batch_state,
    parse_jsonl,
    poll_batch_state,
    records_by_custom_id,
    save_batch_state,
)
from pipeline.support.paths import chunks_dir, existing_chunks_dir


CHUNKS_NAME = "transcript_chunks.json"
SUMMARY_STRATEGY = "luna"
DEFAULT_SUMMARY_MODEL = os.getenv("CHUNK_SUMMARY_MODEL", "gpt-5.6-luna")
DEFAULT_DETAILS_PER_SECTION = 6
DEFAULT_SECTION_SENTENCES = 4
DEFAULT_SECTION_MAX_CHARS = 1200
DEFAULT_GLOBAL_SENTENCES = 6
DEFAULT_GLOBAL_MAX_CHARS = 1800
CHUNK_LEVEL_ORDER = {"global": 0, "section": 1, "detail": 2}
SECTION_BATCH_STATE_NAME = "section_summary_batch_state.json"
SECTION_BATCH_INPUT_NAME = "section_summary_batch_input.jsonl"
SECTION_BATCH_OUTPUT_NAME = "section_summary_batch_output.jsonl"
SECTION_BATCH_ERROR_NAME = "section_summary_batch_error.jsonl"
GLOBAL_BATCH_STATE_NAME = "global_summary_batch_state.json"
GLOBAL_BATCH_INPUT_NAME = "global_summary_batch_input.jsonl"
GLOBAL_BATCH_OUTPUT_NAME = "global_summary_batch_output.jsonl"
GLOBAL_BATCH_ERROR_NAME = "global_summary_batch_error.jsonl"
SUMMARY_SYSTEM_PROMPT = (
    "Tu resumes un transcript en francais. Produis un resume fidel, "
    "clair et autonome. N'invente aucune information. Retourne uniquement "
    "un objet JSON avec la cle summary."
)

FRENCH_STOP_WORDS = {
    "alors",
    "au",
    "aucun",
    "aussi",
    "autre",
    "aux",
    "avec",
    "avoir",
    "bon",
    "car",
    "ce",
    "cela",
    "ces",
    "ceux",
    "chaque",
    "comme",
    "comment",
    "dans",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "encore",
    "est",
    "et",
    "eux",
    "faire",
    "fois",
    "il",
    "ils",
    "je",
    "juste",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "lui",
    "mais",
    "mes",
    "moi",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ou",
    "où",
    "par",
    "pas",
    "peu",
    "plus",
    "pour",
    "pourquoi",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "sans",
    "se",
    "ses",
    "si",
    "son",
    "sont",
    "sur",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tous",
    "tout",
    "très",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "être",
}


def chunks_path(video_path: Path) -> Path:
    return existing_chunks_dir(video_path) / CHUNKS_NAME


def load_chunks(video_path: Path) -> tuple[dict[str, Any], Path]:
    target = chunks_path(video_path)
    if not target.exists():
        raise FileNotFoundError(f"Chunks introuvables: {target}")
    payload = read_json(target)
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
        raise ValueError(f"Format de chunks invalide: {target}")
    return payload, target


def write_chunks(video_path: Path, payload: dict[str, Any]) -> Path:
    target = chunks_dir(video_path) / CHUNKS_NAME
    write_json(target, payload)
    return target


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    pieces = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(str(text))
    return "\n".join(pieces).strip()


def _response_text_from_payload(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if output_text:
        return str(output_text).strip()
    pieces = []
    for output in payload.get("output", []) or []:
        for content in output.get("content", []) or []:
            text = content.get("text")
            if text:
                pieces.append(str(text))
    return "\n".join(pieces).strip()


def summary_response_body(
    text: str,
    *,
    max_sentences: int,
    max_chars: int,
    model: str,
) -> dict[str, Any]:
    return {
        "model": model or DEFAULT_SUMMARY_MODEL,
        "input": [
            {
                "role": "system",
                "content": SUMMARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Resume le texte en {max_sentences} phrases maximum et "
                    f"{max_chars} caracteres maximum.\n\nTEXTE:\n{text}"
                ),
            },
        ],
        "max_output_tokens": max(256, max_chars // 2),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "chunk_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            }
        },
    }


def parse_summary_response(raw: str) -> str:
    if not raw:
        raise RuntimeError("Luna n'a renvoye aucun resume.")
    try:
        summary = json.loads(raw).get("summary", "")
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Reponse Luna non JSON: {raw[:300]!r}") from error
    return str(summary).strip()


def luna_summary(text: str, *, max_sentences: int, max_chars: int, model: str) -> str:
    from openai import OpenAI

    if not str(text).strip():
        return ""
    client = OpenAI()
    response = client.responses.create(
        **summary_response_body(
            text,
            max_sentences=max_sentences,
            max_chars=max_chars,
            model=model,
        )
    )
    return parse_summary_response(_response_text(response))


def chunks_hash(chunks: Iterable[dict[str, Any]]) -> str:
    canonical = [
        {
            "chunk_index": chunk.get("chunk_index"),
            "chunk_level": chunk.get("chunk_level"),
            "content": str(chunk.get("content") or ""),
        }
        for chunk in chunks
    ]
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def chunk_level(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_level") or "detail").strip().lower()


def chunks_at_level(payload: dict[str, Any], level: str) -> list[dict[str, Any]]:
    return [
        dict(chunk)
        for chunk in payload.get("chunks", [])
        if isinstance(chunk, dict) and chunk_level(chunk) == level
    ]


def _without_speaker_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(chunk)
    cleaned.pop("speakers", None)
    meta_data = cleaned.get("meta_data")
    if isinstance(meta_data, dict):
        cleaned_meta_data = dict(meta_data)
        cleaned_meta_data.pop("speakers", None)
        if cleaned_meta_data:
            cleaned["meta_data"] = cleaned_meta_data
        else:
            cleaned.pop("meta_data", None)
    return cleaned


def _section_groups(
    details: list[dict[str, Any]],
    details_per_section: int,
) -> list[list[dict[str, Any]]]:
    return [
        details[index : index + details_per_section]
        for index in range(0, len(details), details_per_section)
    ]


def section_summary_jobs(
    details: list[dict[str, Any]],
    details_per_section: int,
) -> list[dict[str, Any]]:
    return [
        {
            "section_index": section_index,
            "custom_id": f"section-summary-{section_index:05d}",
            "details": group,
            "content": "\n\n".join(
                str(chunk.get("content") or "").strip()
                for chunk in group
            ),
        }
        for section_index, group in enumerate(
            _section_groups(details, details_per_section),
            start=1,
        )
    ]


def apply_section_summaries(
    video_path: Path,
    payload: dict[str, Any],
    jobs: list[dict[str, Any]],
    summaries: dict[str, str],
    *,
    details_per_section: int,
    detail_hash: str,
    hierarchy: dict[str, Any],
) -> Path:
    sections: list[dict[str, Any]] = []
    updated_details: list[dict[str, Any]] = []
    for job in jobs:
        custom_id = job["custom_id"]
        if custom_id not in summaries:
            raise RuntimeError(f"Resume de section manquant: {custom_id}")
        summary = summaries[custom_id]
        section_index = job["section_index"]
        group = job["details"]
        sections.append(
            {
                "chunk_index": section_index,
                "chunk_level": "section",
                "chunk_parent_id": None,
                "chunk_parent": None,
                "meta_data": {
                    "summary_strategy": SUMMARY_STRATEGY,
                    "detail_chunk_indexes": [
                        chunk.get("chunk_index")
                        for chunk in group
                    ],
                },
                "content": summary,
                "char_count": len(summary),
            }
        )
        for detail in group:
            updated = _without_speaker_metadata(detail)
            updated["chunk_parent_id"] = None
            updated["chunk_parent"] = {
                "chunk_level": "section",
                "chunk_index": section_index,
            }
            updated_details.append(updated)

    payload["chunks"] = [*sections, *updated_details]
    payload["hierarchy"] = {
        **hierarchy,
        "strategy": SUMMARY_STRATEGY,
        "details_per_section": details_per_section,
        "sections_source_hash": detail_hash,
        "section_count": len(sections),
        "global_source_hash": None,
    }
    written = write_chunks(video_path, payload)
    print(
        f"[ok] {video_path.name}: {len(sections)} section(s) resumee(s) "
        f"-> {written}"
    )
    return written


def section_batch_path(video_path: Path, name: str) -> Path:
    return chunks_dir(video_path) / name


def section_batch_fingerprint(
    model: str,
    jobs: list[dict[str, Any]],
    *,
    details_per_section: int,
) -> str:
    return batch_request_fingerprint(
        model or DEFAULT_SUMMARY_MODEL,
        custom_ids=(job["custom_id"] for job in jobs),
        options={
            "workflow": "section_summaries",
            "detail_hash": chunks_hash(
                detail
                for job in jobs
                for detail in job["details"]
            ),
            "details_per_section": details_per_section,
            "max_sentences": DEFAULT_SECTION_SENTENCES,
            "max_chars": DEFAULT_SECTION_MAX_CHARS,
            "system_prompt": SUMMARY_SYSTEM_PROMPT,
            "requests": [
                {
                    "custom_id": job["custom_id"],
                    "content": job["content"],
                }
                for job in jobs
            ],
        },
    )


def submit_section_summary_batch(
    video_path: Path,
    model: str,
    jobs: list[dict[str, Any]],
    *,
    request_fingerprint: str,
) -> Path:
    from openai import OpenAI

    client = OpenAI()
    input_path = section_batch_path(video_path, SECTION_BATCH_INPUT_NAME)
    state_path = section_batch_path(video_path, SECTION_BATCH_STATE_NAME)
    records = [
        {
            "custom_id": job["custom_id"],
            "method": "POST",
            "url": "/v1/responses",
            "body": summary_response_body(
                job["content"],
                max_sentences=DEFAULT_SECTION_SENTENCES,
                max_chars=DEFAULT_SECTION_MAX_CHARS,
                model=model,
            ),
        }
        for job in jobs
    ]
    write_jsonl(input_path, records)
    with input_path.open("rb") as batch_file:
        uploaded = client.files.create(file=batch_file, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "script": "chunks.summarize_sections",
            "model": model or DEFAULT_SUMMARY_MODEL,
            "video": Path(video_path).name,
        },
    )
    state = {
        "mode": "batch",
        "model": model or DEFAULT_SUMMARY_MODEL,
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": uploaded.id,
        "submitted_count": len(jobs),
        "request_fingerprint": request_fingerprint,
    }
    save_batch_state(state_path, state)
    print(
        f"[batch] sections submitted id={batch.id} "
        f"status={batch.status} requests={len(jobs)}",
        flush=True,
    )
    return state_path


def finalize_section_summary_batch(
    video_path: Path,
    jobs: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    details_per_section: int,
) -> Path:
    from openai import OpenAI

    if not state.get("output_file_id"):
        raise RuntimeError("Batch de sections complete mais output_file_id absent.")
    client = OpenAI()
    output_path = section_batch_path(video_path, SECTION_BATCH_OUTPUT_NAME)
    error_path = section_batch_path(video_path, SECTION_BATCH_ERROR_NAME)
    download_batch_files(client, state, output_path, error_path)
    records = records_by_custom_id(parse_jsonl(output_path))
    missing_ids = [
        job["custom_id"]
        for job in jobs
        if job["custom_id"] not in records
    ]
    if missing_ids:
        raise RuntimeError(
            "Resultats batch de sections incomplets; custom_id manquant(s): "
            + ", ".join(missing_ids)
        )

    summaries: dict[str, str] = {}
    for job in jobs:
        custom_id = job["custom_id"]
        response = records[custom_id].get("response") or {}
        if response.get("status_code") != 200:
            raise RuntimeError(
                f"Batch {custom_id} en echec avec "
                f"status={response.get('status_code')}"
            )
        raw_payload = response.get("body") or {}
        summaries[custom_id] = parse_summary_response(
            _response_text_from_payload(raw_payload)
        )

    payload, _target = load_chunks(video_path)
    details = chunks_at_level(payload, "detail")
    detail_hash = chunks_hash(details)
    hierarchy = (
        payload.get("hierarchy")
        if isinstance(payload.get("hierarchy"), dict)
        else {}
    )
    return apply_section_summaries(
        video_path,
        payload,
        jobs,
        summaries,
        details_per_section=details_per_section,
        detail_hash=detail_hash,
        hierarchy=hierarchy,
    )


def summarize_sections_batch(
    video_path: Path,
    *,
    force: bool = False,
    details_per_section: int = DEFAULT_DETAILS_PER_SECTION,
    model: str = DEFAULT_SUMMARY_MODEL,
    wait: bool = True,
    poll_interval_seconds: float = 30,
    reset_batch: bool | None = None,
) -> Path | None:
    payload, target = load_chunks(video_path)
    profile = str(
        payload.get("chunking", {}).get("profile") or "short"
    ).strip().lower()
    if profile != "long":
        print(
            f"[skip] {video_path.name}: profil de chunks={profile}, "
            "long attendu"
        )
        return None
    details = chunks_at_level(payload, "detail")
    if not details:
        print(f"[skip] aucun chunk detail: {target}")
        return None
    detail_hash = chunks_hash(details)
    hierarchy = (
        payload.get("hierarchy")
        if isinstance(payload.get("hierarchy"), dict)
        else {}
    )
    existing_sections = chunks_at_level(payload, "section")
    if (
        existing_sections
        and not force
        and hierarchy.get("strategy") == SUMMARY_STRATEGY
        and hierarchy.get("sections_source_hash") == detail_hash
    ):
        print(f"[skip] {video_path.name}: resumes de sections a jour")
        return target

    jobs = section_summary_jobs(details, details_per_section)
    state_path = section_batch_path(video_path, SECTION_BATCH_STATE_NAME)
    batch_artifacts = (
        state_path,
        section_batch_path(video_path, SECTION_BATCH_INPUT_NAME),
        section_batch_path(video_path, SECTION_BATCH_OUTPUT_NAME),
        section_batch_path(video_path, SECTION_BATCH_ERROR_NAME),
    )
    if reset_batch is None:
        reset_batch = force
    if reset_batch:
        for artifact_path in batch_artifacts:
            artifact_path.unlink(missing_ok=True)

    request_fingerprint = section_batch_fingerprint(
        model,
        jobs,
        details_per_section=details_per_section,
    )
    state = load_batch_state(state_path)
    if state is not None and not batch_state_matches(
        state,
        request_fingerprint,
    ):
        print(
            "[batch] etat sections incompatible; nouvelle soumission "
            f"(ancien batch_id={state.get('batch_id', 'inconnu')})",
            flush=True,
        )
        state = None
    if state is None:
        state_path = submit_section_summary_batch(
            video_path,
            model,
            jobs,
            request_fingerprint=request_fingerprint,
        )
        if not wait:
            return state_path
        state = load_batch_state(state_path)
        if state is None:
            raise RuntimeError("Etat batch de sections introuvable.")

    from openai import OpenAI

    state = poll_batch_state(
        OpenAI(),
        state,
        state_path,
        wait=wait,
        poll_interval_seconds=poll_interval_seconds,
        on_wait=lambda current, seconds: print(
            f"[batch] sections status={current['status']} "
            f"batch_id={current['batch_id']} attente {seconds}s",
            flush=True,
        ),
    )
    if not is_terminal_batch_status(state["status"]):
        print(
            f"[batch] sections status={state['status']} "
            f"batch_id={state['batch_id']}",
            flush=True,
        )
        return state_path
    if state["status"] != COMPLETED_BATCH_STATUS:
        raise RuntimeError(
            f"Batch de sections termine avec statut non supporte: "
            f"{state['status']}"
        )
    return finalize_section_summary_batch(
        video_path,
        jobs,
        state,
        details_per_section=details_per_section,
    )


def summarize_sections(
    video_path: Path,
    *,
    force: bool = False,
    details_per_section: int = DEFAULT_DETAILS_PER_SECTION,
    model: str = DEFAULT_SUMMARY_MODEL,
    mode: str = "live",
    reset_batch: bool | None = None,
) -> Path | None:
    if mode == "batch":
        return summarize_sections_batch(
            video_path,
            force=force,
            details_per_section=details_per_section,
            model=model,
            wait=True,
            reset_batch=reset_batch,
        )
    if mode != "live":
        raise ValueError(f"Mode de resume de sections invalide: {mode!r}")
    payload, target = load_chunks(video_path)
    profile = str(payload.get("chunking", {}).get("profile") or "short").strip().lower()
    if profile != "long":
        print(f"[skip] {video_path.name}: profil de chunks={profile}, long attendu")
        return None

    details = chunks_at_level(payload, "detail")
    if not details:
        print(f"[skip] aucun chunk detail: {target}")
        return None

    detail_hash = chunks_hash(details)
    hierarchy = payload.get("hierarchy") if isinstance(payload.get("hierarchy"), dict) else {}
    existing_sections = chunks_at_level(payload, "section")
    if (
        existing_sections
        and not force
        and hierarchy.get("strategy") == SUMMARY_STRATEGY
        and hierarchy.get("sections_source_hash") == detail_hash
    ):
        print(f"[skip] {video_path.name}: resumes de sections a jour")
        return target

    sections: list[dict[str, Any]] = []
    updated_details: list[dict[str, Any]] = []
    for section_index, group in enumerate(
        _section_groups(details, details_per_section),
        start=1,
    ):
        combined = "\n\n".join(str(chunk.get("content") or "").strip() for chunk in group)
        summary = luna_summary(
            combined,
            max_sentences=DEFAULT_SECTION_SENTENCES,
            max_chars=DEFAULT_SECTION_MAX_CHARS,
            model=model,
        )
        sections.append(
            {
                "chunk_index": section_index,
                "chunk_level": "section",
                "chunk_parent_id": None,
                "chunk_parent": None,
                "meta_data": {
                    "summary_strategy": SUMMARY_STRATEGY,
                    "detail_chunk_indexes": [chunk.get("chunk_index") for chunk in group],
                },
                "content": summary,
                "char_count": len(summary),
            }
        )
        for detail in group:
            updated = _without_speaker_metadata(detail)
            updated["chunk_parent_id"] = None
            updated["chunk_parent"] = {
                "chunk_level": "section",
                "chunk_index": section_index,
            }
            updated_details.append(updated)

    payload["chunks"] = [*sections, *updated_details]
    payload["hierarchy"] = {
        **hierarchy,
        "strategy": SUMMARY_STRATEGY,
        "details_per_section": details_per_section,
        "sections_source_hash": detail_hash,
        "section_count": len(sections),
        "global_source_hash": None,
    }
    written = write_chunks(video_path, payload)
    print(f"[ok] {video_path.name}: {len(sections)} section(s) resumee(s) -> {written}")
    return written


def apply_global_summary(
    video_path: Path,
    payload: dict[str, Any],
    sections: list[dict[str, Any]],
    summary: str,
    *,
    section_hash: str,
    hierarchy: dict[str, Any],
) -> Path:
    global_chunk = {
        "chunk_index": 1,
        "chunk_level": "global",
        "chunk_parent_id": None,
        "chunk_parent": None,
        "meta_data": {
            "summary_strategy": SUMMARY_STRATEGY,
            "section_chunk_indexes": [chunk.get("chunk_index") for chunk in sections],
        },
        "content": summary,
        "char_count": len(summary),
    }

    updated_sections: list[dict[str, Any]] = []
    for section in sections:
        updated = _without_speaker_metadata(section)
        updated["chunk_parent_id"] = None
        updated["chunk_parent"] = {
            "chunk_level": "global",
            "chunk_index": 1,
        }
        updated_sections.append(updated)

    details = [
        _without_speaker_metadata(detail)
        for detail in chunks_at_level(payload, "detail")
    ]
    payload["chunks"] = [global_chunk, *updated_sections, *details]
    payload["hierarchy"] = {
        **hierarchy,
        "strategy": SUMMARY_STRATEGY,
        "global_source_hash": section_hash,
        "global_chunk_count": 1,
    }
    written = write_chunks(video_path, payload)
    print(f"[ok] {video_path.name}: resume global -> {written}")
    return written


def summarize_video_batch(
    video_path: Path,
    *,
    force: bool = False,
    model: str = DEFAULT_SUMMARY_MODEL,
    wait: bool = True,
    poll_interval_seconds: float = 30,
    reset_batch: bool | None = None,
) -> Path | None:
    from openai import OpenAI

    payload, target = load_chunks(video_path)
    sections = chunks_at_level(payload, "section")
    if not sections:
        print(f"[skip] aucun chunk section: {target}")
        return None

    section_hash = chunks_hash(sections)
    hierarchy = payload.get("hierarchy") if isinstance(payload.get("hierarchy"), dict) else {}
    existing_global = chunks_at_level(payload, "global")
    if (
        existing_global
        and not force
        and hierarchy.get("strategy") == SUMMARY_STRATEGY
        and hierarchy.get("global_source_hash") == section_hash
    ):
        print(f"[skip] {video_path.name}: resume global a jour")
        return target

    combined = "\n\n".join(str(chunk.get("content") or "").strip() for chunk in sections)
    body = summary_response_body(
        combined,
        max_sentences=DEFAULT_GLOBAL_SENTENCES,
        max_chars=DEFAULT_GLOBAL_MAX_CHARS,
        model=model,
    )
    state_path = section_batch_path(video_path, GLOBAL_BATCH_STATE_NAME)
    batch_artifacts = (
        state_path,
        section_batch_path(video_path, GLOBAL_BATCH_INPUT_NAME),
        section_batch_path(video_path, GLOBAL_BATCH_OUTPUT_NAME),
        section_batch_path(video_path, GLOBAL_BATCH_ERROR_NAME),
    )
    if reset_batch is None:
        reset_batch = force
    if reset_batch:
        for artifact_path in batch_artifacts:
            artifact_path.unlink(missing_ok=True)

    request_fingerprint = batch_request_fingerprint(
        model or DEFAULT_SUMMARY_MODEL,
        sources=(target,),
        custom_ids=("global-summary",),
        options={"workflow": "global_summary", "request_body": body},
    )
    state = load_batch_state(state_path)
    if state is not None and not batch_state_matches(state, request_fingerprint):
        print(
            "[batch] etat global incompatible; nouvelle soumission "
            f"(ancien batch_id={state.get('batch_id', 'inconnu')})",
            flush=True,
        )
        state = None
    if state is None:
        input_path = section_batch_path(video_path, GLOBAL_BATCH_INPUT_NAME)
        write_jsonl(
            input_path,
            [{
                "custom_id": "global-summary",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }],
        )
        client = OpenAI()
        with input_path.open("rb") as batch_file:
            uploaded = client.files.create(file=batch_file, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "workflow": "global_summary",
                "model": model or DEFAULT_SUMMARY_MODEL,
                "video": Path(video_path).name,
            },
        )
        state = {
            "mode": "batch",
            "model": model or DEFAULT_SUMMARY_MODEL,
            "batch_id": batch.id,
            "status": batch.status,
            "input_file_id": uploaded.id,
            "submitted_count": 1,
            "request_fingerprint": request_fingerprint,
        }
        save_batch_state(state_path, state)
        print(
            f"[batch] global submitted id={batch.id} status={batch.status} requests=1",
            flush=True,
        )
        if not wait:
            return state_path

    client = OpenAI()
    state = poll_batch_state(
        client,
        state,
        state_path,
        wait=wait,
        poll_interval_seconds=poll_interval_seconds,
        on_wait=lambda current, seconds: print(
            f"[batch] global status={current['status']} "
            f"batch_id={current['batch_id']} attente {seconds}s",
            flush=True,
        ),
    )
    if not is_terminal_batch_status(state["status"]):
        return state_path
    if state["status"] != COMPLETED_BATCH_STATUS:
        raise RuntimeError(
            f"Batch du resume global termine avec statut non supporte: {state['status']}"
        )
    if not state.get("output_file_id"):
        raise RuntimeError("Batch du resume global complete mais output_file_id absent.")

    output_path = section_batch_path(video_path, GLOBAL_BATCH_OUTPUT_NAME)
    error_path = section_batch_path(video_path, GLOBAL_BATCH_ERROR_NAME)
    download_batch_files(client, state, output_path, error_path)
    record = records_by_custom_id(parse_jsonl(output_path)).get("global-summary")
    if record is None:
        raise RuntimeError("Resultat batch introuvable pour global-summary.")
    response = record.get("response") or {}
    if response.get("status_code") != 200:
        raise RuntimeError(
            f"Batch global-summary en echec avec status={response.get('status_code')}"
        )
    summary = parse_summary_response(
        _response_text_from_payload(response.get("body") or {})
    )
    return apply_global_summary(
        video_path,
        payload,
        sections,
        summary,
        section_hash=section_hash,
        hierarchy=hierarchy,
    )


def summarize_video(
    video_path: Path,
    *,
    force: bool = False,
    model: str = DEFAULT_SUMMARY_MODEL,
    mode: str = "live",
    reset_batch: bool | None = None,
) -> Path | None:
    if mode == "batch":
        return summarize_video_batch(
            video_path,
            force=force,
            model=model,
            wait=True,
            reset_batch=reset_batch,
        )
    if mode != "live":
        raise ValueError(f"Mode de resume global invalide: {mode!r}")
    payload, target = load_chunks(video_path)
    sections = chunks_at_level(payload, "section")
    if not sections:
        print(f"[skip] aucun chunk section: {target}")
        return None

    section_hash = chunks_hash(sections)
    hierarchy = payload.get("hierarchy") if isinstance(payload.get("hierarchy"), dict) else {}
    existing_global = chunks_at_level(payload, "global")
    if (
        existing_global
        and not force
        and hierarchy.get("strategy") == SUMMARY_STRATEGY
        and hierarchy.get("global_source_hash") == section_hash
    ):
        print(f"[skip] {video_path.name}: resume global a jour")
        return target

    combined = "\n\n".join(str(chunk.get("content") or "").strip() for chunk in sections)
    summary = luna_summary(
        combined,
        max_sentences=DEFAULT_GLOBAL_SENTENCES,
        max_chars=DEFAULT_GLOBAL_MAX_CHARS,
        model=model,
    )
    return apply_global_summary(
        video_path,
        payload,
        sections,
        summary,
        section_hash=section_hash,
        hierarchy=hierarchy,
    )
