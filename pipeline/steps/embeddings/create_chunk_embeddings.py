import json
from pathlib import Path

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
from pipeline.support.paths import chunks_dir, existing_chunks_dir, relative_to_video_dir


CHUNKS_NAME = "transcript_chunks.json"
LEGACY_CHUNKS_SUFFIX = "_chunks.json"
EMBEDDING_SUFFIX = "_embedding.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 2000
DEFAULT_CHUNK_LEVEL = "detail"
CHUNK_LEVELS = {"global", "section", "detail"}
EMBEDDING_BATCH_STATE_NAME = "embedding_batch_state.json"
EMBEDDING_BATCH_INPUT_NAME = "embedding_batch_input.jsonl"
EMBEDDING_BATCH_OUTPUT_NAME = "embedding_batch_output.jsonl"
EMBEDDING_BATCH_ERROR_NAME = "embedding_batch_error.jsonl"

def chunks_path(video_path):
    video_chunks_dir = existing_chunks_dir(video_path)
    candidates = (
        video_chunks_dir / CHUNKS_NAME,
        video_chunks_dir / f"{video_path.stem}{LEGACY_CHUNKS_SUFFIX}",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return video_chunks_dir / CHUNKS_NAME


def embedding_path(video_path, chunk_index, chunk_level=DEFAULT_CHUNK_LEVEL):
    chunk_label = str(chunk_index).replace(".", "_")
    if chunk_label.isdigit():
        chunk_label = f"{int(chunk_label):02d}"
    normalized_level = str(chunk_level or DEFAULT_CHUNK_LEVEL).strip().lower()
    if normalized_level not in CHUNK_LEVELS:
        raise ValueError(f"Niveau de chunk invalide: {normalized_level!r}")
    level_label = "" if normalized_level == DEFAULT_CHUNK_LEVEL else f"{normalized_level}_"
    return chunks_dir(video_path) / f"chunk_{level_label}{chunk_label}{EMBEDDING_SUFFIX}"


def load_chunks(path):
    payload = read_json(path)
    return payload.get("meta_data", {}), payload.get("chunks", [])


def existing_embedding_matches(path, model, dimensions, expected_text=None):
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    embedding = payload.get("embedding") if isinstance(payload, dict) else None
    return (
        payload.get("model") == model
        and isinstance(embedding, list)
        and len(embedding) == dimensions
        and (expected_text is None or payload.get("content") == expected_text)
    )


def remove_speakers_from_embedding(path):
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    changed = "speakers" in payload
    payload.pop("speakers", None)
    meta_data = payload.get("meta_data")
    if isinstance(meta_data, dict) and "speakers" in meta_data:
        meta_data = dict(meta_data)
        meta_data.pop("speakers", None)
        if meta_data:
            payload["meta_data"] = meta_data
        else:
            payload.pop("meta_data", None)
        changed = True
    if changed:
        write_json(path, payload, indent=None)
    return changed


def embedding_meta_data(source_meta_data, chunk):
    combined = {
        **(source_meta_data if isinstance(source_meta_data, dict) else {}),
        **(chunk.get("meta_data", {}) if isinstance(chunk.get("meta_data"), dict) else {}),
    }
    combined.pop("speakers", None)
    return combined


def embedding_payload(
    source_meta_data,
    chunk,
    *,
    source,
    video_path,
    chunk_count,
    model,
    dimensions,
    embedding,
):
    text = str(chunk.get("content", "")).strip()
    payload = {
        "model": model,
        "dimensions": dimensions,
        "source": relative_to_video_dir(source, video_path),
        "chunk_count": chunk_count,
        "chunk_index": chunk.get("chunk_index"),
        "chunk_level": str(
            chunk.get("chunk_level") or DEFAULT_CHUNK_LEVEL
        ).strip().lower(),
        "chunk_parent_id": chunk.get("chunk_parent_id"),
        "chunk_parent": chunk.get("chunk_parent"),
        "char_count": chunk.get("char_count"),
        "content": text,
        "embedding": embedding,
    }
    meta_data = embedding_meta_data(source_meta_data, chunk)
    if meta_data:
        payload["meta_data"] = meta_data
    return payload


def create_embeddings(client, model, dimensions, video_path, force=False):
    source = chunks_path(video_path)
    if not source.exists():
        print(f"[skip] chunks introuvables: {source}")
        return None

    source_meta_data, chunks = load_chunks(source)
    if not chunks:
        print(f"[skip] aucun chunk dans: {source}")
        return None

    target_dir = chunks_dir(video_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for chunk in chunks:
        text = str(chunk.get("content", "")).strip()
        if not text:
            continue
        chunk_index = chunk.get("chunk_index")
        chunk_level = str(chunk.get("chunk_level") or DEFAULT_CHUNK_LEVEL).strip().lower()
        target = embedding_path(video_path, chunk_index, chunk_level)
        if target.exists() and not force and existing_embedding_matches(
            target,
            model,
            dimensions,
            expected_text=text,
        ):
            if remove_speakers_from_embedding(target):
                print(f"[clean] speakers retires de {target.name}")
            print(f"[skip] {target.name} existe deja")
            continue
        if target.exists() and not force:
            print(f"[regen] {target.name}: contenu, modele ou dimensions obsoletes")
        response = client.embeddings.create(model=model, dimensions=dimensions, input=text)
        embedding = response.data[0].embedding
        payload = embedding_payload(
            source_meta_data,
            chunk,
            source=source,
            video_path=video_path,
            chunk_count=len(chunks),
            model=model,
            dimensions=dimensions,
            embedding=embedding,
        )
        write_json(target, payload, indent=None)
        written += 1
        print(f"[embed] {video_path.name} chunk {chunk_index}/{len(chunks)} -> {target.name}", flush=True)

    if not written:
        print(f"[skip] aucun nouvel embedding a ecrire pour {video_path.name}")
        return None

    print(f"[ok] {video_path.name} ({written} fichiers embeddings)")
    return True


def create_embeddings_batch(
    model,
    dimensions,
    video_path,
    *,
    force=False,
    wait=True,
    poll_interval_seconds=30,
    reset_batch=None,
):
    from openai import OpenAI

    source = chunks_path(video_path)
    if not source.exists():
        print(f"[skip] chunks introuvables: {source}")
        return None
    source_meta_data, chunks = load_chunks(source)
    if not chunks:
        print(f"[skip] aucun chunk dans: {source}")
        return None

    target_dir = chunks_dir(video_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ordinal, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("content", "")).strip()
        if not text:
            continue
        chunk_index = chunk.get("chunk_index")
        chunk_level = str(
            chunk.get("chunk_level") or DEFAULT_CHUNK_LEVEL
        ).strip().lower()
        target = embedding_path(video_path, chunk_index, chunk_level)
        if target.exists() and not force and existing_embedding_matches(
            target,
            model,
            dimensions,
            expected_text=text,
        ):
            if remove_speakers_from_embedding(target):
                print(f"[clean] speakers retires de {target.name}")
            print(f"[skip] {target.name} existe deja")
            continue
        jobs.append(
            {
                "custom_id": f"embedding-{ordinal:06d}",
                "chunk": chunk,
                "target": target,
                "body": {
                    "model": model,
                    "dimensions": dimensions,
                    "input": text,
                },
            }
        )

    if not jobs:
        print(f"[skip] aucun nouvel embedding a ecrire pour {video_path.name}")
        return None

    state_path = target_dir / EMBEDDING_BATCH_STATE_NAME
    input_path = target_dir / EMBEDDING_BATCH_INPUT_NAME
    output_path = target_dir / EMBEDDING_BATCH_OUTPUT_NAME
    error_path = target_dir / EMBEDDING_BATCH_ERROR_NAME
    if reset_batch is None:
        reset_batch = force
    if reset_batch:
        for path in (state_path, input_path, output_path, error_path):
            path.unlink(missing_ok=True)

    request_fingerprint = batch_request_fingerprint(
        model,
        sources=(source,),
        custom_ids=(job["custom_id"] for job in jobs),
        options={
            "workflow": "chunk_embeddings",
            "dimensions": dimensions,
            "requests": [
                {
                    "custom_id": job["custom_id"],
                    "body": job["body"],
                }
                for job in jobs
            ],
        },
    )
    state = load_batch_state(state_path)
    if state is not None and not batch_state_matches(state, request_fingerprint):
        print(
            "[batch] etat embeddings incompatible; nouvelle soumission "
            f"(ancien batch_id={state.get('batch_id', 'inconnu')})",
            flush=True,
        )
        state = None
    if state is None:
        write_jsonl(
            input_path,
            [
                {
                    "custom_id": job["custom_id"],
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": job["body"],
                }
                for job in jobs
            ],
        )
        client = OpenAI()
        with input_path.open("rb") as batch_file:
            uploaded = client.files.create(file=batch_file, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={
                "workflow": "chunk_embeddings",
                "model": model,
                "video": Path(video_path).name,
            },
        )
        state = {
            "mode": "batch",
            "model": model,
            "dimensions": dimensions,
            "batch_id": batch.id,
            "status": batch.status,
            "input_file_id": uploaded.id,
            "submitted_count": len(jobs),
            "request_fingerprint": request_fingerprint,
        }
        save_batch_state(state_path, state)
        print(
            f"[batch] embeddings submitted id={batch.id} "
            f"status={batch.status} requests={len(jobs)}",
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
            f"[batch] embeddings status={current['status']} "
            f"batch_id={current['batch_id']} attente {seconds}s",
            flush=True,
        ),
    )
    if not is_terminal_batch_status(state["status"]):
        return state_path
    if state["status"] != COMPLETED_BATCH_STATUS:
        raise RuntimeError(
            f"Batch embeddings termine avec statut non supporte: {state['status']}"
        )
    if not state.get("output_file_id"):
        raise RuntimeError("Batch embeddings complete mais output_file_id absent.")

    download_batch_files(client, state, output_path, error_path)
    records = records_by_custom_id(parse_jsonl(output_path))
    missing_ids = [
        job["custom_id"] for job in jobs if job["custom_id"] not in records
    ]
    if missing_ids:
        raise RuntimeError(
            "Resultats batch embeddings incomplets; custom_id manquant(s): "
            + ", ".join(missing_ids)
        )

    for index, job in enumerate(jobs, start=1):
        response = records[job["custom_id"]].get("response") or {}
        if response.get("status_code") != 200:
            raise RuntimeError(
                f"Batch {job['custom_id']} en echec avec "
                f"status={response.get('status_code')}"
            )
        data = (response.get("body") or {}).get("data") or []
        if not data or not isinstance(data[0].get("embedding"), list):
            raise RuntimeError(
                f"Embedding batch invalide pour {job['custom_id']}."
            )
        embedding = data[0]["embedding"]
        if len(embedding) != dimensions:
            raise RuntimeError(
                f"Dimensions inattendues pour {job['custom_id']}: "
                f"{len(embedding)}, attendu {dimensions}."
            )
        payload = embedding_payload(
            source_meta_data,
            job["chunk"],
            source=source,
            video_path=video_path,
            chunk_count=len(chunks),
            model=model,
            dimensions=dimensions,
            embedding=embedding,
        )
        write_json(job["target"], payload, indent=None)
        print(
            f"[embed-batch {index}/{len(jobs)}] -> {job['target'].name}",
            flush=True,
        )

    print(f"[ok] {video_path.name} ({len(jobs)} fichiers embeddings)")
    return True
