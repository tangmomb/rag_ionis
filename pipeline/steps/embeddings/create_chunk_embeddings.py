import json
from pathlib import Path

from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import chunks_dir, existing_chunks_dir, relative_to_video_dir


CHUNKS_NAME = "transcript_chunks.json"
LEGACY_CHUNKS_SUFFIX = "_chunks.json"
EMBEDDING_SUFFIX = "_embedding.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 2000
DEFAULT_CHUNK_LEVEL = "detail"
CHUNK_LEVELS = {"global", "section", "detail"}

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
            print(f"[skip] {target.name} existe deja")
            continue
        if target.exists() and not force:
            print(f"[regen] {target.name}: contenu, modele ou dimensions obsoletes")
        response = client.embeddings.create(model=model, dimensions=dimensions, input=text)
        embedding = response.data[0].embedding
        payload = {
            "model": model,
            "dimensions": dimensions,
            "source": relative_to_video_dir(source, video_path),
            "chunk_count": len(chunks),
            "chunk_index": chunk_index,
            "chunk_level": chunk_level,
            "chunk_parent_id": chunk.get("chunk_parent_id"),
            "chunk_parent": chunk.get("chunk_parent"),
            "char_count": chunk.get("char_count"),
            "meta_data": {
                **source_meta_data,
                **chunk.get("meta_data", {}),
            },
            "content": text,
            "embedding": embedding,
        }
        write_json(target, payload, indent=None)
        written += 1
        print(f"[embed] {video_path.name} chunk {chunk_index}/{len(chunks)} -> {target.name}", flush=True)

    if not written:
        print(f"[skip] aucun nouvel embedding a ecrire pour {video_path.name}")
        return None

    print(f"[ok] {video_path.name} ({written} fichiers embeddings)")
    return True
