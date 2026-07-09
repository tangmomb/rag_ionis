import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from common.pipeline_paths import chunks_dir, existing_chunks_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
CHUNKS_NAME = "transcript_chunks.json"
CHUNKS_SPEAKER_VALIDATED_NAME = "transcript_chunks_speaker_validated.json"
LEGACY_CHUNKS_SUFFIX = "_chunks.json"
LEGACY_CHUNKS_CORRECTED_SUFFIX = "_chunks_corrected.json"
EMBEDDING_SUFFIX = "_embedding.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def video_files(video_dir):
    direct_videos = []
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            direct_videos.append(path)

    if direct_videos:
        yield from direct_videos
        return

    for child in sorted(video_dir.iterdir()):
        if not child.is_dir():
            continue
        for path in sorted(child.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


def latest_video_dir(parent_dir):
    candidates = sorted(path for path in parent_dir.iterdir() if path.is_dir() and any(video_files(path)))
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


def chunks_path(video_path):
    video_chunks_dir = existing_chunks_dir(video_path)
    candidates = (
        video_chunks_dir / CHUNKS_SPEAKER_VALIDATED_NAME,
        video_chunks_dir / f"{video_path.stem}{LEGACY_CHUNKS_CORRECTED_SUFFIX}",
        video_chunks_dir / CHUNKS_NAME,
        video_chunks_dir / f"{video_path.stem}{LEGACY_CHUNKS_SUFFIX}",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return video_chunks_dir / CHUNKS_NAME


def embedding_path(video_path, chunk_index):
    chunk_label = str(chunk_index).replace(".", "_")
    if chunk_label.isdigit():
        chunk_label = f"{int(chunk_label):02d}"
    return chunks_dir(video_path) / f"chunk_{chunk_label}{EMBEDDING_SUFFIX}"


def load_chunks(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("meta_data", {}), payload.get("chunks", [])


def create_embeddings(client, model, video_path, force=False):
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
        target = embedding_path(video_path, chunk_index)
        if target.exists() and not force:
            print(f"[skip] {target.name} existe deja")
            continue
        response = client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        payload = {
            "model": model,
            "source": relative_to_video_dir(source, video_path),
            "chunk_count": len(chunks),
            "chunk_index": chunk_index,
            "char_count": chunk.get("char_count"),
            "meta_data": {
                **source_meta_data,
                **chunk.get("meta_data", {}),
            },
            "content": text,
            "embedding": embedding,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
        print(f"[embed] {video_path.name} chunk {chunk_index}/{len(chunks)} -> {target.name}", flush=True)

    if not written:
        print(f"[skip] aucun nouvel embedding a ecrire pour {video_path.name}")
        return None

    print(f"[ok] {video_path.name} ({written} fichiers embeddings)")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cree des embeddings a partir des chunks de transcript."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier contenant les videos. Defaut: dernier sous-dossier de downloads/youtube",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent utilise si --video-dir est absent. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Modele d'embedding. Defaut: text-embedding-3-large",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere les embeddings meme s'ils existent deja.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    client = OpenAI()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"Modele: {args.model}", flush=True)
    done = 0
    for video_path in videos:
        if create_embeddings(client, args.model, video_path, force=args.force):
            done += 1

    print(f"{done} videos traitees pour les embeddings.")


if __name__ == "__main__":
    main()
