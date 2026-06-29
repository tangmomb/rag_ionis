import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
CHUNKS_SUFFIX = "_transcript_chunks.json"
EMBEDDINGS_SUFFIX = "_transcript_embeddings.json"
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
    return video_path.parent / "chunks" / f"{video_path.stem}{CHUNKS_SUFFIX}"


def embeddings_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{EMBEDDINGS_SUFFIX}"


def load_chunks(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("meta_data", {}), payload.get("chunks", [])


def create_embeddings(client, model, video_path, force=False):
    source = chunks_path(video_path)
    target = embeddings_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] chunks introuvables: {source}")
        return None

    source_meta_data, chunks = load_chunks(source)
    if not chunks:
        print(f"[skip] aucun chunk dans: {source}")
        return None

    embeddings = []
    for chunk in chunks:
        text = str(chunk.get("content", "")).strip()
        if not text:
            continue
        response = client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        embeddings.append(
            {
                "chunk_index": chunk.get("chunk_index"),
                "char_count": chunk.get("char_count"),
                "meta_data": chunk.get("meta_data", {}),
                "content": text,
                "embedding": embedding,
            }
        )
        print(f"[embed] {video_path.name} chunk {chunk.get('chunk_index')}/{len(chunks)}", flush=True)

    payload = {
        "model": model,
        "source": str(source.relative_to(video_path.parent)),
        "chunk_count": len(embeddings),
        "meta_data": source_meta_data,
        "embeddings": embeddings,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] {target} ({len(embeddings)} embeddings)")
    return target


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

    print(f"{done} fichiers embeddings crees.")


if __name__ == "__main__":
    main()
