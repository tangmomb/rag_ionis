import argparse
import json
import re
import sys
from pathlib import Path

from pipeline.support.analysis import update_analysed_infos
from pipeline.support.paths import (
    chunks_dir,
    existing_speakers_dir,
    existing_transcripts_dir,
    relative_to_video_dir,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
PLAIN_NAME = "plain_transcript.txt"
LEGACY_PLAIN_SUFFIX = "_transcript.txt"
OCR_SUBTITLE_NAME = "ocr_subtitles.txt"
LEGACY_OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"
CHUNKS_NAME = "transcript_chunks.json"
OBSOLETE_VALIDATED_CHUNKS_NAME = "transcript_chunks_speaker_validated.json"
DEFAULT_MAX_CHARS = 1000
ALERT_WORD_THRESHOLD = 3000
CHUNK_PROFILES = {"short", "long"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def video_files(video_dir):
    direct_videos = [
        path
        for path in sorted(video_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
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


def transcript_path(video_path):
    transcript_dir = existing_transcripts_dir(video_path)
    preferred = transcript_dir / PLAIN_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_PLAIN_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def ocr_subtitle_path(video_path):
    transcript_dir = existing_transcripts_dir(video_path)
    preferred = transcript_dir / OCR_SUBTITLE_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_OCR_SUBTITLE_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def source_text_path(video_path):
    transcript = transcript_path(video_path)
    if transcript.exists():
        return transcript
    ocr_subtitle = ocr_subtitle_path(video_path)
    return ocr_subtitle if ocr_subtitle.exists() else None


def validated_speakers_path(video_path):
    return existing_speakers_dir(video_path) / SPEAKERS_VALIDATED_NAME


def chunks_path(video_path):
    return chunks_dir(video_path) / CHUNKS_NAME


def load_validated_speakers(video_path):
    source = validated_speakers_path(video_path)
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] JSON speakers invalide pour {video_path.stem}: {exc}")
        return None
    speakers = payload.get("speakers")
    if not isinstance(speakers, list):
        return None
    return [" ".join(str(name).split()).strip() for name in speakers if str(name).strip()]


def word_count(text):
    return len([word for word in str(text).split() if word.strip()])


def normalize_text(text):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines).strip()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [part.strip() for part in parts if part.strip()]


def split_into_chunks(text, max_chars=DEFAULT_MAX_CHARS):
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        for sentence in split_sentences(normalize_text(paragraph)):
            projected = len(current) + len(sentence) + (1 if current else 0)
            if current and projected > max_chars:
                chunks.append(current.strip())
                current = ""
            current = sentence if not current else f"{current} {sentence}"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunks_payload(text, speakers, profile="short"):
    if profile not in CHUNK_PROFILES:
        raise ValueError(f"Profil de chunks invalide: {profile!r}")
    chunks = split_into_chunks(text)
    return {
        "chunking": {
            "profile": profile,
            "max_chars": DEFAULT_MAX_CHARS,
            "cut_policy": "cut_at_next_sentence_after_threshold",
        },
        "chunks": [
            {
                "chunk_index": index + 1,
                "chunk_level": "detail",
                "chunk_parent_id": None,
                "meta_data": {"speakers": speakers},
                "content": chunk,
                "alert": word_count(chunk) > ALERT_WORD_THRESHOLD,
                "alert_reason": "over_3000_words" if word_count(chunk) > ALERT_WORD_THRESHOLD else None,
                "char_count": len(chunk),
            }
            for index, chunk in enumerate(chunks)
        ],
    }


def output_is_current(target, dependencies):
    if not target.exists():
        return False
    target_mtime = target.stat().st_mtime
    return all(not dependency.exists() or dependency.stat().st_mtime <= target_mtime for dependency in dependencies)


def output_matches_profile(target, profile):
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    chunking = payload.get("chunking") if isinstance(payload, dict) else None
    existing_profile = chunking.get("profile") if isinstance(chunking, dict) else None
    if existing_profile is None:
        existing_profile = "short"
    return existing_profile == profile


def create_chunks(video_path, force=False, profile="short"):
    if profile not in CHUNK_PROFILES:
        raise ValueError(f"Profil de chunks invalide: {profile!r}")
    target = chunks_path(video_path)
    source = source_text_path(video_path)
    speakers_source = validated_speakers_path(video_path)
    if source is None:
        print(f"[skip] transcript introuvable pour: {video_path.stem}")
        return None
    speakers = load_validated_speakers(video_path)
    if speakers is None:
        print(f"[skip] speakers valides introuvables ou invalides: {speakers_source}")
        return None
    if (
        target.exists()
        and not force
        and output_is_current(target, (source, speakers_source))
        and output_matches_profile(target, profile)
    ):
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: profil, transcript ou speakers plus recents")
    normalized = normalize_text(source.read_text(encoding="utf-8"))
    if not normalized:
        print(f"[skip] transcript vide: {source}")
        return None
    payload = build_chunks_payload(normalized, speakers, profile=profile)
    payload["source"] = relative_to_video_dir(source, video_path)
    payload["speakers_source"] = relative_to_video_dir(speakers_source, video_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    obsolete = target.parent / OBSOLETE_VALIDATED_CHUNKS_NAME
    if force and obsolete.exists():
        obsolete.unlink()
    update_analysed_infos(
        video_path,
        "create_chunks",
        {
            "status": "done",
            "source": relative_to_video_dir(source, video_path),
            "speakers_source": relative_to_video_dir(speakers_source, video_path),
            "chunks_file": relative_to_video_dir(target, video_path),
            "chunk_count": len(payload["chunks"]),
            "speakers": speakers,
        },
    )
    print(f"[ok] {target} ({len(payload['chunks'])} chunks, {len(speakers)} speaker(s))")
    return target


def parse_args():
    parser = argparse.ArgumentParser(description="Cree les chunks avec les speakers deja valides.")
    parser.add_argument("--video-dir", help="Dossier contenant les videos.")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(CHUNK_PROFILES)),
        default="short",
        help="Profil de chunking: short ou long. Le profil long est ensuite hierarchise.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    done = sum(
        bool(create_chunks(video_path, force=args.force, profile=args.profile))
        for video_path in videos
    )
    print(f"{done} transcripts chunkes.")


if __name__ == "__main__":
    main()
