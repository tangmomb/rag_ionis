import json
import re
from pathlib import Path

from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import (
    chunks_dir,
    existing_speakers_dir,
    existing_transcripts_dir,
    output_is_current,
    relative_to_video_dir,
)
from pipeline.steps.transcripts.artifacts import (
    LEGACY_TRANSCRIPT_PLAIN_NAMES,
    TRANSCRIPT_PLAIN_NAME,
)


PLAIN_NAME = TRANSCRIPT_PLAIN_NAME
LEGACY_PLAIN_SUFFIX = "_transcript.txt"
OCR_SUBTITLE_NAME = "ocr_subtitles.txt"
LEGACY_OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"
CHUNKS_NAME = "transcript_chunks.json"
OBSOLETE_VALIDATED_CHUNKS_NAME = "transcript_chunks_speaker_validated.json"
DEFAULT_MAX_CHARS = 1000
ALERT_WORD_THRESHOLD = 3000
CHUNK_PROFILES = {"short", "long"}

def transcript_path(video_path, *, transcripts_dir_name=None):
    transcript_dir = existing_transcripts_dir(
        video_path,
        name=transcripts_dir_name,
    )
    preferred = transcript_dir / PLAIN_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_PLAIN_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    for name in LEGACY_TRANSCRIPT_PLAIN_NAMES:
        legacy_current = transcript_dir / name
        if legacy_current.exists() and not preferred.exists():
            return legacy_current
    return preferred


def ocr_subtitle_path(video_path, *, transcripts_dir_name=None):
    transcript_dir = existing_transcripts_dir(
        video_path,
        name=transcripts_dir_name,
    )
    preferred = transcript_dir / OCR_SUBTITLE_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_OCR_SUBTITLE_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def source_text_path(video_path, *, transcripts_dir_name=None):
    transcript = transcript_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    if transcript.exists():
        return transcript
    ocr_subtitle = ocr_subtitle_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
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
        payload = read_json(source)
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


def output_matches_profile(target, profile):
    try:
        payload = read_json(target)
    except (OSError, json.JSONDecodeError):
        return False
    chunking = payload.get("chunking") if isinstance(payload, dict) else None
    existing_profile = chunking.get("profile") if isinstance(chunking, dict) else None
    if existing_profile is None:
        existing_profile = "short"
    return existing_profile == profile


def create_chunks(
    video_path,
    force=False,
    profile="short",
    *,
    transcripts_dir_name=None,
):
    if profile not in CHUNK_PROFILES:
        raise ValueError(f"Profil de chunks invalide: {profile!r}")
    target = chunks_path(video_path)
    source = source_text_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
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
    write_json(target, payload)
    obsolete = target.parent / OBSOLETE_VALIDATED_CHUNKS_NAME
    if force and obsolete.exists():
        obsolete.unlink()
    print(f"[ok] {target} ({len(payload['chunks'])} chunks, {len(speakers)} speaker(s))")
    return target
