from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import chunks_dir, existing_chunks_dir


CHUNKS_NAME = "transcript_chunks.json"
SUMMARY_STRATEGY = "extractive_frequency_v1"
DEFAULT_DETAILS_PER_SECTION = 6
DEFAULT_SECTION_SENTENCES = 4
DEFAULT_SECTION_MAX_CHARS = 1200
DEFAULT_GLOBAL_SENTENCES = 6
DEFAULT_GLOBAL_MAX_CHARS = 1800
CHUNK_LEVEL_ORDER = {"global": 0, "section": 1, "detail": 2}

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


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def sentence_words(sentence: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'-]+", sentence.casefold())
        if len(word) >= 3 and word not in FRENCH_STOP_WORDS
    ]


def _truncate_summary(sentences: Iterable[str], max_chars: int) -> str:
    selected: list[str] = []
    current_length = 0
    for sentence in sentences:
        projected = current_length + len(sentence) + (1 if selected else 0)
        if selected and projected > max_chars:
            break
        if not selected and len(sentence) > max_chars:
            return sentence[: max_chars - 1].rstrip() + "…"
        selected.append(sentence)
        current_length = projected
    return " ".join(selected).strip()


def extractive_summary(text: str, *, max_sentences: int, max_chars: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return _truncate_summary(sentences, max_chars)

    frequencies = Counter(
        word
        for sentence in sentences
        for word in sentence_words(sentence)
    )
    if not frequencies:
        return _truncate_summary(sentences[:max_sentences], max_chars)

    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        words = sentence_words(sentence)
        if not words:
            score = 0.0
        else:
            score = sum(frequencies[word] for word in words) / math.sqrt(len(words))
        scored.append((score, index, sentence))

    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    chronological = [sentence for _score, _index, sentence in sorted(top, key=lambda item: item[1])]
    return _truncate_summary(chronological, max_chars)


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


def _speakers_for_chunks(chunks: Iterable[dict[str, Any]]) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        meta_data = chunk.get("meta_data")
        raw_speakers = meta_data.get("speakers", []) if isinstance(meta_data, dict) else []
        if not isinstance(raw_speakers, list):
            continue
        for speaker in raw_speakers:
            normalized = " ".join(str(speaker).split()).strip()
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                speakers.append(normalized)
    return speakers


def _section_groups(
    details: list[dict[str, Any]],
    details_per_section: int,
) -> list[list[dict[str, Any]]]:
    return [
        details[index : index + details_per_section]
        for index in range(0, len(details), details_per_section)
    ]


def summarize_sections(
    video_path: Path,
    *,
    force: bool = False,
    details_per_section: int = DEFAULT_DETAILS_PER_SECTION,
) -> Path | None:
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
        summary = extractive_summary(
            combined,
            max_sentences=DEFAULT_SECTION_SENTENCES,
            max_chars=DEFAULT_SECTION_MAX_CHARS,
        )
        speakers = _speakers_for_chunks(group)
        sections.append(
            {
                "chunk_index": section_index,
                "chunk_level": "section",
                "chunk_parent_id": None,
                "chunk_parent": None,
                "meta_data": {
                    "speakers": speakers,
                    "summary_strategy": SUMMARY_STRATEGY,
                    "detail_chunk_indexes": [chunk.get("chunk_index") for chunk in group],
                },
                "content": summary,
                "char_count": len(summary),
            }
        )
        for detail in group:
            updated = dict(detail)
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


def summarize_video(video_path: Path, *, force: bool = False) -> Path | None:
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
        and hierarchy.get("global_source_hash") == section_hash
    ):
        print(f"[skip] {video_path.name}: resume global a jour")
        return target

    combined = "\n\n".join(str(chunk.get("content") or "").strip() for chunk in sections)
    summary = extractive_summary(
        combined,
        max_sentences=DEFAULT_GLOBAL_SENTENCES,
        max_chars=DEFAULT_GLOBAL_MAX_CHARS,
    )
    global_chunk = {
        "chunk_index": 1,
        "chunk_level": "global",
        "chunk_parent_id": None,
        "chunk_parent": None,
        "meta_data": {
            "speakers": _speakers_for_chunks(sections),
            "summary_strategy": SUMMARY_STRATEGY,
            "section_chunk_indexes": [chunk.get("chunk_index") for chunk in sections],
        },
        "content": summary,
        "char_count": len(summary),
    }

    updated_sections: list[dict[str, Any]] = []
    for section in sections:
        updated = dict(section)
        updated["chunk_parent_id"] = None
        updated["chunk_parent"] = {
            "chunk_level": "global",
            "chunk_index": 1,
        }
        updated_sections.append(updated)

    details = chunks_at_level(payload, "detail")
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
