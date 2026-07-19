import re
import unicodedata
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path

from pipeline.support.json_io import read_json
from pipeline.support.paths import (
    CANONICAL_TRANSCRIPTS_DIR_NAME,
    existing_speakers_dir,
    existing_transcripts_dir,
    speakers_dir,
    video_base_dir,
)


SPEAKER_CANDIDATES_NAME = "speaker_candidates.json"
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"
OBSOLETE_CORRECTIONS_NAME = "speaker_transcript_corrections.json"
OBSOLETE_SPEAKER_CORRECTED_SUFFIX = "_speaker_corrected.txt"
SPEAKER_LABEL_PATTERN = re.compile(r"\bSPEAKER_(\d+)\b")

def normalize_name(name):
    normalized = unicodedata.normalize("NFKD", str(name).strip().casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^0-9a-z]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def clean_name(name):
    return " ".join(str(name).split()).strip()


def load_json(path):
    return read_json(path)


def validated_path(video_path):
    return existing_speakers_dir(video_path) / SPEAKERS_VALIDATED_NAME


def candidates_path(video_path, validated_payload=None):
    relative_source = str((validated_payload or {}).get("source") or "").strip()
    if relative_source:
        source = video_base_dir(video_path) / Path(relative_source)
        if source.exists():
            return source
    return existing_speakers_dir(video_path) / SPEAKER_CANDIDATES_NAME


def remove_obsolete_corrections(video_path):
    paths = {
        existing_speakers_dir(video_path) / OBSOLETE_CORRECTIONS_NAME,
        speakers_dir(video_path) / OBSOLETE_CORRECTIONS_NAME,
    }
    for path in paths:
        if path.exists():
            path.unlink()


def candidate_entries(payload):
    entries = []
    for item in payload.get("candidates", []) or []:
        if not isinstance(item, dict):
            continue
        name = clean_name(item.get("name", ""))
        methods = [clean_name(method) for method in item.get("methods", []) or [] if clean_name(method)]
        if name:
            entries.append({"name": name, "methods": methods})
    if entries:
        return entries
    return [
        {"name": clean_name(name), "methods": []}
        for name in payload.get("speakers", []) or []
        if clean_name(name)
    ]


def validated_speakers(payload):
    speakers = []
    seen = set()
    for name in payload.get("speakers", []) or []:
        name = clean_name(name)
        key = normalize_name(name)
        if not name or not key or key in seen:
            continue
        seen.add(key)
        speakers.append(name)
    return speakers


def is_transcript_candidate(entry):
    methods = entry.get("methods", [])
    return not methods or any(str(method).startswith("transcript_") for method in methods)


def build_speaker_mappings(candidates, speakers):
    transcript_candidates = [entry for entry in candidates if is_transcript_candidate(entry)]
    available = list(range(len(transcript_candidates)))
    mappings = []

    for speaker in speakers:
        if not available:
            break
        speaker_key = normalize_name(speaker)
        best_index = max(
            available,
            key=lambda index: SequenceMatcher(
                None,
                normalize_name(transcript_candidates[index]["name"]),
                speaker_key,
            ).ratio(),
        )
        available.remove(best_index)
        candidate = transcript_candidates[best_index]
        source_name = candidate["name"]
        mappings.append(
            {
                "source_name": source_name,
                "validated_name": speaker,
                "methods": candidate.get("methods", []),
                "changed": normalize_name(source_name) != speaker_key,
            }
        )
    return mappings


def source_transcript_paths(video_path, candidates_payload):
    transcript_dir = existing_transcripts_dir(
        video_path,
        name=CANONICAL_TRANSCRIPTS_DIR_NAME,
    )
    sources = []
    relative_source = str(candidates_payload.get("source") or "").strip()
    if relative_source:
        candidate_source = video_base_dir(video_path) / Path(relative_source)
        if candidate_source.exists():
            sources.append(candidate_source)

    if transcript_dir.exists():
        for path in sorted(transcript_dir.glob("*.txt")):
            lower_name = path.name.lower()
            if lower_name.endswith(OBSOLETE_SPEAKER_CORRECTED_SUFFIX):
                continue
            if "corrected" in lower_name or "enriched" in lower_name or "enrichi" in lower_name:
                sources.append(path)

    unique = []
    seen = set()
    for path in sources:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def replacement_pattern(name):
    words = [re.escape(word) for word in str(name).split() if word]
    if not words:
        return None
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", re.IGNORECASE)


def replacement_names(mapping):
    source_words = mapping["source_name"].split()
    validated_words = mapping["validated_name"].split()
    if len(source_words) != len(validated_words):
        return [mapping["source_name"]]
    variants = {
        " ".join(words)
        for words in product(*zip(source_words, validated_words))
    }
    validated_key = normalize_name(mapping["validated_name"])
    return sorted(
        (name for name in variants if normalize_name(name) != validated_key),
        key=len,
        reverse=True,
    )


def apply_mappings(text, mappings):
    corrected = text
    counts = {}
    changed_mappings = sorted(
        (mapping for mapping in mappings if mapping.get("changed")),
        key=lambda mapping: len(mapping["source_name"]),
        reverse=True,
    )
    for mapping in changed_mappings:
        replacement_count = 0
        for source_name in replacement_names(mapping):
            pattern = replacement_pattern(source_name)
            if pattern is None:
                continue
            corrected, count = pattern.subn(mapping["validated_name"], corrected)
            replacement_count += count
        counts[mapping["source_name"]] = replacement_count
    return corrected, counts


def apply_speaker_labels(text, speakers):
    corrected = str(text)
    counts = {}
    if not speakers:
        return corrected, counts

    def replacement(match):
        label = match.group(0)
        index = int(match.group(1))
        if len(speakers) == 1:
            speaker = speakers[0]
        elif index < len(speakers):
            speaker = speakers[index]
        else:
            return label
        counts[label] = counts.get(label, 0) + 1
        return speaker

    return SPEAKER_LABEL_PATTERN.sub(replacement, corrected), counts


def correct_speaker_files(
    video_path,
    sources,
    force=False,
    *,
    replace_speaker_labels=False,
):
    remove_obsolete_corrections(video_path)
    validated_source = validated_path(video_path)
    if not validated_source.exists():
        print(f"[skip] speakers valides introuvables: {validated_source}")
        return None

    validated_payload = load_json(validated_source)
    candidate_source = candidates_path(video_path, validated_payload)
    if not candidate_source.exists():
        print(f"[skip] candidats speakers introuvables: {candidate_source}")
        return None

    candidates_payload = load_json(candidate_source)
    candidates = candidate_entries(candidates_payload)
    speakers = validated_speakers(validated_payload)
    mappings = build_speaker_mappings(candidates, speakers)
    sources = [Path(source) for source in sources if Path(source).exists()]
    if not sources:
        print(f"[skip] aucun transcript source trouve pour: {video_path.stem}")
        return None

    total_replacements = 0
    for source in sources:
        original = source.read_text(encoding="utf-8")
        corrected, counts = apply_mappings(original, mappings)
        label_counts = {}
        if replace_speaker_labels:
            corrected, label_counts = apply_speaker_labels(
                corrected,
                speakers,
            )
        if corrected != original:
            source.write_text(corrected, encoding="utf-8")
        replacement_count = sum(counts.values()) + sum(label_counts.values())
        total_replacements += replacement_count

    print(
        f"[ok] speakers appliques "
        f"({total_replacements} remplacement(s), {len(sources)} transcript(s))"
    )
    return total_replacements


def correct_speaker_transcripts(video_path, force=False, source_only=False):
    validated_source = validated_path(video_path)
    if not validated_source.exists():
        print(f"[skip] speakers valides introuvables: {validated_source}")
        return None
    validated_payload = load_json(validated_source)
    candidate_source = candidates_path(video_path, validated_payload)
    if not candidate_source.exists():
        print(f"[skip] candidats speakers introuvables: {candidate_source}")
        return None
    candidates_payload = load_json(candidate_source)
    if source_only:
        relative_source = str(candidates_payload.get("source") or "").strip()
        source = video_base_dir(video_path) / Path(relative_source) if relative_source else None
        sources = [source] if source is not None and source.exists() else []
    else:
        sources = source_transcript_paths(video_path, candidates_payload)
    return correct_speaker_files(video_path, sources, force=force)
