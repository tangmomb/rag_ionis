import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import (
    existing_ocr_dir,
    existing_ocr_raw_dir,
    existing_transcripts_dir,
    existing_youtube_api_infos_path,
    relative_to_video_dir,
    speakers_dir,
)
from pipeline.steps.transcripts.artifacts import (
    LEGACY_TRANSCRIPT_2_NAMES,
    TRANSCRIPT_2_CORRECTED_NAME,
)


OCR_PROCESSED_NAME = "01_processed_ocr_items.json"
OCR_PROCESSED_CORRECTED_NAME = "corrected_ocr_items.json"
LEGACY_OCR_PROCESSED_CORRECTED_SUFFIX = "_ocr_processed_corrected.json"
SPEAKER_CANDIDATES_NAME = "speaker_candidates.json"
SPEAKER_PROPOSAL_VERSION = 7
MIN_RAW_OCR_TEXT_LENGTH = 7
OCR_LOWER_THIRD_MIN_TOP = 320
TRANSCRIPT_SPEAKER_PATTERN = re.compile(r"\bSPEAKER_\d+\b", re.IGNORECASE)
SPEAKER_INTRO_PATTERN = re.compile(r"je m'appelle\s+", re.IGNORECASE)
SPEAKER_JE_SUIS_PATTERN = re.compile(r"je suis\s+", re.IGNORECASE)
SPEAKER_MOI_CEST_PATTERN = re.compile(r"moi\s+c['’]est\s+", re.IGNORECASE)
SPEAKER_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’\-]+")
LOWERCASE_CONNECTORS = {"d", "d'", "d’", "de", "du", "des", "la", "le"}
SPEAKER_NAME_STOP_WORDS = {"je", "j'ai", "j’ai", "j", "moi"}
NON_PERSON_NAME_KEYWORDS = {
    "analyst",
    "bi",
    "buisness",
    "business",
    "ceo",
    "chief",
    "charge",
    "conseil",
    "crm",
    "diagnostic",
    "diagnostics",
    "directeur",
    "directrice",
    "ecole",
    "energie",
    "energy",
    "etudiant",
    "etudiante",
    "fondateur",
    "fondatrice",
    "management",
    "managemen",
    "manager",
    "managers",
    "autour",
    "par",
    "promo",
    "projet",
    "projets",
    "responsable",
    "responsables",
    "reunissent",
    "se",
    "sciences",
    "societe",
    "www",
}

def source_text_path(video_path, *, transcripts_dir_name=None):
    transcript_dir = existing_transcripts_dir(
        video_path,
        name=transcripts_dir_name,
    )
    corrected = transcript_dir / TRANSCRIPT_2_CORRECTED_NAME
    if corrected.exists():
        return corrected
    for name in LEGACY_TRANSCRIPT_2_NAMES:
        legacy_corrected = transcript_dir / name
        if legacy_corrected.exists():
            return legacy_corrected
    return None


def ocr_processed_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    candidates = (
        video_ocr_dir / OCR_PROCESSED_CORRECTED_NAME,
        video_ocr_dir / f"{video_path.stem}{LEGACY_OCR_PROCESSED_CORRECTED_SUFFIX}",
        video_ocr_dir / "ocr_processed_corrected.json",
        video_ocr_dir / OCR_PROCESSED_NAME,
    )
    return next((path for path in candidates if path.exists()), candidates[-1])


def candidates_path(video_path):
    return speakers_dir(video_path) / SPEAKER_CANDIDATES_NAME


def video_title(video_path):
    source = existing_youtube_api_infos_path(video_path)
    if source.exists():
        try:
            payload = read_json(source)
            title = payload.get("title") or payload.get("snippet", {}).get("title")
            if title:
                return str(title).strip()
        except (OSError, json.JSONDecodeError):
            pass
    return video_path.stem


def speaker_words(name):
    return [
        word
        for word in SPEAKER_WORD_PATTERN.findall(name)
        if any(char.isalpha() for char in word)
    ]


def normalize_speaker_text(text):
    return re.sub(r"\s+", " ", str(text)).strip(" ,.;:!?-–—")


def normalize_match_text(text):
    without_accents = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(char)
    )
    normalized = re.sub(r"[^0-9A-Za-z]+", " ", without_accents.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def is_capitalized_word(word):
    return bool(word) and word[0].isalpha() and word[0].isupper()


def titlecase_all_caps_name(name):
    words = speaker_words(name)
    if not words or any(any(char.islower() for char in word) for word in words):
        return name
    parts = []
    for part in re.split(r"(\s+|-|')", name):
        if not part or part.isspace() or part in {"-", "'"}:
            parts.append(part)
        elif any(char.isalpha() for char in part):
            parts.append(part[:1].upper() + part[1:].lower())
        else:
            parts.append(part)
    return "".join(parts)


def is_non_person_name(name):
    words = normalize_match_text(name).split()
    return any(keyword in words for keyword in NON_PERSON_NAME_KEYWORDS)


def is_probable_speaker_name(name):
    words = speaker_words(name)
    return bool(words) and any(is_capitalized_word(word) for word in words) and not is_non_person_name(name)


def has_multiple_speaker_words(name):
    return len(speaker_words(name)) >= 2


def add_candidate(candidates, name, method):
    name = titlecase_all_caps_name(normalize_speaker_text(name))
    if not is_probable_speaker_name(name):
        return
    key = normalize_match_text(name)
    if not key:
        return
    entry = candidates.setdefault(key, {"name": name, "methods": []})
    if method not in entry["methods"]:
        entry["methods"].append(method)


def extract_speaker_name(text, start_index):
    words = []
    text_after_intro = text[start_index:]
    phrase_end = re.search(r"[\n\r,.;:!?]", text_after_intro)
    if phrase_end:
        text_after_intro = text_after_intro[: phrase_end.start()]
    matches = list(SPEAKER_WORD_PATTERN.finditer(text_after_intro))
    index = 0
    while index < len(matches):
        word = matches[index].group(0)
        lower_word = word.lower()
        if lower_word in SPEAKER_NAME_STOP_WORDS:
            break
        if is_capitalized_word(word):
            words.append(word)
            index += 1
            continue
        if lower_word in LOWERCASE_CONNECTORS:
            next_index = index + 1
            if lower_word == "de" and next_index + 1 < len(matches):
                next_word = matches[next_index].group(0)
                if next_word.lower() == "la" and is_capitalized_word(matches[next_index + 1].group(0)):
                    words.extend([word, next_word])
                    index += 2
                    continue
            if next_index < len(matches) and is_capitalized_word(matches[next_index].group(0)):
                words.append(word)
                index += 1
                continue
        break
    return " ".join(words).strip(" ,.;:!?-–—")


def ocr_box_bounds(item):
    try:
        xs = [float(point[0]) for point in item.get("box", [])]
        ys = [float(point[1]) for point in item.get("box", [])]
    except (TypeError, ValueError, IndexError):
        return None
    return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None


def is_lower_third_ocr_item(item):
    bounds = ocr_box_bounds(item)
    return bounds is not None and bounds[1] >= OCR_LOWER_THIRD_MIN_TOP


def has_lower_third_companion(item, items):
    bounds = ocr_box_bounds(item)
    if bounds is None:
        return False
    left, _top, right, bottom = bounds
    width = max(right - left, 1)
    image = item.get("image")
    for other in items:
        other_kind = str(other.get("kind", "")).strip().lower()
        if other is item or other_kind not in {"lower_third", "others"}:
            continue
        if image and other.get("image") != image:
            continue
        other_bounds = ocr_box_bounds(other)
        if other_bounds is None:
            continue
        other_left, other_top, other_right, _other_bottom = other_bounds
        gap = other_top - bottom
        overlap = min(right, other_right) - max(left, other_left)
        left_delta = abs(other_left - left)
        if (
            0 <= gap <= 80
            and left_delta <= max(90, width * 0.25)
            and overlap >= min(width, other_right - other_left) * 0.4
        ):
            return True
    return False


def is_standalone_person_name_candidate(text):
    text = normalize_speaker_text(text)
    if not text or is_non_person_name(text) or re.search(r"[@:/\\0-9()\[\]]", text):
        return False
    words = speaker_words(text)
    if len(words) < 2 or len(words) > 4:
        return False
    if sum(word.lower() not in LOWERCASE_CONNECTORS for word in words) < 2:
        return False
    return all(word.lower() in LOWERCASE_CONNECTORS or is_capitalized_word(word) for word in words)


def load_ocr_speaker_candidates(video_path):
    path = ocr_processed_path(video_path)
    if not path.exists():
        return [], None
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] OCR processed illisible pour {video_path.stem}: {exc}")
        return [], path
    items = payload.get("items", [])
    names = []
    for item in items:
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"name", "others"}:
            continue
        if not is_lower_third_ocr_item(item) or not has_lower_third_companion(item, items):
            continue
        text = normalize_speaker_text(item.get("text", ""))
        if is_standalone_person_name_candidate(text):
            names.append(text)
    return names, path


def iter_rec_texts(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"rec_text", "rec_texts"}:
                texts = nested if isinstance(nested, list) else [nested]
                for text in texts:
                    if text is not None:
                        yield str(text)
            else:
                yield from iter_rec_texts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_rec_texts(nested)


def are_close_ocr_text_variants(shorter, longer):
    shorter_key = normalize_match_text(shorter)
    longer_key = normalize_match_text(longer)
    if not shorter_key or not longer_key:
        return False
    if shorter_key == longer_key:
        return True
    if len(shorter_key) < 24 and len(shorter_key.split()) < 4:
        return False
    if len(shorter_key) / len(longer_key) < 0.72:
        return False
    return (
        f" {shorter_key} " in f" {longer_key} "
        or SequenceMatcher(None, shorter_key, longer_key).ratio() >= 0.92
    )


def collapse_close_ocr_texts(texts):
    ranked = sorted(
        enumerate(texts),
        key=lambda item: (
            -len(normalize_match_text(item[1])),
            -len(item[1]),
            item[0],
        ),
    )
    kept = []
    for index, text in ranked:
        if any(
            are_close_ocr_text_variants(text, kept_text)
            for _kept_index, kept_text in kept
        ):
            continue
        kept.append((index, text))
    return [text for _index, text in sorted(kept)]


def load_raw_ocr_texts(video_path):
    raw_dir = existing_ocr_raw_dir(video_path)
    if not raw_dir.is_dir():
        return [], []

    texts = []
    seen = set()
    sources = []
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[warn] OCR brut illisible pour {video_path.stem}: "
                f"{path.name}: {exc}"
            )
            continue
        sources.append(path)
        for raw_text in iter_rec_texts(payload):
            text = re.sub(r"\s+", " ", raw_text).strip()
            key = text.casefold()
            if len(text) < MIN_RAW_OCR_TEXT_LENGTH or key in seen:
                continue
            seen.add(key)
            texts.append(text)
    return collapse_close_ocr_texts(texts), sources


def transcript_speaker_count(text):
    return len(
        {
            match.group(0).upper()
            for match in TRANSCRIPT_SPEAKER_PATTERN.finditer(str(text))
        }
    )


def propose_speakers(text, ocr_names):
    candidates = {}
    transcript_detections = []
    for match in SPEAKER_INTRO_PATTERN.finditer(text):
        transcript_detections.append(
            (
                match.start(),
                extract_speaker_name(text, match.end()),
                "transcript_je_m_appelle",
            )
        )
    for match in SPEAKER_JE_SUIS_PATTERN.finditer(text):
        name = extract_speaker_name(text, match.end())
        if has_multiple_speaker_words(name):
            transcript_detections.append(
                (match.start(), name, "transcript_je_suis")
            )
    for match in SPEAKER_MOI_CEST_PATTERN.finditer(text):
        name = extract_speaker_name(text, match.end())
        if has_multiple_speaker_words(name):
            transcript_detections.append(
                (match.start(), name, "transcript_moi_c_est")
            )
    for _position, name, method in sorted(
        transcript_detections,
        key=lambda detection: detection[0],
    ):
        add_candidate(candidates, name, method)
    for name in ocr_names:
        add_candidate(candidates, name, "ocr_lower_third")
    values = list(candidates.values())
    return {
        "speakers": [entry["name"] for entry in values],
        "candidates": values,
    }


def propose_for_video(
    video_path,
    force=False,
    *,
    transcripts_dir_name=None,
):
    source = source_text_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    target = candidates_path(video_path)
    if source is None:
        print(f"[skip] transcript introuvable pour: {video_path.stem}")
        return None
    expected_source = relative_to_video_dir(source, video_path)
    ocr_names, ocr_source = load_ocr_speaker_candidates(video_path)
    raw_ocr_texts, raw_ocr_sources = load_raw_ocr_texts(video_path)
    dependencies = [source]
    if ocr_source is not None and ocr_source.exists():
        dependencies.append(ocr_source)
    dependencies.extend(raw_ocr_sources)
    if target.exists() and not force:
        try:
            existing_payload = read_json(target)
        except (OSError, json.JSONDecodeError):
            existing_payload = {}
        if (
            existing_payload.get("proposal_version") == SPEAKER_PROPOSAL_VERSION
            and existing_payload.get("source") == expected_source
            and target.stat().st_mtime
            >= max(dependency.stat().st_mtime for dependency in dependencies)
        ):
            print(f"[skip] {target.name} existe deja")
            return target
        print(
            f"[regen] {target.name}: source Whisper ou OCR plus recente "
            f"ou differente"
        )
    text = source.read_text(encoding="utf-8")
    expected_speaker_count = transcript_speaker_count(text)
    if not text.strip():
        payload = {
            "proposal_version": SPEAKER_PROPOSAL_VERSION,
            "status": "no_speech",
            "speakers": [],
            "candidates": [],
            "video_title": video_title(video_path),
            "expected_speaker_count": expected_speaker_count,
            "source": expected_source,
            "raw_ocr_texts": raw_ocr_texts,
            "raw_ocr_sources": [
                relative_to_video_dir(raw_source, video_path)
                for raw_source in raw_ocr_sources
            ],
            "ocr_source": (
                relative_to_video_dir(ocr_source, video_path)
                if ocr_source and ocr_source.exists()
                else None
            ),
            "reason": "transcript_empty",
        }
        write_json(target, payload)
        print(f"[skip] transcript vide: {source}; aucun speaker a proposer")
        return target
    payload = propose_speakers(text, ocr_names)
    payload.update(
        {
            "proposal_version": SPEAKER_PROPOSAL_VERSION,
            "video_title": video_title(video_path),
            "expected_speaker_count": expected_speaker_count,
            "source": expected_source,
            "raw_ocr_texts": raw_ocr_texts,
            "raw_ocr_sources": [
                relative_to_video_dir(raw_source, video_path)
                for raw_source in raw_ocr_sources
            ],
            "ocr_source": relative_to_video_dir(ocr_source, video_path) if ocr_source and ocr_source.exists() else None,
        }
    )
    write_json(target, payload)
    print(f"[ok] {target} ({len(payload['speakers'])} candidat(s))")
    return target
