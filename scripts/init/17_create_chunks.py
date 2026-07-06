import argparse
import os
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

from analysed_infos import update_analysed_infos

DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
PLAIN_SUFFIX = "_transcript.txt"
OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
OCR_PROCESSED_SUFFIX = "_ocr_processed.json"
OCR_PROCESSED_CORRECTED_SUFFIX = "_ocr_processed_corrected.json"
CHUNKS_SUFFIX = "_chunks.json"
DEFAULT_MAX_CHARS = 1000
ALERT_WORD_THRESHOLD = 3000
API = "https://www.googleapis.com/youtube/v3"
DEFAULT_YOUTUBE_API_SLEEP_SECONDS = 0.5
OCR_LOWER_THIRD_MIN_TOP = 320
SPEAKER_INTRO_PATTERN = re.compile(r"je m'appelle\s+", re.IGNORECASE)
SPEAKER_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+")
SPACY_FRENCH_MODEL = os.environ.get("SPACY_FRENCH_MODEL", "fr_dep_news_trf")
SPACY_REQUIRE_GPU = os.environ.get("SPACY_REQUIRE_GPU", "0").strip().lower() not in {"0", "false", "no"}
SPACY_PERSON_LABELS = {"PER", "PERSON"}
SPACY_ORG_LABELS = {"ORG"}

LOWERCASE_CONNECTORS = {"d", "d'", "d’", "de", "du", "des", "la", "le"}
SPEAKER_NAME_STOP_WORDS = {"je", "j'ai", "j’ai", "j", "moi"}
NON_PERSON_NAME_KEYWORDS = {
    "analyst",
    "bi",
    "buisness",
    "business",
    "ceo",
    "chief",
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
    "promo",
    "sciences",
    "societe",
    "www",
}
_SPACY_NLP = None
_SPACY_LOAD_ATTEMPTED = False
_SPACY_WARNING_SHOWN = False
YOUTUBE_API_INFOS_SUFFIX = ".youtube_api_infos.json"
LEGACY_INFO_SUFFIX = ".info.json"

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


def transcript_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{PLAIN_SUFFIX}"


def ocr_subtitle_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{OCR_SUBTITLE_SUFFIX}"


def ocr_processed_path(video_path):
    transcript_dir = video_path.parent / "transcript"
    corrected = transcript_dir / f"{video_path.stem}{OCR_PROCESSED_CORRECTED_SUFFIX}"
    if corrected.exists():
        return corrected
    return transcript_dir / f"{video_path.stem}{OCR_PROCESSED_SUFFIX}"


def chunks_path(video_path):
    return video_path.parent / "chunks" / f"{video_path.stem}{CHUNKS_SUFFIX}"


def source_text_path(video_path):
    transcript = transcript_path(video_path)
    if transcript.exists():
        return transcript

    ocr_subtitle = ocr_subtitle_path(video_path)
    if ocr_subtitle.exists():
        return ocr_subtitle

    return None


def youtube(endpoint, **params):
    url = f"{API}/{endpoint}"
    query = {**params, "key": os.environ["YOUTUBE_API_KEY"]}
    print(f"{url}?{urlencode({**query, 'key': '***'})}")
    time.sleep(DEFAULT_YOUTUBE_API_SLEEP_SECONDS)
    response = requests.get(url, params=query, timeout=30)
    response.raise_for_status()
    return response.json()


def video_info_path(video_path):
    candidates = [
        video_path.with_name(f"{video_path.stem}{YOUTUBE_API_INFOS_SUFFIX}"),
        video_path.with_name(f"{video_path.stem}{LEGACY_INFO_SUFFIX}"),
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def published_at(video_path):
    info_path = video_info_path(video_path)
    if info_path.exists():
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            published_at = payload.get("published_at")
            if published_at:
                return published_at
        except Exception:
            pass

    dt = datetime.fromtimestamp(video_path.stat().st_mtime, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def video_url(video_path):
    return f"https://www.youtube.com/watch?v={video_path.stem}"


def video_title(video_path):
    info_path = video_info_path(video_path)
    if info_path.exists():
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            title = payload.get("title")
            if title:
                return title
        except Exception:
            pass
    return video_path.stem


def is_capitalized_word(word):
    return bool(word) and word[0].isalpha() and word[0].isupper()


def speaker_words(name):
    return [
        word
        for word in SPEAKER_WORD_PATTERN.findall(name)
        if any(char.isalpha() for char in word)
    ]


def normalize_speaker_name(words):
    return " ".join(words).strip(" ,.;:!?-–—")


def normalize_speaker_text(text):
    return re.sub(r"\s+", " ", str(text)).strip(" ,.;:!?-–—")


def titlecase_all_caps_name(name):
    words = speaker_words(name)
    if not words:
        return name
    if any(any(char.islower() for char in word) for word in words):
        return name

    parts = []
    for part in re.split(r"(\s+|-|')", name):
        if not part or part.isspace() or part in {"-", "'"}:
            parts.append(part)
            continue
        if any(char.isalpha() for char in part):
            parts.append(part[:1].upper() + part[1:].lower())
        else:
            parts.append(part)
    return "".join(parts)


def normalize_match_text(text):
    without_accents = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(char)
    )
    normalized = re.sub(r"[^0-9A-Za-z]+", " ", without_accents.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def is_probable_speaker_name(name):
    if not name:
        return False
    words = speaker_words(name)
    if not words:
        return False
    return any(is_capitalized_word(word) for word in words)


def has_multiple_speaker_words(name):
    return len(speaker_words(name)) >= 2


def is_non_person_name(name):
    normalized = normalize_match_text(name)
    return any(keyword in normalized.split() for keyword in NON_PERSON_NAME_KEYWORDS)


def add_speaker_name(names, seen, name):
    name = titlecase_all_caps_name(normalize_speaker_text(name))
    if not is_probable_speaker_name(name):
        return
    if is_non_person_name(name):
        return

    key = normalize_match_text(name)
    if key in seen:
        return

    seen.add(key)
    names.append(name)


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
                if next_word.lower() == "la":
                    after_la = matches[next_index + 1].group(0)
                    if is_capitalized_word(after_la):
                        words.extend([word, next_word])
                        index += 2
                        continue
            if next_index < len(matches) and is_capitalized_word(matches[next_index].group(0)):
                words.append(word)
                index += 1
                continue

        break

    return normalize_speaker_name(words)


def warn_spacy_unavailable(reason):
    global _SPACY_WARNING_SHOWN

    if _SPACY_WARNING_SHOWN:
        return

    print(
        "[warn] Detection spaCy ignoree: "
        f"{reason}. Installez spaCy, CuPy et le modele francais {SPACY_FRENCH_MODEL}."
    )
    _SPACY_WARNING_SHOWN = True


def prepare_spacy_gpu(spacy):
    if not SPACY_REQUIRE_GPU:
        spacy.require_cpu()
        return True

    try:
        spacy.require_gpu()
    except Exception as exc:
        warn_spacy_unavailable(f"GPU requis mais indisponible ({exc})")
        return False

    return True


def load_french_spacy_model():
    global _SPACY_NLP, _SPACY_LOAD_ATTEMPTED

    if _SPACY_LOAD_ATTEMPTED:
        return _SPACY_NLP

    _SPACY_LOAD_ATTEMPTED = True
    try:
        import spacy
    except ImportError as exc:
        warn_spacy_unavailable(str(exc))
        return None

    if not prepare_spacy_gpu(spacy):
        return None

    try:
        _SPACY_NLP = spacy.load(SPACY_FRENCH_MODEL)
    except OSError as exc:
        warn_spacy_unavailable(str(exc))
        return None

    return _SPACY_NLP


def extract_proper_noun_speakers(doc):
    names = []
    current = []

    def flush_current():
        nonlocal current
        if current:
            names.append(normalize_speaker_text(" ".join(current)))
            current = []

    tokens = list(doc)
    for index, token in enumerate(tokens):
        lower_text = token.text.lower()
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None

        if token.pos_ == "PROPN" and token.is_alpha:
            current.append(token.text)
            continue

        if (
            lower_text in LOWERCASE_CONNECTORS
            and current
            and next_token is not None
            and next_token.pos_ == "PROPN"
        ):
            current.append(token.text)
            continue

        if token.text in {"-", "–", "—"} and current:
            current.append(token.text)
            continue

        flush_current()

    flush_current()
    return names


def extract_spacy_speakers_from_doc(doc):
    organization_names = {
        normalize_speaker_text(ent.text).casefold()
        for ent in doc.ents
        if ent.label_ in SPACY_ORG_LABELS
    }
    entity_names = [
        normalize_speaker_text(ent.text)
        for ent in doc.ents
        if ent.label_ in SPACY_PERSON_LABELS
        and normalize_speaker_text(ent.text).casefold() not in organization_names
    ]
    if entity_names:
        return entity_names

    return extract_proper_noun_speakers(doc)


def extract_spacy_speakers(text):
    nlp = load_french_spacy_model()
    if nlp is None:
        return []

    nlp.max_length = max(nlp.max_length, len(text) + 100)
    return extract_spacy_speakers_from_doc(nlp(text))


def load_processed_non_subtitle_texts(video_path):
    path = ocr_processed_path(video_path)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] OCR processed illisible pour {video_path.stem}: {exc}")
        return None

    texts = []
    for item in payload.get("items", []):
        kind = str(item.get("kind", "")).strip().lower()
        if kind in {"subtitle", "ocr_error"}:
            continue

        text = normalize_speaker_text(item.get("text", ""))
        if text:
            texts.append(text)

    return texts


def ocr_box_bounds(item):
    try:
        xs = [float(point[0]) for point in item.get("box", [])]
        ys = [float(point[1]) for point in item.get("box", [])]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None

    return min(xs), min(ys), max(xs), max(ys)


def is_lower_third_ocr_item(item):
    bounds = ocr_box_bounds(item)
    if bounds is None:
        return False

    return bounds[1] >= OCR_LOWER_THIRD_MIN_TOP


def has_lower_third_companion(item, items):
    bounds = ocr_box_bounds(item)
    if bounds is None:
        return False

    left, _top, right, bottom = bounds
    width = max(right - left, 1)
    image = item.get("image")

    for other in items:
        if other is item:
            continue
        if str(other.get("kind", "")).strip().lower() != "lower_third":
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


def load_processed_speaker_candidate_texts(video_path):
    path = ocr_processed_path(video_path)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] OCR processed illisible pour {video_path.stem}: {exc}")
        return None

    texts = []
    items = payload.get("items", [])
    for item in items:
        kind = str(item.get("kind", "")).strip().lower()
        if kind != "name":
            continue
        if not is_lower_third_ocr_item(item):
            continue
        if not has_lower_third_companion(item, items):
            continue

        text = normalize_speaker_text(item.get("text", ""))
        if text:
            texts.append(text)

    return texts


def processed_non_subtitle_match_text(processed_non_subtitle_texts):
    if processed_non_subtitle_texts is None:
        return None

    return " ".join(
        text
        for text in (normalize_match_text(text) for text in processed_non_subtitle_texts)
        if text
    )


def load_processed_non_subtitle_text(video_path):
    return processed_non_subtitle_match_text(load_processed_non_subtitle_texts(video_path))


def speaker_candidate_in_processed(name, processed_non_subtitle_text):
    if processed_non_subtitle_text is None:
        return True

    normalized_name = normalize_match_text(name)
    return bool(normalized_name) and normalized_name in processed_non_subtitle_text


def is_standalone_person_name_candidate(text):
    text = normalize_speaker_text(text)
    if not text:
        return False
    if is_non_person_name(text):
        return False
    if re.search(r"[@:/\\0-9()\[\]]", text):
        return False

    words = speaker_words(text)
    if len(words) < 2 or len(words) > 4:
        return False

    for word in words:
        lower_word = word.lower()
        if lower_word in LOWERCASE_CONNECTORS:
            continue
        if not is_capitalized_word(word):
            return False

    return True


def extract_processed_speakers(processed_non_subtitle_texts):
    if not processed_non_subtitle_texts:
        return []

    candidate_texts = [
        text
        for text in processed_non_subtitle_texts
        if is_standalone_person_name_candidate(text)
    ]
    if not candidate_texts:
        return []

    nlp = load_french_spacy_model()
    if nlp is None:
        return candidate_texts

    nlp.max_length = max(
        nlp.max_length,
        max((len(text) for text in candidate_texts), default=0) + 100,
    )
    names = []
    for doc in nlp.pipe(candidate_texts):
        names.extend(
            name
            for name in extract_spacy_speakers_from_doc(doc)
            if is_standalone_person_name_candidate(name)
        )
    return names or candidate_texts


def extract_speakers(text, processed_non_subtitle_text=None, processed_non_subtitle_texts=None):
    names = []
    seen = set()

    for match in SPEAKER_INTRO_PATTERN.finditer(text):
        name = extract_speaker_name(text, match.end())
        if speaker_candidate_in_processed(name, processed_non_subtitle_text):
            add_speaker_name(names, seen, name)

    for name in extract_spacy_speakers(text):
        if has_multiple_speaker_words(name) and speaker_candidate_in_processed(name, processed_non_subtitle_text):
            add_speaker_name(names, seen, name)

    for name in extract_processed_speakers(processed_non_subtitle_texts):
        if has_multiple_speaker_words(name):
            add_speaker_name(names, seen, name)

    return names


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
    current_len = 0

    def flush_chunk():
        nonlocal current, current_len
        if not current.strip():
            return
        chunks.append(current.strip())
        current = ""
        current_len = 0

    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)
        if not paragraph:
            continue
        for sentence in split_sentences(paragraph):
            projected = current_len + len(sentence) + (1 if current else 0)
            if current and projected > max_chars:
                flush_chunk()
            if not current:
                current = sentence
                current_len = len(sentence)
            else:
                current = f"{current} {sentence}"
                current_len = len(current)

    flush_chunk()
    return chunks


def build_chunks_payload(text, meta_data):
    chunks = split_into_chunks(text)
    return {
        "chunking": {
            "max_chars": DEFAULT_MAX_CHARS,
            "cut_policy": "cut_at_next_sentence_after_threshold",
        },
        "chunks": [
            {
                "chunk_index": index + 1,
                "meta_data": meta_data,
                "content": chunk,
                "alert": word_count(chunk) > ALERT_WORD_THRESHOLD,
                "alert_reason": "over_3000_words" if word_count(chunk) > ALERT_WORD_THRESHOLD else None,
                "char_count": len(chunk),
            }
            for index, chunk in enumerate(chunks)
        ],
    }


def create_chunks(video_path, force=False):
    target = chunks_path(video_path)
    source = source_text_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if source is None:
        print(f"[skip] transcript ou ocr_subtitle introuvable pour: {video_path.stem}")
        return None

    text = source.read_text(encoding="utf-8")
    normalized = normalize_text(text)
    if not normalized:
        print(f"[skip] transcript vide: {source}")
        return None

    processed_non_subtitle_texts = load_processed_non_subtitle_texts(video_path)
    processed_non_subtitle_text = processed_non_subtitle_match_text(processed_non_subtitle_texts)
    processed_speaker_candidate_texts = load_processed_speaker_candidate_texts(video_path)
    meta_data = {
        "speakers": extract_speakers(
            normalized,
            processed_non_subtitle_text,
            processed_speaker_candidate_texts,
        ),
    }
    payload = build_chunks_payload(normalized, meta_data)
    payload["source"] = str(source.relative_to(video_path.parent))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "create_chunks",
        {
            "status": "done",
            "source": str(source.relative_to(video_path.parent)).replace("\\", "/"),
            "chunks_file": str(target.relative_to(video_path.parent)).replace("\\", "/"),
            "chunk_count": len(payload["chunks"]),
            "speakers": meta_data.get("speakers", []),
        },
    )
    print(f"[ok] {target} ({len(payload['chunks'])} chunks)")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cree des chunks JSON a partir des transcripts sans timecodes."
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
        "--force",
        action="store_true",
        help="Regenere les chunks meme s'ils existent deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if create_chunks(video_path, force=args.force):
            done += 1

    print(f"{done} transcripts chunkes.")


if __name__ == "__main__":
    main()
