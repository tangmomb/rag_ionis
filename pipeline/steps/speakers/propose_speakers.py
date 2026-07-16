import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from pipeline.support.analysis import update_analysed_infos
from pipeline.support.paths import (
    existing_ocr_dir,
    existing_transcripts_dir,
    existing_youtube_api_infos_path,
    relative_to_video_dir,
    speakers_dir,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
PLAIN_NAME = "plain_transcript.txt"
LEGACY_PLAIN_SUFFIX = "_transcript.txt"
OCR_SUBTITLE_NAME = "ocr_subtitles.txt"
LEGACY_OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
WHISPER_TIMECODED_NAME = "whisper_transcript_timecoded.txt"
LEGACY_WHISPER_TIMECODED_SUFFIX = "_transcript_timecodes.txt"
OCR_TIMECODED_CORRECTED_NAME = "ocr_subtitles_timecoded_corrected.txt"
LEGACY_OCR_TIMECODED_CORRECTED_SUFFIX = "_ocr_subtitle_timecodes_corrected.txt"
OCR_PROCESSED_NAME = "01_processed_ocr_items.json"
OCR_PROCESSED_CORRECTED_NAME = "corrected_ocr_items.json"
LEGACY_OCR_PROCESSED_CORRECTED_SUFFIX = "_ocr_processed_corrected.json"
SPEAKER_CANDIDATES_NAME = "speaker_candidates.json"
OCR_LOWER_THIRD_MIN_TOP = 320
SPEAKER_INTRO_PATTERN = re.compile(r"je m'appelle\s+", re.IGNORECASE)
SPEAKER_JE_SUIS_PATTERN = re.compile(r"je suis\s+", re.IGNORECASE)
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
    transcript_dir = existing_transcripts_dir(video_path)
    whisper_timecoded = transcript_dir / WHISPER_TIMECODED_NAME
    if whisper_timecoded.exists():
        return whisper_timecoded
    legacy_whisper_timecoded = sorted(transcript_dir.glob(f"*{LEGACY_WHISPER_TIMECODED_SUFFIX}"))
    if legacy_whisper_timecoded:
        return legacy_whisper_timecoded[0]
    ocr_timecoded_corrected = transcript_dir / OCR_TIMECODED_CORRECTED_NAME
    if ocr_timecoded_corrected.exists():
        return ocr_timecoded_corrected
    legacy_ocr_timecoded_corrected = sorted(
        transcript_dir.glob(f"*{LEGACY_OCR_TIMECODED_CORRECTED_SUFFIX}")
    )
    if legacy_ocr_timecoded_corrected:
        return legacy_ocr_timecoded_corrected[0]
    transcript = transcript_path(video_path)
    if transcript.exists():
        return transcript
    ocr_subtitle = ocr_subtitle_path(video_path)
    return ocr_subtitle if ocr_subtitle.exists() else None


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
            payload = json.loads(source.read_text(encoding="utf-8"))
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
        if other is item or str(other.get("kind", "")).strip().lower() != "lower_third":
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
    return all(word.lower() in LOWERCASE_CONNECTORS or is_capitalized_word(word) for word in words)


def load_ocr_speaker_candidates(video_path):
    path = ocr_processed_path(video_path)
    if not path.exists():
        return [], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] OCR processed illisible pour {video_path.stem}: {exc}")
        return [], path
    items = payload.get("items", [])
    names = []
    for item in items:
        if str(item.get("kind", "")).strip().lower() != "name":
            continue
        if not is_lower_third_ocr_item(item) or not has_lower_third_companion(item, items):
            continue
        text = normalize_speaker_text(item.get("text", ""))
        if is_standalone_person_name_candidate(text):
            names.append(text)
    return names, path


def propose_speakers(text, ocr_names):
    candidates = {}
    for match in SPEAKER_INTRO_PATTERN.finditer(text):
        add_candidate(candidates, extract_speaker_name(text, match.end()), "transcript_je_m_appelle")
    for match in SPEAKER_JE_SUIS_PATTERN.finditer(text):
        name = extract_speaker_name(text, match.end())
        if has_multiple_speaker_words(name):
            add_candidate(candidates, name, "transcript_je_suis")
    for name in ocr_names:
        add_candidate(candidates, name, "ocr_lower_third")
    values = list(candidates.values())
    return {
        "speakers": [entry["name"] for entry in values],
        "candidates": values,
    }


def propose_for_video(video_path, force=False):
    source = source_text_path(video_path)
    target = candidates_path(video_path)
    if source is None:
        print(f"[skip] transcript introuvable pour: {video_path.stem}")
        return None
    expected_source = relative_to_video_dir(source, video_path)
    if target.exists() and not force:
        try:
            existing_payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_payload = {}
        if (
            existing_payload.get("source") == expected_source
            and target.stat().st_mtime >= source.stat().st_mtime
        ):
            print(f"[skip] {target.name} existe deja")
            return target
        print(f"[regen] {target.name}: source Whisper brute plus recente ou differente")
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        print(f"[skip] transcript vide: {source}")
        return None
    ocr_names, ocr_source = load_ocr_speaker_candidates(video_path)
    payload = propose_speakers(text, ocr_names)
    payload.update(
        {
            "video_title": video_title(video_path),
            "source": expected_source,
            "ocr_source": relative_to_video_dir(ocr_source, video_path) if ocr_source and ocr_source.exists() else None,
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "propose_speakers",
        {
            "status": "done",
            "candidates_file": relative_to_video_dir(target, video_path),
            "speaker_candidates": payload["speakers"],
        },
    )
    print(f"[ok] {target} ({len(payload['speakers'])} candidat(s))")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Propose les speakers depuis le transcript brut avant sa correction."
    )
    parser.add_argument("--video-dir", help="Dossier contenant les videos.")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    done = sum(bool(propose_for_video(video_path, force=args.force)) for video_path in videos)
    print(f"{done} JSON de candidats speakers generes.")


if __name__ == "__main__":
    main()
