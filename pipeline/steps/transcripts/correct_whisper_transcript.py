import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.support.brand_normalization import (
    IONIS_STM_TARGET,
    normalize_ionis_stm_text,
)
from pipeline.support.json_io import read_json
from pipeline.support.paths import existing_ocr_dir, existing_transcripts_dir
from pipeline.steps.transcripts.artifacts import (
    LEGACY_TRANSCRIPT_1_NAMES,
    TRANSCRIPT_1_BRUT_NAME,
    TRANSCRIPT_2_CORRECTED_NAME,
    TRANSCRIPT_2_CORRECTIONS_NAME,
)

TIMECODED_SOURCE_NAMES = (
    TRANSCRIPT_1_BRUT_NAME,
    *LEGACY_TRANSCRIPT_1_NAMES,
    "ocr_subtitles_timecoded.txt",
)
LEGACY_TIMECODED_SUFFIXES = ("_transcript_timecodes.txt", "_ocr_subtitle_timecodes.txt")
CORRECTED_SUFFIX = "_corrected.txt"
CORRECTIONS_SUFFIX = "_corrections.tsv"
LEGACY_CORRECTED_WORDS_SUFFIX = "_corrected_words.txt"
WORD_PATTERN = re.compile(r"\w+|\W+", re.UNICODE)
TOKEN_PATTERN = re.compile(r"\b[\w'â€™\-]+\b", re.UNICODE)
TIMECODE_PREFIX = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})(?:-((?:\d{2}:)?\d{2}:\d{2}))?\]\s*")
COMMON_WORDS = {
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "de", "du", "des", "le", "la", "les", "un", "une", "et", "ou",
    "a", "au", "aux", "en", "dans", "sur", "pour", "avec", "sans", "par",
    "ce", "ces", "cet", "cette", "son", "sa", "ses",
    "mon", "ma", "mes", "ton", "ta", "tes", "notre", "votre", "leur", "leurs",
    "qui", "que", "quoi", "dont", "mais", "donc", "car", "ni", "or",
    "bonjour", "merci", "alors", "voila", "oui", "non",
    "premier", "second", "deuxieme", "projet", "charge", "charges",
    "responsable", "assistant", "assistante", "chef", "fondatrice",
    "analyste", "analyst", "business", "buisness", "affaire", "affaires",
    "management", "energie", "communication", "confiance", "parole",
    "forum", "forums", "meeting", "meetings", "workshop", "coaching",
    "entretien", "entretiens", "promo", "promotion",
}
OCR_NAME_KINDS = {"name", "lower_third", "title", "logo"}
CORRECTION_MODES = {
    "conservative": {"min_count": 3, "same_initial_cutoff": 0.90, "any_initial_cutoff": 0.95},
    "balanced": {"min_count": 2, "same_initial_cutoff": 0.84, "any_initial_cutoff": 0.90},
    "aggressive": {"min_count": 1, "same_initial_cutoff": 0.78, "any_initial_cutoff": 0.84},
}
DEFAULT_CORRECTION_MODE = "balanced"

def processed_ocr_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    corrected_candidates = (
        video_ocr_dir / "corrected_ocr_items.json",
        video_ocr_dir / "ocr_processed_corrected.json",
        video_ocr_dir / f"{video_path.stem}_ocr_processed_corrected.json",
    )
    for candidate in corrected_candidates:
        if candidate.exists():
            return candidate
    legacy_transcript_dir = video_path.parent / "transcript"
    legacy_corrected = legacy_transcript_dir / f"{video_path.stem}_ocr_processed_corrected.json"
    if legacy_corrected.exists():
        return legacy_corrected
    processed_candidates = (video_ocr_dir / "01_processed_ocr_items.json",)
    for candidate in processed_candidates:
        if candidate.exists():
            return candidate
    return processed_candidates[0]


def timecodes_source_path(video_path, *, transcripts_dir_name=None):
    transcript_dir = existing_transcripts_dir(
        video_path,
        name=transcripts_dir_name,
    )
    candidates = []
    for name in TIMECODED_SOURCE_NAMES:
        path = transcript_dir / name
        if path.exists():
            candidates.append(path)
    for suffix in LEGACY_TIMECODED_SUFFIXES:
        candidates.extend(sorted(transcript_dir.glob(f"{video_path.stem}{suffix}")))
    if candidates:
        return candidates[0]

    generic_candidates = sorted(
        path
        for pattern in ("*timecoded.txt", f"{video_path.stem}*timecodes*.txt")
        for path in transcript_dir.glob(pattern)
        if not path.name.endswith(CORRECTED_SUFFIX)
        and not path.name.endswith(CORRECTIONS_SUFFIX)
        and not path.name.endswith(LEGACY_CORRECTED_WORDS_SUFFIX)
    )
    if generic_candidates:
        return generic_candidates[0]

    raise FileNotFoundError(f"Aucun fichier timecodes trouve pour {video_path.stem} dans {transcript_dir}")


def corrected_path(source_path):
    if source_path.name in {
        TRANSCRIPT_1_BRUT_NAME,
        *LEGACY_TRANSCRIPT_1_NAMES,
    }:
        return source_path.with_name(TRANSCRIPT_2_CORRECTED_NAME)
    if source_path.name in TIMECODED_SOURCE_NAMES:
        return source_path.with_name(source_path.name[: -len(".txt")] + CORRECTED_SUFFIX)
    for suffix in LEGACY_TIMECODED_SUFFIXES:
        if source_path.name.endswith(suffix):
            return source_path.with_name(source_path.name[: -len(".txt")] + CORRECTED_SUFFIX)
    return source_path.with_name(f"{source_path.stem}{CORRECTED_SUFFIX}")


def corrected_words_path(source_path):
    if source_path.name in {
        TRANSCRIPT_1_BRUT_NAME,
        *LEGACY_TRANSCRIPT_1_NAMES,
    }:
        return source_path.with_name(TRANSCRIPT_2_CORRECTIONS_NAME)
    if source_path.name in TIMECODED_SOURCE_NAMES:
        return source_path.with_name(source_path.name[: -len(".txt")] + CORRECTIONS_SUFFIX)
    for suffix in LEGACY_TIMECODED_SUFFIXES:
        if source_path.name.endswith(suffix):
            return source_path.with_name(source_path.name[: -len(".txt")] + LEGACY_CORRECTED_WORDS_SUFFIX)
    return source_path.with_name(f"{source_path.stem}{CORRECTIONS_SUFFIX}")


def strip_accents(text):
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_word(word):
    return re.sub(r"[^0-9a-z]+", "", strip_accents(word).casefold())


def tokenize_words(text):
    return [token for token in TOKEN_PATTERN.findall(str(text)) if token.strip()]


def is_ocr_name_kind(kind):
    normalized = str(kind or "").strip().lower()
    return normalized in OCR_NAME_KINDS or normalized in {"graphic", "outro"}


def load_ocr_lexicon(path, min_count=2):
    payload = read_json(path)
    counts = Counter()
    forms = defaultdict(Counter)
    name_phrases = defaultdict(Counter)
    by_initial = defaultdict(set)
    for item in payload.get("items", []):
        if not is_ocr_name_kind(item.get("kind")):
            continue
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        words = [clean_word_form(word) for word in tokenize_words(text)]
        phrase_words = [word for word in words if is_ocr_name_candidate(word)]
        if 2 <= len(phrase_words) <= 4:
            phrase_key = tuple(normalize_word(word) for word in phrase_words)
            if all(phrase_key):
                name_phrases[phrase_key[0]][tuple(phrase_words)] += 1
        for word in words:
            if not is_ocr_name_candidate(word):
                continue
            normalized = normalize_word(word)
            if not normalized:
                continue
            counts[normalized] += 1
            forms[normalized][clean_word_form(word)] += 1
            by_initial[normalized[:1]].add(normalized)

    counts = Counter({word: count for word, count in counts.items() if count >= min_count})
    forms = defaultdict(Counter, {word: forms[word] for word in counts})
    by_initial = defaultdict(set)
    for word in counts:
        by_initial[word[:1]].add(word)
    name_phrases = defaultdict(
        Counter,
        {
            first_word: Counter({phrase: count for phrase, count in phrases.items() if count >= min_count})
            for first_word, phrases in name_phrases.items()
        },
    )
    return counts, forms, by_initial, name_phrases


def choose_correction(word, lexicon_counts, lexicon_forms, lexicon_by_initial, settings):
    normalized = normalize_word(word)
    if not normalized:
        return word
    if normalized in lexicon_counts:
        return word
    if not is_proper_noun_candidate(word):
        return word

    match = best_lexicon_match(normalized, lexicon_counts, lexicon_forms, lexicon_by_initial, settings)
    if not match:
        return word

    best = match
    if best == normalized:
        return word

    display = best_display_form(best, lexicon_forms)
    if not is_proper_noun_candidate(display):
        return word

    return adapt_casing(word, display)


def clean_word_form(word):
    return str(word).strip(" \t\r\n,.;:!?\"'()[]{}")


def is_ocr_name_candidate(word):
    cleaned = clean_word_form(word)
    normalized = normalize_word(cleaned)
    if not normalized or normalized in COMMON_WORDS:
        return False
    if len(normalized) < 3 or normalized.isdigit():
        return False
    if cleaned.isupper() and len(normalized) <= 4:
        return False
    return any(char.isupper() for char in cleaned) or "-" in cleaned


def similarity(left, right):
    return SequenceMatcher(None, left, right).ratio()


def required_cutoff(source, candidate, same_initial, settings):
    cutoff = settings["same_initial_cutoff"] if same_initial else settings["any_initial_cutoff"]
    shortest = min(len(source), len(candidate))
    if shortest <= 4:
        cutoff += 0.06
    elif shortest >= 9:
        cutoff -= 0.02
    return min(0.97, max(0.0, cutoff))


def candidate_length_is_plausible(source, candidate):
    delta = abs(len(source) - len(candidate))
    return delta <= max(2, round(max(len(source), len(candidate)) * 0.35))


def common_prefix_length(left, right):
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def has_strong_stem_match(source, candidate):
    if source[:1] != candidate[:1]:
        return False
    shortest = min(len(source), len(candidate))
    if shortest < 8:
        return False
    return common_prefix_length(source, candidate) >= max(7, round(shortest * 0.70))


def display_quality(candidate, lexicon_forms):
    display = best_display_form(candidate, lexicon_forms)
    if not display.isupper():
        return 2
    if "-" in display:
        return 1
    return 0


def best_lexicon_match(normalized, lexicon_counts, lexicon_forms, lexicon_by_initial, settings):
    same_initial = lexicon_by_initial.get(normalized[:1], set())
    candidate_groups = [(same_initial, True)]
    if len(normalized) >= 5:
        candidate_groups.append((set(lexicon_counts) - same_initial, False))

    best = None
    best_score = 0.0
    for candidates, initial_matches in candidate_groups:
        for candidate in candidates:
            if not candidate_length_is_plausible(normalized, candidate):
                continue
            score = similarity(normalized, candidate)
            if score < required_cutoff(normalized, candidate, initial_matches, settings) and not has_strong_stem_match(normalized, candidate):
                continue
            ranking = (display_quality(candidate, lexicon_forms), score, lexicon_counts[candidate], len(candidate))
            if not best or ranking > best_score:
                best = candidate
                best_score = ranking
        if best:
            return best
    return None


def best_display_form(normalized, lexicon_forms):
    forms = lexicon_forms.get(normalized)
    if not forms:
        return normalized
    return forms.most_common(1)[0][0]


def titlecase_hyphenated(text):
    return "-".join(part[:1].upper() + part[1:].lower() for part in text.split("-"))


def adapt_casing(source_word, display_word):
    if source_word.isupper():
        return display_word.upper()
    if display_word.isupper():
        if "-" in display_word and all(len(part) > 3 for part in display_word.split("-")):
            return titlecase_hyphenated(display_word)
        if len(normalize_word(display_word)) > 4:
            return display_word[:1].upper() + display_word[1:].lower()
    if source_word[:1].isupper():
        return display_word[:1].upper() + display_word[1:]
    return display_word


def is_proper_noun_candidate(word):
    cleaned = clean_word_form(word)
    if not cleaned:
        return False
    if " " in cleaned:
        return any(is_proper_noun_candidate(part) for part in re.split(r"[-']", cleaned) if part)
    if normalize_word(cleaned) in COMMON_WORDS:
        return False
    if len(cleaned) < 3:
        return False
    if cleaned.isupper() and len(cleaned) <= 4:
        return False
    return cleaned[:1].isupper()


def suffix_match_length(left, right):
    count = 0
    for left_char, right_char in zip(reversed(left), reversed(right)):
        if left_char != right_char:
            break
        count += 1
    return count


def is_contextual_name_match(source_word, candidate_word):
    source = normalize_word(source_word)
    candidate = normalize_word(candidate_word)
    if source == candidate:
        return False
    if source[:1] == candidate[:1]:
        return False
    if len(source) < 4 or len(candidate) < 4:
        return False
    if not candidate_length_is_plausible(source, candidate):
        return False
    return similarity(source, candidate) >= 0.65 or suffix_match_length(source, candidate) >= 4


def choose_phrase_correction(first_word, second_word, name_phrases):
    first_normalized = normalize_word(first_word)
    if not first_normalized or first_normalized not in name_phrases:
        return None
    if not is_proper_noun_candidate(first_word) or not is_proper_noun_candidate(second_word):
        return None

    best = None
    best_score = 0.0
    for phrase, count in name_phrases[first_normalized].items():
        if len(phrase) < 2:
            continue
        candidate = phrase[1]
        if not is_proper_noun_candidate(candidate):
            continue
        if not is_contextual_name_match(second_word, candidate):
            continue
        score = (similarity(normalize_word(second_word), normalize_word(candidate)), count)
        if not best or score > best_score:
            best = candidate
            best_score = score
    return adapt_casing(second_word, best) if best else None


def normalize_ionis_stm(text, corrections):
    """Apply the known brand correction when subtitles cannot be reconciled."""
    updated, matched_sources = normalize_ionis_stm_text(text)
    for source in matched_sources:
        if source != IONIS_STM_TARGET:
            corrections[source].add(IONIS_STM_TARGET)
    return updated


def correct_text(text, lexicon_counts, lexicon_forms, lexicon_by_initial, name_phrases, settings, corrections):
    text = normalize_ionis_stm(text, corrections)
    tokens = WORD_PATTERN.findall(text)
    pieces = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if TOKEN_PATTERN.fullmatch(token):
            if index + 2 < len(tokens) and tokens[index + 1].isspace() and TOKEN_PATTERN.fullmatch(tokens[index + 2]):
                phrase_corrected = choose_phrase_correction(token, tokens[index + 2], name_phrases)
                if phrase_corrected and phrase_corrected != tokens[index + 2]:
                    corrections[tokens[index + 2]].add(phrase_corrected)
                    pieces.extend([token, tokens[index + 1], phrase_corrected])
                    index += 3
                    continue
            corrected = choose_correction(token, lexicon_counts, lexicon_forms, lexicon_by_initial, settings)
            if corrected != token and (is_proper_noun_candidate(token) or is_proper_noun_candidate(corrected)):
                corrections[token].add(corrected)
            pieces.append(corrected)
        else:
            pieces.append(token)
        index += 1
    return "".join(pieces)


def correct_line(line, lexicon_counts, lexicon_forms, lexicon_by_initial, name_phrases, settings, corrections):
    match = TIMECODE_PREFIX.match(line)
    if match:
        prefix = match.group(0)
        body = line[len(prefix) :]
        return prefix + correct_text(body, lexicon_counts, lexicon_forms, lexicon_by_initial, name_phrases, settings, corrections)
    return correct_text(line, lexicon_counts, lexicon_forms, lexicon_by_initial, name_phrases, settings, corrections)


def correction_settings(mode):
    return CORRECTION_MODES[mode]


def correct_file(
    video_path,
    force=False,
    mode=DEFAULT_CORRECTION_MODE,
    *,
    transcripts_dir_name=None,
):
    source = timecodes_source_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    analyse = processed_ocr_path(video_path)
    target = corrected_path(source)
    words_target = corrected_words_path(source)

    if not source.exists():
        print(f"[skip] timecodes introuvable: {source}")
        return None
    if not analyse.exists():
        print(f"[skip] analyse introuvable: {analyse}")
        return None
    if (
        target.exists()
        and not force
        and target.stat().st_mtime >= max(source.stat().st_mtime, analyse.stat().st_mtime)
    ):
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: Whisper brut ou OCR plus recent")

    settings = correction_settings(mode)
    lexicon_counts, lexicon_forms, lexicon_by_initial, name_phrases = load_ocr_lexicon(analyse, settings["min_count"])
    corrections = defaultdict(set)
    lines = [
        correct_line(line, lexicon_counts, lexicon_forms, lexicon_by_initial, name_phrases, settings, corrections)
        for line in source.read_text(encoding="utf-8").splitlines()
    ]
    target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    correction_lines = [
        f"{old_word}\t{new_word}"
        for old_word in sorted(corrections)
        for new_word in sorted(corrections[old_word])
    ]
    words_target.write_text(("\n".join(correction_lines).strip() + "\n") if correction_lines else "", encoding="utf-8")
    print(f"[ok] {target}")
    print(f"[ok] {words_target}")
    return target
