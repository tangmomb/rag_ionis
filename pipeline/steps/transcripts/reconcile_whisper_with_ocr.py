from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.steps.transcripts.artifacts import (
    TRANSCRIPT_1_BRUT_NAME,
    TRANSCRIPT_2_CORRECTED_NAME,
    TRANSCRIPT_2_CORRECTIONS_NAME,
)
from pipeline.support.paths import (
    CANONICAL_TRANSCRIPTS_DIR_NAME,
    OCR_CORRECTION_TRANSCRIPTS_DIR_NAME,
    existing_transcripts_dir,
    output_is_current,
)


WHISPER_SOURCE_NAME = TRANSCRIPT_1_BRUT_NAME
OCR_SOURCE_NAMES = (
    "plain_transcript.txt",
)
TARGET_NAME = TRANSCRIPT_2_CORRECTED_NAME
REPORT_NAME = TRANSCRIPT_2_CORRECTIONS_NAME
WHISPER_LINE = re.compile(
    r"^(?P<prefix>\[(?P<start>(?:\d{2}:)?\d{2}:\d{2})"
    r"(?:-(?P<end>(?:\d{2}:)?\d{2}:\d{2}))?\]\s*)"
    r"(?P<speaker>SPEAKER_\d+\s*:\s*)?"
    r"(?P<body>.*)$"
)
OCR_LINE = re.compile(
    r"^\[(?P<time>(?:\d{2}:)?\d{2}:\d{2})\]\s*(?P<body>.*)$"
)
WORD_TOKEN = re.compile(
    r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+"
    r"(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*"
)
MODE_CUTOFFS = {
    "conservative": 0.93,
    "balanced": 0.87,
    "aggressive": 0.82,
}
COMMON_WORDS = {
    "alors",
    "avec",
    "bonjour",
    "car",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "est",
    "et",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "mais",
    "nous",
    "on",
    "ou",
    "par",
    "pas",
    "pour",
    "que",
    "qui",
    "sommes",
    "sur",
    "un",
    "une",
    "vous",
}


def parse_timecode(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def compact_normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^0-9a-z]+", "", without_marks)


def token_normalize(value: str) -> str:
    return compact_normalize(value)


def lexical_word_count(value: str) -> int:
    return len(lexical_components(value))


def lexical_components(value: str) -> list[str]:
    return [
        compact_normalize(component)
        for component in re.findall(
            r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+",
            str(value),
        )
        if compact_normalize(component)
    ]


def components_are_close(
    source: str,
    candidate: str,
    cutoff: float,
) -> bool:
    source_components = lexical_components(source)
    candidate_components = lexical_components(candidate)
    if len(source_components) != len(candidate_components):
        return False
    component_cutoff = max(0.75, cutoff - 0.10)
    return all(
        left == right
        or SequenceMatcher(None, left, right).ratio() >= component_cutoff
        for left, right in zip(source_components, candidate_components)
    )


def whisper_source_path(video_path: Path) -> Path:
    return (
        existing_transcripts_dir(
            video_path,
            name=CANONICAL_TRANSCRIPTS_DIR_NAME,
        )
        / WHISPER_SOURCE_NAME
    )


def ocr_source_path(video_path: Path) -> Path | None:
    directory = existing_transcripts_dir(
        video_path,
        name=OCR_CORRECTION_TRANSCRIPTS_DIR_NAME,
    )
    return next(
        (
            directory / name
            for name in OCR_SOURCE_NAMES
            if (directory / name).exists()
        ),
        None,
    )


def corrected_path(video_path: Path) -> Path:
    return (
        existing_transcripts_dir(
            video_path,
            name=CANONICAL_TRANSCRIPTS_DIR_NAME,
        )
        / TARGET_NAME
    )


def corrections_path(video_path: Path) -> Path:
    return corrected_path(video_path).with_name(REPORT_NAME)


def parse_ocr_entries(text: str) -> list[dict[str, object]]:
    entries = []
    for line in str(text).splitlines():
        match = OCR_LINE.match(line.strip())
        if not match:
            continue
        body = match.group("body").strip()
        if body:
            entries.append(
                {
                    "second": parse_timecode(match.group("time")),
                    "text": body,
                }
            )
    return entries


def style_score(text: str, tokens: list[re.Match[str]]) -> int:
    token_values = [match.group(0) for match in tokens]
    special = 4 if any(character in text for character in "-'’") else 0
    acronym = 3 if any(
        token.isupper() and len(token_normalize(token)) >= 2
        for token in token_values
    ) else 0
    titled = sum(
        1
        for token in token_values
        if token[:1].isupper() and not token.isupper()
    )
    multiple_names = 2 if titled >= 2 else 0
    mixed_alphanumeric = 2 if (
        any(character.isalpha() for character in text)
        and any(character.isdigit() for character in text)
    ) else 0
    return special + acronym + multiple_names + mixed_alphanumeric


def canonical_terms(text: str, *, max_words: int = 4) -> list[dict[str, object]]:
    matches = list(WORD_TOKEN.finditer(str(text)))
    candidates = []
    seen = set()
    for start_index in range(len(matches)):
        for size in range(1, min(max_words, len(matches) - start_index) + 1):
            selected = matches[start_index : start_index + size]
            start = selected[0].start()
            end = selected[-1].end()
            display = str(text)[start:end]
            normalized = compact_normalize(display)
            if len(normalized) < 4:
                continue
            normalized_tokens = {
                token_normalize(match.group(0))
                for match in selected
            }
            if normalized_tokens and normalized_tokens <= COMMON_WORDS:
                continue
            score = style_score(display, selected)
            if (
                score <= 0
                and size == 1
                and start_index > 0
                and display[:1].isupper()
            ):
                # Un mot capitalise au milieu d'un sous-titre est un candidat
                # raisonnable pour un nom propre, contrairement au premier mot
                # d'une phrase qui est capitalise par convention.
                score = 1
            if score <= 0:
                continue
            key = (normalized, display)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "display": display,
                    "normalized": normalized,
                    "style_score": score,
                    "word_count": lexical_word_count(display),
                }
            )
    return candidates


def whisper_spans(text: str, *, max_words: int = 4) -> list[dict[str, object]]:
    matches = list(WORD_TOKEN.finditer(str(text)))
    spans = []
    for start_index in range(len(matches)):
        for size in range(1, min(max_words, len(matches) - start_index) + 1):
            selected = matches[start_index : start_index + size]
            start = selected[0].start()
            end = selected[-1].end()
            display = str(text)[start:end]
            normalized = compact_normalize(display)
            if len(normalized) < 4:
                continue
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "display": display,
                    "normalized": normalized,
                    "word_count": lexical_word_count(display),
                }
            )
    return spans


def plausible_match(source: str, candidate: str, cutoff: float) -> float | None:
    if source == candidate:
        return 1.0
    length_delta = abs(len(source) - len(candidate))
    if length_delta > max(2, round(max(len(source), len(candidate)) * 0.30)):
        return None
    ratio = SequenceMatcher(None, source, candidate).ratio()
    return ratio if ratio >= cutoff else None


def correction_proposals(
    whisper_text: str,
    ocr_texts: list[str],
    *,
    mode: str = "balanced",
) -> list[dict[str, object]]:
    cutoff = MODE_CUTOFFS[mode]
    spans = whisper_spans(whisper_text)
    proposals = []
    for ocr_text in ocr_texts:
        for term in canonical_terms(ocr_text):
            for span in spans:
                if span["normalized"] != term["normalized"]:
                    if span["word_count"] != term["word_count"]:
                        continue
                    if not components_are_close(
                        str(span["display"]),
                        str(term["display"]),
                        cutoff,
                    ):
                        continue
                ratio = plausible_match(
                    str(span["normalized"]),
                    str(term["normalized"]),
                    cutoff,
                )
                if ratio is None or str(span["display"]) == str(term["display"]):
                    continue
                density = float(term["style_score"]) / int(term["word_count"])
                proposals.append(
                    {
                        **span,
                        "replacement": term["display"],
                        "similarity": ratio,
                        "exact_normalized": ratio == 1.0,
                        "style_density": density,
                    }
                )
    return proposals


def apply_proposals(
    text: str,
    proposals: list[dict[str, object]],
) -> tuple[str, list[tuple[str, str]]]:
    selected = []
    occupied: list[tuple[int, int]] = []
    ranked = sorted(
        proposals,
        key=lambda proposal: (
            bool(proposal["exact_normalized"]),
            float(proposal["style_density"]),
            float(proposal["similarity"]),
            -int(proposal["word_count"]),
        ),
        reverse=True,
    )
    for proposal in ranked:
        start = int(proposal["start"])
        end = int(proposal["end"])
        if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied):
            continue
        occupied.append((start, end))
        selected.append(proposal)

    corrected = str(text)
    changes = []
    for proposal in sorted(selected, key=lambda item: int(item["start"]), reverse=True):
        start = int(proposal["start"])
        end = int(proposal["end"])
        original = corrected[start:end]
        replacement = str(proposal["replacement"])
        corrected = corrected[:start] + replacement + corrected[end:]
        changes.append((original, replacement))
    changes.reverse()
    return corrected, changes


def reconcile_text(
    whisper_text: str,
    ocr_texts: list[str],
    *,
    mode: str = "balanced",
) -> tuple[str, list[tuple[str, str]]]:
    proposals = correction_proposals(whisper_text, ocr_texts, mode=mode)
    return apply_proposals(whisper_text, proposals)


def reconcile_transcripts(
    whisper_text: str,
    ocr_text: str,
    *,
    mode: str = "balanced",
    tolerance_seconds: int = 2,
) -> tuple[str, list[tuple[str, str]]]:
    if mode not in MODE_CUTOFFS:
        raise ValueError(f"Mode de correction invalide: {mode!r}")
    ocr_entries = parse_ocr_entries(ocr_text)
    plain_ocr_texts = [str(ocr_text).strip()] if not ocr_entries and str(ocr_text).strip() else []
    rendered = []
    corrections = []
    for raw_line in str(whisper_text).splitlines():
        match = WHISPER_LINE.match(raw_line)
        if not match:
            rendered.append(raw_line)
            continue
        start = parse_timecode(match.group("start"))
        end = parse_timecode(match.group("end") or match.group("start"))
        nearby_texts = (
            [
                str(entry["text"])
                for entry in ocr_entries
                if start - tolerance_seconds
                <= int(entry["second"])
                <= end + tolerance_seconds
            ]
            if ocr_entries
            else plain_ocr_texts
        )
        corrected_body, line_corrections = reconcile_text(
            match.group("body"),
            nearby_texts,
            mode=mode,
        )
        rendered.append(
            match.group("prefix")
            + (match.group("speaker") or "")
            + corrected_body
        )
        corrections.extend(line_corrections)
    suffix = "\n" if str(whisper_text).endswith("\n") or rendered else ""
    return "\n".join(rendered) + suffix, corrections


def reconcile_file(
    video_path: Path,
    *,
    force: bool = False,
    mode: str = "balanced",
) -> Path | None:
    whisper_source = whisper_source_path(video_path)
    ocr_source = ocr_source_path(video_path)
    target = corrected_path(video_path)
    report = corrections_path(video_path)
    if not whisper_source.exists():
        print(f"[skip] transcript WhisperX brut introuvable: {whisper_source}")
        return None

    dependencies = [whisper_source]
    if ocr_source is not None:
        dependencies.append(ocr_source)
    if (
        target.exists()
        and report.exists()
        and not force
        and output_is_current(target, dependencies)
        and output_is_current(report, dependencies)
    ):
        print(f"[skip] {target.name} existe deja")
        return target

    whisper_text = whisper_source.read_text(encoding="utf-8")
    if ocr_source is None:
        corrected_text = whisper_text
        corrections: list[tuple[str, str]] = []
        print("[warn] transcript OCR absent; WhisperX est conserve sans rapprochement")
    else:
        corrected_text, corrections = reconcile_transcripts(
            whisper_text,
            ocr_source.read_text(encoding="utf-8"),
            mode=mode,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(corrected_text, encoding="utf-8")
    report_lines = sorted({f"{source}\t{replacement}" for source, replacement in corrections})
    report.write_text(
        "\n".join(report_lines) + ("\n" if report_lines else ""),
        encoding="utf-8",
    )
    print(f"[ok] {target} ({len(corrections)} rapprochement(s) OCR)")
    print(f"[ok] {report}")
    return target
