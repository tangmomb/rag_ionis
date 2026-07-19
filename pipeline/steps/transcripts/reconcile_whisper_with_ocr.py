from __future__ import annotations

import json
import os
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
DEFAULT_CORRECTION_MODEL = os.getenv(
    "TRANSCRIPT_CORRECTION_MODEL",
    "gpt-5.6-luna",
)
WHISPER_LINE = re.compile(
    r"^(?P<prefix>\[(?P<start>(?:\d{2}:)?\d{2}:\d{2})"
    r"(?:-(?P<end>(?:\d{2}:)?\d{2}:\d{2}))?\]\s*)"
    r"(?P<speaker>SPEAKER_\d+\s*:\s*)?"
    r"(?P<body>.*)$"
)
WORD_TOKEN = re.compile(
    r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+"
    r"(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*"
)


def compact_normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^0-9a-z]+", "", without_marks)


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


def _response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    pieces = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(str(text))
    return "\n".join(pieces).strip()


def luna_request(
    model: str,
    whisper_segments: list[dict[str, object]],
    ocr_text: str,
) -> dict[str, object]:
    return {
        "model": model or DEFAULT_CORRECTION_MODEL,
        "input": [
            {
                "role": "system",
                "content": (
                    "Tu corriges une transcription WhisperX en francais a partir "
                    "des sous-titres OCR de la meme video. Le transcript WhisperX "
                    "reste la structure canonique et l'OCR sert de reference de "
                    "correction. Corrige les noms propres, marques, mots mal "
                    "entendus, mots manquants, accords et pluriels lorsque l'OCR "
                    "permet de les etablir. Ne paraphrase pas, ne resume pas et "
                    "n'ajoute aucun texte OCR qui ne correspond pas a une parole. "
                    "Conserve tous les segments, leurs index et leur ordre. "
                    "Retourne uniquement l'objet JSON demande."
                ),
            },
            {
                "role": "user",
                "content": (
                    "SEGMENTS WHISPERX:\n"
                    + json.dumps(whisper_segments, ensure_ascii=False)
                    + "\n\nTRANSCRIPT OCR:\n"
                    + str(ocr_text).strip()
                ),
            },
        ],
        "max_output_tokens": min(
            32768,
            max(2048, sum(len(str(item["text"])) for item in whisper_segments) // 2),
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "corrected_whisper_segments",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": {"type": "integer"},
                                    "text": {"type": "string"},
                                },
                                "required": ["index", "text"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["segments"],
                    "additionalProperties": False,
                },
            }
        },
    }


def word_level_corrections(
    source: str,
    corrected: str,
) -> list[tuple[str, str]]:
    source_words = [match.group(0) for match in WORD_TOKEN.finditer(str(source))]
    corrected_words = [match.group(0) for match in WORD_TOKEN.finditer(str(corrected))]
    matcher = SequenceMatcher(
        None,
        [compact_normalize(word) for word in source_words],
        [compact_normalize(word) for word in corrected_words],
    )
    changes = []
    for operation, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        source_text = " ".join(source_words[source_start:source_end]) or "∅"
        target_text = " ".join(corrected_words[target_start:target_end]) or "∅"
        changes.append((source_text, target_text))
    return changes


def reconcile_transcripts_with_luna(
    client: object,
    model: str,
    whisper_text: str,
    ocr_text: str,
) -> tuple[str, list[tuple[str, str]]]:
    rendered_lines = str(whisper_text).splitlines()
    parsed_lines: list[tuple[int, re.Match[str]]] = []
    whisper_segments = []
    for line_index, raw_line in enumerate(rendered_lines):
        match = WHISPER_LINE.match(raw_line)
        if not match:
            continue
        segment_index = len(whisper_segments)
        parsed_lines.append((line_index, match))
        whisper_segments.append(
            {
                "index": segment_index,
                "text": match.group("body"),
            }
        )
    if not whisper_segments:
        return whisper_text, []

    response = client.responses.create(
        **luna_request(model, whisper_segments, ocr_text)
    )
    raw_response = _response_text(response)
    if not raw_response:
        raise RuntimeError("Luna n'a renvoye aucune correction de transcript.")
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Reponse Luna non JSON: {raw_response[:300]!r}"
        ) from error
    returned_segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(returned_segments, list):
        raise RuntimeError("Reponse Luna invalide: liste segments absente.")

    corrected_by_index = {}
    for item in returned_segments:
        if not isinstance(item, dict) or not isinstance(item.get("index"), int):
            raise RuntimeError("Reponse Luna invalide: index de segment absent.")
        index = item["index"]
        text = item.get("text")
        if not 0 <= index < len(whisper_segments):
            raise RuntimeError("Reponse Luna invalide: index de segment inconnu.")
        if (
            index in corrected_by_index
            or not isinstance(text, str)
            or (str(whisper_segments[index]["text"]).strip() and not text.strip())
        ):
            raise RuntimeError("Reponse Luna invalide: segment incomplet ou duplique.")
        corrected_by_index[index] = text.strip()
    expected_indexes = set(range(len(whisper_segments)))
    if set(corrected_by_index) != expected_indexes:
        raise RuntimeError("Reponse Luna invalide: des segments WhisperX manquent.")

    corrections = []
    for segment_index, (line_index, match) in enumerate(parsed_lines):
        corrected_body = corrected_by_index[segment_index]
        corrections.extend(
            word_level_corrections(match.group("body"), corrected_body)
        )
        rendered_lines[line_index] = (
            match.group("prefix")
            + (match.group("speaker") or "")
            + corrected_body
        )
    suffix = "\n" if str(whisper_text).endswith("\n") or rendered_lines else ""
    return "\n".join(rendered_lines) + suffix, corrections


def reconcile_file(
    video_path: Path,
    *,
    force: bool = False,
    client: object | None = None,
    model: str = DEFAULT_CORRECTION_MODEL,
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
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        corrected_text, corrections = reconcile_transcripts_with_luna(
            client,
            model,
            whisper_text,
            ocr_source.read_text(encoding="utf-8"),
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
