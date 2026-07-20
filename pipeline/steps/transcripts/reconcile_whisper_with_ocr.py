from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

from pipeline.support.json_io import write_jsonl
from pipeline.support.openai_batch import (
    COMPLETED_BATCH_STATUS,
    batch_request_fingerprint,
    batch_state_matches,
    download_batch_files,
    is_terminal_batch_status,
    load_batch_state,
    parse_jsonl,
    poll_batch_state,
    records_by_custom_id,
    save_batch_state,
)
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
BATCH_STATE_NAME = "transcript_reconciliation_batch_state.json"
BATCH_INPUT_NAME = "transcript_reconciliation_batch_input.jsonl"
BATCH_OUTPUT_NAME = "transcript_reconciliation_batch_output.jsonl"
BATCH_ERROR_NAME = "transcript_reconciliation_batch_error.jsonl"
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


def _response_text_from_payload(payload: dict[str, object]) -> str:
    output_text = payload.get("output_text")
    if output_text:
        return str(output_text).strip()
    pieces = []
    for output in payload.get("output", []) or []:
        for content in output.get("content", []) or []:
            text = content.get("text")
            if text:
                pieces.append(str(text))
    return "\n".join(pieces).strip()


def batch_artifact_path(video_path: Path, name: str) -> Path:
    return corrected_path(video_path).parent / name


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


def _write_reconciliation(
    target: Path,
    report: Path,
    corrected_text: str,
    corrections: list[tuple[str, str]],
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(corrected_text, encoding="utf-8")
    report_lines = sorted(
        {f"{source}\t{replacement}" for source, replacement in corrections}
    )
    report.write_text(
        "\n".join(report_lines) + ("\n" if report_lines else ""),
        encoding="utf-8",
    )
    print(f"[ok] {target} ({len(corrections)} rapprochement(s) OCR)")
    print(f"[ok] {report}")
    return target


def reconcile_file_batch(
    video_path: Path,
    *,
    force: bool = False,
    model: str = DEFAULT_CORRECTION_MODEL,
    wait: bool = True,
    poll_interval_seconds: float = 30,
    reset_batch: bool | None = None,
) -> Path | None:
    from openai import OpenAI

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
        print("[warn] transcript OCR absent; WhisperX est conserve sans rapprochement")
        return _write_reconciliation(target, report, whisper_text, [])

    ocr_text = ocr_source.read_text(encoding="utf-8")
    segments = []
    for raw_line in whisper_text.splitlines():
        match = WHISPER_LINE.match(raw_line)
        if match:
            segments.append(
                {"index": len(segments), "text": match.group("body")}
            )
    if not segments:
        return _write_reconciliation(target, report, whisper_text, [])

    body = luna_request(model, segments, ocr_text)
    state_path = batch_artifact_path(video_path, BATCH_STATE_NAME)
    input_path = batch_artifact_path(video_path, BATCH_INPUT_NAME)
    output_path = batch_artifact_path(video_path, BATCH_OUTPUT_NAME)
    error_path = batch_artifact_path(video_path, BATCH_ERROR_NAME)
    if reset_batch is None:
        reset_batch = force
    if reset_batch:
        for path in (state_path, input_path, output_path, error_path):
            path.unlink(missing_ok=True)

    request_fingerprint = batch_request_fingerprint(
        model or DEFAULT_CORRECTION_MODEL,
        sources=(whisper_source, ocr_source),
        custom_ids=("transcript-reconciliation",),
        options={"workflow": "transcript_reconciliation", "request_body": body},
    )
    state = load_batch_state(state_path)
    if state is not None and not batch_state_matches(state, request_fingerprint):
        print(
            "[batch] etat reconciliation incompatible; nouvelle soumission "
            f"(ancien batch_id={state.get('batch_id', 'inconnu')})",
            flush=True,
        )
        state = None
    if state is None:
        write_jsonl(
            input_path,
            [{
                "custom_id": "transcript-reconciliation",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }],
        )
        client = OpenAI()
        with input_path.open("rb") as batch_file:
            uploaded = client.files.create(file=batch_file, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "workflow": "transcript_reconciliation",
                "model": model or DEFAULT_CORRECTION_MODEL,
                "video": Path(video_path).name,
            },
        )
        state = {
            "mode": "batch",
            "model": model or DEFAULT_CORRECTION_MODEL,
            "batch_id": batch.id,
            "status": batch.status,
            "input_file_id": uploaded.id,
            "submitted_count": 1,
            "request_fingerprint": request_fingerprint,
        }
        save_batch_state(state_path, state)
        print(
            f"[batch] reconciliation submitted id={batch.id} "
            f"status={batch.status} requests=1",
            flush=True,
        )
        if not wait:
            return state_path

    client = OpenAI()
    state = poll_batch_state(
        client,
        state,
        state_path,
        wait=wait,
        poll_interval_seconds=poll_interval_seconds,
        on_wait=lambda current, seconds: print(
            f"[batch] reconciliation status={current['status']} "
            f"batch_id={current['batch_id']} attente {seconds}s",
            flush=True,
        ),
    )
    if not is_terminal_batch_status(state["status"]):
        return state_path
    if state["status"] != COMPLETED_BATCH_STATUS:
        raise RuntimeError(
            "Batch de reconciliation termine avec statut non supporte: "
            f"{state['status']}"
        )
    if not state.get("output_file_id"):
        raise RuntimeError(
            "Batch de reconciliation complete mais output_file_id absent."
        )

    download_batch_files(client, state, output_path, error_path)
    record = records_by_custom_id(parse_jsonl(output_path)).get(
        "transcript-reconciliation"
    )
    if record is None:
        raise RuntimeError(
            "Resultat batch introuvable pour transcript-reconciliation."
        )
    response = record.get("response") or {}
    if response.get("status_code") != 200:
        raise RuntimeError(
            "Batch transcript-reconciliation en echec avec "
            f"status={response.get('status_code')}"
        )
    raw_response = _response_text_from_payload(response.get("body") or {})
    static_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(output_text=raw_response)
        )
    )
    corrected_text, corrections = reconcile_transcripts_with_luna(
        static_client,
        model,
        whisper_text,
        ocr_text,
    )
    return _write_reconciliation(target, report, corrected_text, corrections)


def reconcile_file(
    video_path: Path,
    *,
    force: bool = False,
    client: object | None = None,
    model: str = DEFAULT_CORRECTION_MODEL,
    mode: str = "live",
    reset_batch: bool | None = None,
) -> Path | None:
    if mode == "batch":
        return reconcile_file_batch(
            video_path,
            force=force,
            model=model,
            wait=True,
            reset_batch=reset_batch,
        )
    if mode != "live":
        raise ValueError(f"Mode de reconciliation invalide: {mode!r}")
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

    return _write_reconciliation(target, report, corrected_text, corrections)
