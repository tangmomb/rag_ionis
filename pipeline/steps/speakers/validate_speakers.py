import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.support.json_io import read_json, write_json, write_jsonl
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
from pipeline.support.paths import existing_speakers_dir, relative_to_video_dir, speakers_dir


SPEAKER_CANDIDATES_NAME = "speaker_candidates.json"
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"
BATCH_STATE_NAME = "speaker_validation_batch_state.json"
BATCH_INPUT_NAME = "speaker_validation_batch_input.jsonl"
BATCH_OUTPUT_NAME = "speaker_validation_batch_output.jsonl"
BATCH_ERROR_NAME = "speaker_validation_batch_error.jsonl"
MAX_OUTPUT_TOKENS = 2048
MAX_LIVE_ATTEMPTS = 3
SYSTEM_PROMPT = (
    "Tu verifies une liste de speakers detectes automatiquement dans une video. "
    "Garde uniquement les noms qui designent vraiment des personnes physiques. "
    "Rejete les entreprises, ecoles, services, metiers, titres, lieux, slogans, URLs, "
    "mots OCR parasites et noms incomplets. Tu disposes du titre de la video et "
    "eventuellement d'un extrait du debut du transcript, de textes detectes par OCR "
    "et de candidats existants. "
    "Recherche les noms des personnes dans toutes ces sources, meme lorsqu'aucun "
    "candidat n'a encore ete propose. "
    "Pour chaque personne, retourne aussi son title, c'est-a-dire son poste, sa "
    "fonction ou son role. Recherche et extrais aussi explicitement dans les sources "
    "le nom de son entreprise ou organisation, puis inclus-le dans title. Le poste "
    "et l'entreprise sont souvent tres proches du nom de la personne. Si aucun title "
    "n'est fiable, utilise une chaine vide. "
    "Reponds uniquement avec l'objet JSON demande."
)
def candidates_path(video_path):
    return existing_speakers_dir(video_path) / SPEAKER_CANDIDATES_NAME


def validated_path(video_path):
    return speakers_dir(video_path) / SPEAKERS_VALIDATED_NAME


def batch_state_path(video_path):
    return speakers_dir(video_path) / BATCH_STATE_NAME


def batch_input_path(video_path):
    return speakers_dir(video_path) / BATCH_INPUT_NAME


def batch_output_path(video_path):
    return speakers_dir(video_path) / BATCH_OUTPUT_NAME


def batch_error_path(video_path):
    return speakers_dir(video_path) / BATCH_ERROR_NAME


def load_json(path):
    return read_json(path, encoding="utf-8-sig")


def normalize_model_name(model):
    compact = str(model).strip().lower().replace("_", "").replace("-", "")
    if compact == "gpt5.4nano":
        return "gpt-5.4-nano"
    return model


def normalize_name(name):
    normalized = unicodedata.normalize("NFKD", str(name).strip().casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^0-9a-z]+", "", normalized)


def apply_title_spelling(name, video_title):
    title_parts = re.findall(
        r"[^\W\d_]+(?:[-'’][^\W\d_]+)*",
        str(video_title),
        flags=re.UNICODE,
    )
    corrected_parts = []
    for name_part in str(name).split():
        normalized_part = normalize_name(name_part)
        if len(normalized_part) < 4 or not title_parts:
            corrected_parts.append(name_part)
            continue
        best_title_part = max(
            title_parts,
            key=lambda title_part: SequenceMatcher(
                None,
                normalized_part,
                normalize_name(title_part),
            ).ratio(),
        )
        similarity = SequenceMatcher(
            None,
            normalized_part,
            normalize_name(best_title_part),
        ).ratio()
        corrected_parts.append(best_title_part if similarity >= 0.9 else name_part)
    return " ".join(corrected_parts)


def preserve_candidate_name_parts(valid_speakers, candidates, video_title=""):
    candidate_names = [
        " ".join(str(item.get("name", "")).split()).strip()
        for item in candidates or []
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    ]
    preserved = []

    for valid_speaker in valid_speakers:
        valid_name = " ".join(str(valid_speaker).split()).strip()
        valid_parts = valid_name.split()
        if not valid_parts:
            preserved.append(valid_speaker)
            continue

        best_match = None
        for candidate_name in candidate_names:
            candidate_parts = candidate_name.split()
            if len(candidate_parts) <= len(valid_parts):
                continue

            available = set(range(len(candidate_parts)))
            replacements = {}
            similarities = []
            for valid_part in valid_parts:
                if not available:
                    break
                best_index = max(
                    available,
                    key=lambda index: SequenceMatcher(
                        None,
                        normalize_name(valid_part),
                        normalize_name(candidate_parts[index]),
                    ).ratio(),
                )
                similarity = SequenceMatcher(
                    None,
                    normalize_name(valid_part),
                    normalize_name(candidate_parts[best_index]),
                ).ratio()
                if similarity < 0.8:
                    break
                available.remove(best_index)
                replacements[best_index] = valid_part
                similarities.append(similarity)
            else:
                score = sum(similarities) / len(similarities)
                if best_match is None or score > best_match[0]:
                    best_match = (score, candidate_parts, replacements)

        if best_match is None:
            complete_name = valid_name
        else:
            _score, candidate_parts, replacements = best_match
            complete_name = " ".join(
                replacements.get(index, part)
                for index, part in enumerate(candidate_parts)
            )
        preserved.append(apply_title_spelling(complete_name, video_title))

    return preserved


def unique_speakers(payload):
    speakers = []
    seen = set()
    for speaker in payload.get("speakers", []) or []:
        speaker = " ".join(str(speaker).split()).strip()
        key = normalize_name(speaker)
        if not speaker or not key or key in seen:
            continue
        seen.add(key)
        speakers.append(speaker)
    return speakers


def candidate_entries(payload):
    entries = []
    for item in payload.get("candidates", []) or []:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("name", "")).split()).strip()
        methods = [
            str(method).strip()
            for method in item.get("methods", []) or []
            if str(method).strip()
        ]
        if name:
            entries.append({"name": name, "methods": methods})
    if entries:
        return entries
    return [{"name": name, "methods": []} for name in unique_speakers(payload)]


def validation_context(payload):
    return (
        str(payload.get("video_title") or "").strip(),
        candidate_entries(payload),
        [
            " ".join(str(text).split()).strip()
            for text in (
                payload.get("filtered_ocr_texts")
                or payload.get("raw_ocr_texts")
                or []
            )
            if " ".join(str(text).split()).strip()
        ],
        max(0, int(payload.get("expected_speaker_count") or 0)),
        str(payload.get("transcript_excerpt") or ""),
    )


def response_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    pieces = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(str(text))
    return "\n".join(pieces).strip()


def response_text_from_payload(payload):
    output_text = payload.get("output_text")
    if output_text:
        return str(output_text)

    pieces = []
    for output in payload.get("output", []) or []:
        for content in output.get("content", []) or []:
            text = content.get("text")
            if text:
                pieces.append(str(text))
    return "\n".join(pieces).strip()


def openai_client():
    from openai import OpenAI

    return OpenAI()


def response_diagnostics(response):
    incomplete_details = getattr(response, "incomplete_details", None)
    if hasattr(incomplete_details, "model_dump"):
        incomplete_details = incomplete_details.model_dump()
    error = getattr(response, "error", None)
    if hasattr(error, "model_dump"):
        error = error.model_dump()
    return {
        "status": getattr(response, "status", None),
        "incomplete_details": incomplete_details,
        "error": error,
        "output_types": [getattr(item, "type", None) for item in getattr(response, "output", []) or []],
    }


def response_diagnostics_from_payload(payload):
    return {
        "status": payload.get("status"),
        "incomplete_details": payload.get("incomplete_details"),
        "error": payload.get("error"),
        "output_types": [item.get("type") for item in payload.get("output", []) or []],
    }


def structured_output_config():
    return {
        "format": {
            "type": "json_schema",
            "name": "speaker_validation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "speakers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {"type": "string"},
                                "title": {"type": "string"},
                            },
                            "required": ["speaker", "title"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["speakers"],
                "additionalProperties": False,
            },
        }
    }


def build_response_request(
    model,
    speakers,
    video_title="",
    candidates=None,
    raw_ocr_texts=None,
    expected_speaker_count=0,
    transcript_excerpt="",
):
    candidate_details = [
        {"name": str(candidate.get("name") or "").strip()}
        for candidate in (candidates or [])
        if isinstance(candidate, dict) and str(candidate.get("name") or "").strip()
    ]
    if not candidate_details:
        candidate_details = [{"name": name} for name in speakers]
    user_prompt = (
        "Identifie les speakers qui sont vraiment des personnes physiques. "
        "Voici le titre de la video, un extrait du debut du transcript, les textes "
        "detectes via OCR dans la video, puis les speakers deja soupconnes. Recherche "
        "aussi les noms dans le titre et le transcript, et ajoute-les meme s'ils ne "
        "figurent pas encore parmi les candidats. Place uniquement les personnes "
        "physiques dans la reponse. "
        "Pour chaque personne, retourne speaker avec son nom et title avec son poste, "
        "sa fonction ou son role. Recherche et extrais explicitement le nom de "
        "l'entreprise ou de l'organisation dans le titre, le transcript ou les textes OCR, puis "
        "inclus-le dans title, par exemple 'CTO - Mappy.com'. N'omets pas l'entreprise "
        "lorsqu'elle est identifiable. Si tu ne peux pas confirmer le title, utilise "
        "une chaine vide. Reponds sans commentaire. "
        + "\n\n"
        + json.dumps(
            {
                "video_title": str(video_title).strip(),
                "transcript_excerpt": transcript_excerpt,
                "ocr_detected_texts": raw_ocr_texts or [],
                "candidates": candidate_details,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    request_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    body = {
        "model": model,
        "input": request_messages,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "text": structured_output_config(),
    }
    request_log = {
        "model": model,
        "messages": request_messages,
        "api": "responses.create",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "text": body["text"],
    }
    return body, request_log


def speaker_answer_error(answer):
    if not str(answer).strip():
        return "Reponse OpenAI vide."
    try:
        parse_speaker_details(answer)
    except ValueError as error:
        return str(error)
    return None


def response_is_incomplete(diagnostics):
    return (
        diagnostics.get("status") == "incomplete"
        or bool(diagnostics.get("incomplete_details"))
    )


def warn_retry(attempt, reason, diagnostics):
    if attempt >= MAX_LIVE_ATTEMPTS:
        return
    print(
        f"[warn] Reponse OpenAI inutilisable "
        f"(tentative {attempt}/{MAX_LIVE_ATTEMPTS}): {reason}; relance... "
        f"details={json.dumps(diagnostics, ensure_ascii=False)}",
        flush=True,
    )


def exhausted_response_error(reason, diagnostics):
    return RuntimeError(
        f"OpenAI n'a renvoye aucun JSON complet apres "
        f"{MAX_LIVE_ATTEMPTS} tentatives. "
        f"Derniere erreur: {reason}. "
        f"Details: {json.dumps(diagnostics, ensure_ascii=False)}"
    )


def ask_gpt(
    client,
    model,
    speakers,
    video_title="",
    candidates=None,
    raw_ocr_texts=None,
    expected_speaker_count=0,
    transcript_excerpt="",
):
    body, request_log = build_response_request(
        model,
        speakers,
        video_title,
        candidates,
        raw_ocr_texts,
        expected_speaker_count,
        transcript_excerpt,
    )
    if hasattr(client, "responses"):
        diagnostics = None
        failure_reason = "Reponse OpenAI vide."
        for attempt in range(1, MAX_LIVE_ATTEMPTS + 1):
            response = client.responses.create(**body)
            answer = response_text(response)
            diagnostics = response_diagnostics(response)
            if response_is_incomplete(diagnostics):
                failure_reason = "Reponse tronquee par la limite de sortie"
                warn_retry(attempt, failure_reason, diagnostics)
                continue
            failure_reason = speaker_answer_error(answer)
            if failure_reason is None:
                request_log["attempt_count"] = attempt
                return answer, request_log
            warn_retry(attempt, failure_reason, diagnostics)
        raise exhausted_response_error(failure_reason, diagnostics)

    request_log["api"] = "chat.completions.create"
    request_log["max_completion_tokens"] = body["max_output_tokens"]
    diagnostics = None
    failure_reason = "Reponse OpenAI vide."
    for attempt in range(1, MAX_LIVE_ATTEMPTS + 1):
        response = client.chat.completions.create(
            model=model,
            messages=body["input"],
            max_completion_tokens=body["max_output_tokens"],
        )
        choice = response.choices[0]
        answer = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        diagnostics = {"finish_reason": finish_reason}
        if finish_reason in {"length", "max_tokens"}:
            failure_reason = "Reponse tronquee par la limite de sortie"
            warn_retry(attempt, failure_reason, diagnostics)
            continue
        failure_reason = speaker_answer_error(answer)
        if failure_reason is None:
            request_log["attempt_count"] = attempt
            return answer, request_log
        warn_retry(attempt, failure_reason, diagnostics)
    raise exhausted_response_error(failure_reason, diagnostics)


def parse_speaker_details(answer):
    text = str(answer).strip()
    if not text:
        raise ValueError("Reponse OpenAI vide pendant la validation des speakers.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Reponse OpenAI non JSON: {text[:500]!r}") from error
    if isinstance(parsed, dict):
        parsed = parsed.get("speakers", parsed.get("valid_speakers"))
    if not isinstance(parsed, list):
        raise ValueError(f"Reponse GPT inattendue: {answer!r}")
    details = []
    for item in parsed:
        if isinstance(item, str):
            details.append({"speaker": item, "title": ""})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Speaker GPT inattendu: {item!r}")
        speaker = item.get("speaker", item.get("name"))
        if speaker is None:
            raise ValueError(f"Nom de speaker absent: {item!r}")
        details.append(
            {
                "speaker": str(speaker),
                "title": str(item.get("title") or ""),
            }
        )
    return details


def parse_valid_speakers(answer):
    return [detail["speaker"] for detail in parse_speaker_details(answer)]


def preserve_speaker_details(details, candidates, video_title=""):
    preserved_names = preserve_candidate_name_parts(
        [detail["speaker"] for detail in details],
        candidates,
        video_title,
    )
    return [
        {
            "speaker": preserved_name,
            "title": detail["title"],
        }
        for detail, preserved_name in zip(details, preserved_names)
    ]


def write_validation_output(
    model,
    video_path,
    source,
    target,
    speakers,
    valid_speakers,
    answer,
    request_log,
    video_title="",
    speaker_details=None,
):
    validated = {
        "source": relative_to_video_dir(source, video_path),
        "video_title": video_title,
        "speakers": valid_speakers,
        "speaker_details": speaker_details or [],
        "validation": {
            "model": model,
            "answer": answer,
            "reviewed": len(speakers),
            "kept": len(valid_speakers),
            "rejected": max(0, len(speakers) - len(valid_speakers)),
            "request": request_log if (speakers or video_title) else None,
        },
    }
    write_json(target, validated)
    print(
        f"[write] {target} ({len(valid_speakers)} speakers gardes, "
        f"{max(0, len(speakers) - len(valid_speakers))} rejetes)",
        flush=True,
    )
    return target


def validate_file_live(client, model, video_path, force=False):
    source = candidates_path(video_path)
    target = validated_path(video_path)
    if not source.exists():
        print(f"[skip] candidats speakers introuvables: {source}")
        return None
    if target.exists() and not force and target.stat().st_mtime >= source.stat().st_mtime:
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: candidats speakers plus recents")

    payload = load_json(source)
    speakers = unique_speakers(payload)
    (
        video_title,
        candidates,
        raw_ocr_texts,
        expected_speaker_count,
        transcript_excerpt,
    ) = validation_context(payload)
    no_speech = payload.get("status") == "no_speech"
    if no_speech:
        answer = "[]"
        valid_speakers = []
        speaker_details = []
        request_log = None
    elif speakers or video_title or raw_ocr_texts or transcript_excerpt:
        answer, request_log = ask_gpt(
            client,
            model,
            speakers,
            video_title,
            candidates,
            raw_ocr_texts,
            expected_speaker_count,
            transcript_excerpt,
        )
        answer = answer.strip()
        speaker_details = preserve_speaker_details(
            parse_speaker_details(answer),
            candidates,
            video_title,
        )
        valid_speakers = [detail["speaker"] for detail in speaker_details]
    else:
        body, request_log = build_response_request(
            model,
            speakers,
            video_title,
            candidates,
            raw_ocr_texts,
            expected_speaker_count,
            transcript_excerpt,
        )
        _ = body
        answer = "[]"
        valid_speakers = []
        speaker_details = []

    return write_validation_output(
        model,
        video_path,
        source,
        target,
        speakers,
        valid_speakers,
        answer,
        request_log,
        video_title,
        speaker_details,
    )


def speaker_batch_fingerprint(
    model,
    source,
    speakers,
    video_title,
    candidates,
    raw_ocr_texts,
    expected_speaker_count,
    transcript_excerpt,
):
    body, _request_log = build_response_request(
        model,
        speakers,
        video_title,
        candidates,
        raw_ocr_texts,
        expected_speaker_count,
        transcript_excerpt,
    )
    return batch_request_fingerprint(
        model,
        sources=(source,),
        custom_ids=("speaker-validation",),
        options={
            "workflow": "speaker_validation",
            "request_body": body,
        },
    )


def submit_batch_validation(
    video_path,
    model,
    speakers,
    video_title,
    candidates,
    raw_ocr_texts,
    expected_speaker_count,
    transcript_excerpt,
    *,
    request_fingerprint,
):
    client = openai_client()
    video_speakers_dir = speakers_dir(video_path)
    video_speakers_dir.mkdir(parents=True, exist_ok=True)
    input_path = batch_input_path(video_path)
    state_path = batch_state_path(video_path)
    body, _request_log = build_response_request(
        model,
        speakers,
        video_title,
        candidates,
        raw_ocr_texts,
        expected_speaker_count,
        transcript_excerpt,
    )

    write_jsonl(
        input_path,
        [
            {
            "custom_id": "speaker-validation",
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
            }
        ],
    )

    with input_path.open("rb") as batch_file:
        uploaded = client.files.create(file=batch_file, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "script": Path(sys.argv[0]).name,
            "model": model,
            "video": Path(video_path).name,
        },
    )
    state = {
        "mode": "batch",
        "model": model,
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": uploaded.id,
        "submitted_count": 1,
        "source": source_name_for_state(video_path),
        "request_fingerprint": request_fingerprint,
    }
    save_batch_state(state_path, state)
    print(f"[batch] submitted id={batch.id} status={batch.status} requests=1", flush=True)
    return state_path


def source_name_for_state(video_path):
    source = candidates_path(video_path)
    return source.name if source.exists() else SPEAKER_CANDIDATES_NAME


def finalize_batch_validation(video_path, model, source, target, speakers, state):
    client = openai_client()
    output_file_id = state.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch complete mais output_file_id absent.")

    download_batch_files(
        client,
        state,
        batch_output_path(video_path),
        batch_error_path(video_path),
    )
    record_by_id = records_by_custom_id(parse_jsonl(batch_output_path(video_path)))
    record = record_by_id.get("speaker-validation")
    if not record:
        raise RuntimeError("Resultat batch introuvable pour speaker-validation")

    response = record.get("response") or {}
    if response.get("status_code") != 200:
        raise RuntimeError(f"Batch speaker-validation en echec avec status={response.get('status_code')}")

    raw_payload = response.get("body") or {}
    answer = response_text_from_payload(raw_payload).strip()
    source_payload = load_json(source)
    (
        video_title,
        candidates,
        raw_ocr_texts,
        expected_speaker_count,
        transcript_excerpt,
    ) = validation_context(source_payload)
    if (speakers or video_title or raw_ocr_texts or transcript_excerpt) and not answer:
        diagnostics = response_diagnostics_from_payload(raw_payload)
        raise RuntimeError(
            "Le batch OpenAI n'a renvoye aucun texte pour speaker-validation. "
            f"Details: {json.dumps(diagnostics, ensure_ascii=False)}"
        )
    speaker_details = (
        preserve_speaker_details(
            parse_speaker_details(answer),
            candidates,
            video_title,
        )
        if answer
        else []
    )
    valid_speakers = [detail["speaker"] for detail in speaker_details]
    _body, request_log = build_response_request(
        model,
        speakers,
        video_title,
        candidates,
        raw_ocr_texts,
        expected_speaker_count,
        transcript_excerpt,
    )
    return write_validation_output(
        model,
        video_path,
        source,
        target,
        speakers,
        valid_speakers,
        answer or "[]",
        request_log,
        video_title,
        speaker_details,
    )


def validate_file_batch(model, video_path, force=False, wait=False, poll_interval_seconds=30):
    source = candidates_path(video_path)
    target = validated_path(video_path)
    if not source.exists():
        print(f"[skip] candidats speakers introuvables: {source}")
        return None
    if target.exists() and not force and target.stat().st_mtime >= source.stat().st_mtime:
        print(f"[skip] {target.name} existe deja")
        return target
    if target.exists() and not force:
        print(f"[regen] {target.name}: candidats speakers plus recents")
        force = True

    payload = load_json(source)
    speakers = unique_speakers(payload)
    (
        video_title,
        candidates,
        raw_ocr_texts,
        expected_speaker_count,
        transcript_excerpt,
    ) = validation_context(payload)
    no_speech = payload.get("status") == "no_speech"
    if no_speech or (
        not speakers
        and not video_title
        and not raw_ocr_texts
        and not transcript_excerpt
    ):
        _body, request_log = build_response_request(
            model,
            speakers,
            video_title,
            candidates,
            raw_ocr_texts,
            expected_speaker_count,
            transcript_excerpt,
        )
        return write_validation_output(
            model,
            video_path,
            source,
            target,
            speakers,
            [],
            "[]",
            request_log,
            video_title,
        )

    if force:
        for artifact_path in (
            target,
            batch_state_path(video_path),
            batch_input_path(video_path),
            batch_output_path(video_path),
            batch_error_path(video_path),
        ):
            if artifact_path.exists():
                artifact_path.unlink()

    state_path = batch_state_path(video_path)
    request_fingerprint = speaker_batch_fingerprint(
        model,
        source,
        speakers,
        video_title,
        candidates,
        raw_ocr_texts,
        expected_speaker_count,
        transcript_excerpt,
    )
    state = load_batch_state(state_path)
    if state is not None and not batch_state_matches(
        state,
        request_fingerprint,
    ):
        print(
            "[batch] etat existant incompatible; nouvelle soumission "
            f"(ancien batch_id={state.get('batch_id', 'inconnu')})",
            flush=True,
        )
        state = None
    if state is None:
        state_path = submit_batch_validation(
            video_path,
            model,
            speakers,
            video_title,
            candidates,
            raw_ocr_texts,
            expected_speaker_count,
            transcript_excerpt,
            request_fingerprint=request_fingerprint,
        )
        if not wait:
            return state_path
        state = load_batch_state(state_path)
        if state is None:
            raise RuntimeError("Etat batch introuvable apres la soumission.")

    client = openai_client()
    state = poll_batch_state(
        client,
        state,
        state_path,
        wait=wait,
        poll_interval_seconds=poll_interval_seconds,
        on_wait=lambda current, seconds: print(
            f"[batch] status={current['status']} batch_id={current['batch_id']} attente {seconds}s",
            flush=True,
        ),
    )

    if not is_terminal_batch_status(state["status"]):
        print(f"[batch] status={state['status']} batch_id={state['batch_id']}", flush=True)
        return batch_state_path(video_path)

    if state["status"] != COMPLETED_BATCH_STATUS:
        raise RuntimeError(f"Batch termine avec statut non supporte: {state['status']}")

    return finalize_batch_validation(video_path, model, source, target, speakers, state)
