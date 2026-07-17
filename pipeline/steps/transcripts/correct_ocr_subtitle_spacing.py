import json
import re
from pathlib import Path

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
from pipeline.support.paths import existing_transcripts_dir, relative_to_video_dir


MAX_OUTPUT_TOKENS = 256
SOURCE_NAME = "ocr_subtitles_timecoded.txt"
TARGET_NAME = "ocr_subtitles_timecoded_corrected.txt"
BATCH_STATE_NAME = "ocr_spacing_batch_state.json"
BATCH_INPUT_NAME = "ocr_spacing_batch_input.jsonl"
BATCH_OUTPUT_NAME = "ocr_spacing_batch_output.jsonl"
BATCH_ERROR_NAME = "ocr_spacing_batch_error.jsonl"
LEGACY_SOURCE_SUFFIX = "_ocr_subtitle_timecodes.txt"
LEGACY_TARGET_SUFFIX = "_ocr_subtitle_timecodes_corrected.txt"
SUBTITLE_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
SYSTEM_PROMPT = (
    "Tu corriges uniquement les espaces manquants ou superflus dans une ligne de sous-titre OCR en francais. "
    "Ne change pas les mots, ne corrige pas l'orthographe, ne reformule pas, ne modifies pas la ponctuation sauf si elle sert strictement a remettre un espace evident. "
    "Si le texte est deja correct, renvoie-le tel quel. "
    "Reponds uniquement avec la ligne corrigee, sans guillemets ni commentaire."
)


def ocr_transcripts_dir(
    video_path,
    *,
    transcripts_dir_name="transcripts_ocr",
):
    return existing_transcripts_dir(video_path, name=transcripts_dir_name)


def subtitle_source_path(
    video_path,
    *,
    transcripts_dir_name="transcripts_ocr",
):
    transcript_dir = ocr_transcripts_dir(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    preferred = transcript_dir / SOURCE_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_SOURCE_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def subtitle_target_path(
    video_path,
    *,
    transcripts_dir_name="transcripts_ocr",
):
    transcript_dir = ocr_transcripts_dir(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    preferred = transcript_dir / TARGET_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_TARGET_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def batch_state_path(video_path, *, transcripts_dir_name="transcripts_ocr"):
    return (
        ocr_transcripts_dir(
            video_path,
            transcripts_dir_name=transcripts_dir_name,
        )
        / BATCH_STATE_NAME
    )

def batch_input_path(video_path, *, transcripts_dir_name="transcripts_ocr"):
    return (
        ocr_transcripts_dir(
            video_path,
            transcripts_dir_name=transcripts_dir_name,
        )
        / BATCH_INPUT_NAME
    )


def batch_output_path(video_path, *, transcripts_dir_name="transcripts_ocr"):
    return (
        ocr_transcripts_dir(
            video_path,
            transcripts_dir_name=transcripts_dir_name,
        )
        / BATCH_OUTPUT_NAME
    )


def batch_error_path(video_path, *, transcripts_dir_name="transcripts_ocr"):
    return (
        ocr_transcripts_dir(
            video_path,
            transcripts_dir_name=transcripts_dir_name,
        )
        / BATCH_ERROR_NAME
    )


def openai_client():
    from openai import OpenAI

    return OpenAI()


def response_text(response):
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


def response_text_from_payload(payload):
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


def build_response_request(model, text):
    user_prompt = (
        "Corrige uniquement les espaces manquants ou en trop dans cette ligne. "
        "Si rien n'est a changer, renvoie exactement le meme texte.\n\n"
        f"Ligne: {text}"
    )
    request_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return (
        {
            "model": model,
            "input": request_messages,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        {
            "model": model,
            "messages": request_messages,
            "api": "responses.create",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )


def request_spacing_fix(client, model, text):
    body, _request_log = build_response_request(model, text)
    if hasattr(client, "responses"):
        response = client.responses.create(**body)
        return response_text(response)

    response = client.chat.completions.create(
        model=model,
        messages=body["input"],
        max_completion_tokens=body["max_output_tokens"],
    )
    return (response.choices[0].message.content or "").strip()


def normalize_answer(text):
    answer = str(text).strip()
    if answer.startswith("```"):
        answer = re.sub(r"^```(?:text)?\s*", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\s*```$", "", answer)
    answer = answer.splitlines()[0].strip() if answer else ""
    return " ".join(answer.split())


def corrected_line(prefix, original_text, corrected_text):
    final_text = normalize_answer(corrected_text) or " ".join(str(original_text).split())
    return f"[{prefix}] {final_text}".rstrip()


def prepare_line_jobs(lines):
    jobs = []
    output_lines = []
    for index, line in enumerate(lines, start=1):
        match = SUBTITLE_LINE.match(line)
        if not match:
            output_lines.append(line.strip() if line.strip() else "")
            continue
        timecode, text = match.groups()
        cleaned_text = " ".join(str(text).split()).strip()
        if not cleaned_text:
            output_lines.append(f"[{timecode}]")
            continue
        job = {
            "index": index,
            "custom_id": f"line-{index:05d}",
            "timecode": timecode,
            "text": cleaned_text,
        }
        jobs.append(job)
        output_lines.append(job)
    return jobs, output_lines


def write_corrected_output(target, output_lines):
    rendered_lines = []
    for item in output_lines:
        if isinstance(item, dict):
            rendered_lines.append(corrected_line(item["timecode"], item["text"], item.get("corrected_text", item["text"])))
        else:
            rendered_lines.append(str(item))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rendered_lines).strip() + "\n", encoding="utf-8")
    print(f"[ok] {target}")
    return target


def process_video_live(
    client,
    model,
    video_path,
    force=False,
    *,
    transcripts_dir_name="transcripts_ocr",
):
    source = subtitle_source_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    target = subtitle_target_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] sous-titres OCR timecodes introuvables: {source}")
        return None

    lines = source.read_text(encoding="utf-8").splitlines()
    jobs, output_lines = prepare_line_jobs(lines)
    jobs_by_id = {job["custom_id"]: job for job in jobs}
    changed_count = 0
    request_count = 0

    for job in jobs:
        fixed_text = request_spacing_fix(client, model, job["text"])
        normalized_fixed = normalize_answer(fixed_text)
        if normalized_fixed != job["text"]:
            changed_count += 1
        jobs_by_id[job["custom_id"]]["corrected_text"] = normalized_fixed
        request_count += 1
        print(f"[line {job['index']}/{len(lines)}] {video_path.name}", flush=True)

    return write_corrected_output(target, output_lines)


def spacing_batch_fingerprint(model, source, jobs):
    return batch_request_fingerprint(
        model,
        sources=(source,),
        custom_ids=(job["custom_id"] for job in jobs),
        options={
            "workflow": "ocr_subtitle_spacing",
            "system_prompt": SYSTEM_PROMPT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )


def submit_batch_spacing(
    video_path,
    model,
    jobs,
    *,
    request_fingerprint,
    transcripts_dir_name="transcripts_ocr",
):
    client = openai_client()
    transcript_dir = ocr_transcripts_dir(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    transcript_dir.mkdir(parents=True, exist_ok=True)
    input_path = batch_input_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    state_path = batch_state_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )

    records = []
    for job in jobs:
        body, _request_log = build_response_request(model, job["text"])
        records.append(
            {
                "custom_id": job["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )
    write_jsonl(input_path, records)

    with input_path.open("rb") as batch_file:
        uploaded = client.files.create(file=batch_file, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "script": "16_OCR_correct_ocr_subtitle_spacing.py",
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
        "submitted_count": len(jobs),
        "source": relative_to_video_dir(
            subtitle_source_path(
                video_path,
                transcripts_dir_name=transcripts_dir_name,
            ),
            video_path,
        ),
        "request_fingerprint": request_fingerprint,
    }
    save_batch_state(state_path, state)
    print(f"[batch] submitted id={batch.id} status={batch.status} requests={len(jobs)}", flush=True)
    return state_path


def finalize_batch_spacing(
    video_path,
    model,
    source,
    target,
    jobs,
    output_lines,
    state,
    *,
    transcripts_dir_name="transcripts_ocr",
):
    client = openai_client()
    output_file_id = state.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch complete mais output_file_id absent.")

    download_batch_files(
        client,
        state,
        batch_output_path(
            video_path,
            transcripts_dir_name=transcripts_dir_name,
        ),
        batch_error_path(
            video_path,
            transcripts_dir_name=transcripts_dir_name,
        ),
    )
    record_by_id = records_by_custom_id(
        parse_jsonl(
            batch_output_path(
                video_path,
                transcripts_dir_name=transcripts_dir_name,
            )
        )
    )
    jobs_by_id = {job["custom_id"]: job for job in jobs}
    changed_count = 0
    request_count = 0

    for job in jobs:
        record = record_by_id.get(job["custom_id"])
        if not record:
            raise RuntimeError(f"Resultat batch introuvable pour {job['custom_id']}")
        response = record.get("response") or {}
        if response.get("status_code") != 200:
            raise RuntimeError(f"Batch {job['custom_id']} en echec avec status={response.get('status_code')}")
        raw_payload = response.get("body") or {}
        normalized_fixed = normalize_answer(response_text_from_payload(raw_payload))
        if normalized_fixed != job["text"]:
            changed_count += 1
        jobs_by_id[job["custom_id"]]["corrected_text"] = normalized_fixed
        request_count += 1
        print(f"[batch-line {job['index']}/{len(jobs)}] {video_path.name}", flush=True)

    return write_corrected_output(target, output_lines)


def process_video_batch(
    model,
    video_path,
    force=False,
    wait=False,
    poll_interval_seconds=30,
    *,
    transcripts_dir_name="transcripts_ocr",
):
    source = subtitle_source_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    target = subtitle_target_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] sous-titres OCR timecodes introuvables: {source}")
        return None

    lines = source.read_text(encoding="utf-8").splitlines()
    jobs, output_lines = prepare_line_jobs(lines)
    if not jobs:
        return write_corrected_output(target, output_lines)

    if force:
        for artifact_path in (
            target,
            batch_state_path(
                video_path,
                transcripts_dir_name=transcripts_dir_name,
            ),
            batch_input_path(
                video_path,
                transcripts_dir_name=transcripts_dir_name,
            ),
            batch_output_path(
                video_path,
                transcripts_dir_name=transcripts_dir_name,
            ),
            batch_error_path(
                video_path,
                transcripts_dir_name=transcripts_dir_name,
            ),
        ):
            if artifact_path.exists():
                artifact_path.unlink()

    state_path = batch_state_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    request_fingerprint = spacing_batch_fingerprint(model, source, jobs)
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
        state_path = submit_batch_spacing(
            video_path,
            model,
            jobs,
            request_fingerprint=request_fingerprint,
            transcripts_dir_name=transcripts_dir_name,
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
        return batch_state_path(
            video_path,
            transcripts_dir_name=transcripts_dir_name,
        )

    if state["status"] != COMPLETED_BATCH_STATUS:
        raise RuntimeError(f"Batch termine avec statut non supporte: {state['status']}")

    return finalize_batch_spacing(
        video_path,
        model,
        source,
        target,
        jobs,
        output_lines,
        state,
        transcripts_dir_name=transcripts_dir_name,
    )
