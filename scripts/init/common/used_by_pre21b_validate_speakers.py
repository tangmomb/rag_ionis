import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from common.pipeline_analysis import update_analysed_infos
from common.pipeline_paths import existing_speakers_dir, relative_to_video_dir, speakers_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SPEAKER_CANDIDATES_NAME = "speaker_candidates.json"
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"
BATCH_STATE_NAME = "speaker_validation_batch_state.json"
BATCH_INPUT_NAME = "speaker_validation_batch_input.jsonl"
BATCH_OUTPUT_NAME = "speaker_validation_batch_output.jsonl"
BATCH_ERROR_NAME = "speaker_validation_batch_error.jsonl"
DEFAULT_WAIT_FOR_BATCH = True
DEFAULT_MODEL = "gpt-5.4-nano"
MAX_OUTPUT_TOKENS = 512
MAX_LIVE_ATTEMPTS = 3
SYSTEM_PROMPT = (
    "Tu verifies une liste de speakers detectes automatiquement dans une video. "
    "Garde uniquement les noms qui designent vraiment des personnes physiques. "
    "Rejete les entreprises, ecoles, services, metiers, titres, lieux, slogans, URLs, "
    "mots OCR parasites et noms incomplets. Tu as aussi le titre de la video pour voir "
    "si un speaker s'y trouve. Si un speaker candidat ressemble beaucoup a celui "
    "dans le titre, le titre prevaut: utilise l'orthographe complete du titre. "
    "Reponds uniquement avec l'objet JSON demande."
)
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


def current_openai_mode():
    normalized = str(os.getenv("PIPELINE_OPENAI_MODE", "normal")).strip().lower()
    if normalized in {"batch", "normal"}:
        return normalized
    return "normal"


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
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_model_name(model):
    compact = str(model).strip().lower().replace("_", "").replace("-", "")
    if compact == "gpt5.4nano":
        return "gpt-5.4-nano"
    return model


def normalize_name(name):
    normalized = unicodedata.normalize("NFKD", str(name).strip().casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^0-9a-z]+", "", normalized)


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
                    "valid_speakers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    }
                },
                "required": ["valid_speakers"],
                "additionalProperties": False,
            },
        }
    }


def build_response_request(model, speakers, video_title="", candidates=None):
    candidate_details = candidates or [{"name": name, "methods": []} for name in speakers]
    user_prompt = (
        "Parmi cette liste de speakers, lesquels sont vraiment des personnes ? "
        "Place uniquement les noms valides dans valid_speakers, sans commentaire. "
        "Tu as aussi le titre de la video pour voir si un speaker s'y trouve. "
        "Si un candidat ressemble beaucoup a un nom dans le titre, renvoie l'orthographe "
        "du titre, qui prevaut.\n\n"
        + json.dumps(
            {
                "video_title": video_title,
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


def ask_gpt(client, model, speakers, video_title="", candidates=None):
    body, request_log = build_response_request(model, speakers, video_title, candidates)
    if hasattr(client, "responses"):
        diagnostics = None
        for attempt in range(1, MAX_LIVE_ATTEMPTS + 1):
            response = client.responses.create(**body)
            answer = response_text(response)
            diagnostics = response_diagnostics(response)
            if answer:
                request_log["attempt_count"] = attempt
                return answer, request_log
            if attempt < MAX_LIVE_ATTEMPTS:
                print(
                    f"[warn] Reponse OpenAI vide (tentative {attempt}/{MAX_LIVE_ATTEMPTS}); relance... "
                    f"details={json.dumps(diagnostics, ensure_ascii=False)}",
                    flush=True,
                )
        raise RuntimeError(
            f"OpenAI n'a renvoye aucun texte apres {MAX_LIVE_ATTEMPTS} tentatives. "
            f"Details: {json.dumps(diagnostics, ensure_ascii=False)}"
        )

    response = client.chat.completions.create(
        model=model,
        messages=body["input"],
        max_completion_tokens=body["max_output_tokens"],
    )
    answer = response.choices[0].message.content or ""
    request_log["api"] = "chat.completions.create"
    request_log["max_completion_tokens"] = body["max_output_tokens"]
    if not answer.strip():
        raise RuntimeError("OpenAI n'a renvoye aucun texte via chat.completions.create.")
    return answer, request_log


def parse_valid_speakers(answer):
    text = str(answer).strip()
    if not text:
        raise ValueError("Reponse OpenAI vide pendant la validation des speakers.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("["):
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            text = match.group(0)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Reponse OpenAI non JSON: {text[:500]!r}") from error
    if isinstance(parsed, dict):
        parsed = parsed.get("valid_speakers")
    if not isinstance(parsed, list):
        raise ValueError(f"Reponse GPT inattendue: {answer!r}")
    return parsed


def save_batch_state(path, state):
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_batch_state(video_path):
    path = batch_state_path(video_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_batch_state(video_path, client, state):
    batch = client.batches.retrieve(state["batch_id"])
    state.update(
        {
            "status": batch.status,
            "input_file_id": getattr(batch, "input_file_id", state.get("input_file_id")),
            "output_file_id": getattr(batch, "output_file_id", state.get("output_file_id")),
            "error_file_id": getattr(batch, "error_file_id", state.get("error_file_id")),
        }
    )
    request_counts = getattr(batch, "request_counts", None)
    if request_counts is not None:
        state["request_counts"] = request_counts.model_dump() if hasattr(request_counts, "model_dump") else dict(request_counts)
    save_batch_state(batch_state_path(video_path), state)
    return state


def is_terminal_batch_status(status):
    return status in {"completed", "failed", "expired", "cancelled"}


def parse_batch_lines(path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


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
):
    validated = {
        "source": relative_to_video_dir(source, video_path),
        "video_title": video_title,
        "speakers": valid_speakers,
        "validation": {
            "model": model,
            "answer": answer,
            "reviewed": len(speakers),
            "kept": len(valid_speakers),
            "rejected": max(0, len(speakers) - len(valid_speakers)),
            "request": request_log if speakers else None,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(validated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "validate_speakers",
        {
            "status": "done",
            "validated_file": relative_to_video_dir(target, video_path),
            "speakers": valid_speakers,
            "reviewed": len(speakers),
            "rejected": max(0, len(speakers) - len(valid_speakers)),
        },
    )
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
    video_title, candidates = validation_context(payload)
    if speakers:
        answer, request_log = ask_gpt(client, model, speakers, video_title, candidates)
        answer = answer.strip()
        valid_speakers = parse_valid_speakers(answer)
    else:
        body, request_log = build_response_request(model, speakers, video_title, candidates)
        _ = body
        answer = "[]"
        valid_speakers = []

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
    )


def submit_batch_validation(video_path, model, speakers, video_title, candidates):
    client = openai_client()
    video_speakers_dir = speakers_dir(video_path)
    video_speakers_dir.mkdir(parents=True, exist_ok=True)
    input_path = batch_input_path(video_path)
    state_path = batch_state_path(video_path)
    body, _request_log = build_response_request(model, speakers, video_title, candidates)

    with input_path.open("w", encoding="utf-8") as handle:
        record = {
            "custom_id": "speaker-validation",
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

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

    client.files.content(output_file_id).write_to_file(batch_output_path(video_path))

    error_file_id = state.get("error_file_id")
    if error_file_id:
        client.files.content(error_file_id).write_to_file(batch_error_path(video_path))

    records = parse_batch_lines(batch_output_path(video_path))
    record_by_id = {record.get("custom_id"): record for record in records if record.get("custom_id")}
    record = record_by_id.get("speaker-validation")
    if not record:
        raise RuntimeError("Resultat batch introuvable pour speaker-validation")

    response = record.get("response") or {}
    if response.get("status_code") != 200:
        raise RuntimeError(f"Batch speaker-validation en echec avec status={response.get('status_code')}")

    raw_payload = response.get("body") or {}
    answer = response_text_from_payload(raw_payload).strip()
    if speakers and not answer:
        diagnostics = response_diagnostics_from_payload(raw_payload)
        raise RuntimeError(
            "Le batch OpenAI n'a renvoye aucun texte pour speaker-validation. "
            f"Details: {json.dumps(diagnostics, ensure_ascii=False)}"
        )
    source_payload = load_json(source)
    video_title, candidates = validation_context(source_payload)
    valid_speakers = (
        parse_valid_speakers(answer)
        if speakers
        else []
    )
    _body, request_log = build_response_request(model, speakers, video_title, candidates)
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
    video_title, candidates = validation_context(payload)
    if not speakers:
        _body, request_log = build_response_request(model, speakers, video_title, candidates)
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

    state = load_batch_state(video_path)
    if state is None:
        state_path = submit_batch_validation(
            video_path,
            model,
            speakers,
            video_title,
            candidates,
        )
        if not wait:
            return state_path
        state = load_batch_state(video_path)

    client = openai_client()
    state = refresh_batch_state(video_path, client, state)
    while wait and not is_terminal_batch_status(state["status"]):
        print(f"[batch] status={state['status']} batch_id={state['batch_id']} attente {poll_interval_seconds}s", flush=True)
        time.sleep(poll_interval_seconds)
        state = refresh_batch_state(video_path, client, state)

    if not is_terminal_batch_status(state["status"]):
        print(f"[batch] status={state['status']} batch_id={state['batch_id']}", flush=True)
        return batch_state_path(video_path)

    if state["status"] != "completed":
        raise RuntimeError(f"Batch termine avec statut non supporte: {state['status']}")

    return finalize_batch_validation(video_path, model, source, target, speakers, state)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valide les candidats speakers avec OpenAI avant la correction du transcript."
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
        "--model",
        default=os.getenv("CHUNK_SPEAKER_VALIDATION_MODEL", DEFAULT_MODEL),
        help=f"Modele OpenAI de validation. Defaut: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "batch"),
        default=current_openai_mode(),
        help="Mode d'execution OpenAI. Defaut: valeur du pipeline global.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=DEFAULT_WAIT_FOR_BATCH,
        help="En mode batch, attend la fin du job et telecharge les resultats. Defaut: actif.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=30,
        help="En mode batch avec --wait, intervalle entre deux polls. Defaut: 30.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le JSON de speakers valides meme s'il existe deja.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    args.model = normalize_model_name(args.model)
    openai_mode = current_openai_mode()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"Modele validation speakers: {args.model}", flush=True)
    if args.mode != openai_mode:
        print(f"[info] mode OpenAI global={openai_mode}, step 20B executee en mode {args.mode}.", flush=True)
    client = openai_client() if args.mode == "normal" else None
    done = 0
    for video_path in videos:
        if args.mode == "batch":
            result = validate_file_batch(
                args.model,
                video_path,
                force=args.force,
                wait=args.wait,
                poll_interval_seconds=args.poll_interval_seconds,
            )
        else:
            result = validate_file_live(client, args.model, video_path, force=args.force)
        if result:
            done += 1

    print(f"{done} JSON de speakers valides generes.")


if __name__ == "__main__":
    main()
