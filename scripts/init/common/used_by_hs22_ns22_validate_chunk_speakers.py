import argparse
import json
import os
import re
import sys
import time
import unicodedata
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
from common.pipeline_paths import chunks_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
CHUNKS_NAME = "transcript_chunks.json"
CHUNKS_SPEAKER_VALIDATED_NAME = "transcript_chunks_speaker_validated.json"
SPEAKER_VALIDATION_LOG_NAME = "speaker_validation_log.json"
BATCH_STATE_NAME = "speaker_validation_batch_state.json"
BATCH_INPUT_NAME = "speaker_validation_batch_input.jsonl"
BATCH_OUTPUT_NAME = "speaker_validation_batch_output.jsonl"
BATCH_ERROR_NAME = "speaker_validation_batch_error.jsonl"
DEFAULT_WAIT_FOR_BATCH = True
LEGACY_CHUNKS_SUFFIX = "_chunks.json"
LEGACY_CHUNKS_CORRECTED_SUFFIX = "_chunks_corrected.json"
DEFAULT_MODEL = "gpt-5.4-nano"
SYSTEM_PROMPT = (
    "Tu verifies une liste de speakers detectes automatiquement dans une video. "
    "Garde uniquement les noms qui designent vraiment des personnes physiques. "
    "Rejete les entreprises, ecoles, services, metiers, titres, lieux, slogans, URLs, "
    "mots OCR parasites et noms incomplets. Reponds uniquement par un tableau JSON "
    "de chaines, en reprenant exactement les noms valides fournis."
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


def chunks_path(video_path):
    video_chunks_dir = chunks_dir(video_path)
    preferred = video_chunks_dir / CHUNKS_NAME
    legacy = video_chunks_dir / f"{video_path.stem}{LEGACY_CHUNKS_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def corrected_chunks_path(video_path):
    video_chunks_dir = chunks_dir(video_path)
    preferred = video_chunks_dir / CHUNKS_SPEAKER_VALIDATED_NAME
    legacy = video_chunks_dir / f"{video_path.stem}{LEGACY_CHUNKS_CORRECTED_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def validation_log_path(video_path):
    return chunks_dir(video_path) / SPEAKER_VALIDATION_LOG_NAME


def batch_state_path(video_path):
    return chunks_dir(video_path) / BATCH_STATE_NAME


def batch_input_path(video_path):
    return chunks_dir(video_path) / BATCH_INPUT_NAME


def batch_output_path(video_path):
    return chunks_dir(video_path) / BATCH_OUTPUT_NAME


def batch_error_path(video_path):
    return chunks_dir(video_path) / BATCH_ERROR_NAME


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
    for chunk in payload.get("chunks", []):
        for speaker in chunk.get("meta_data", {}).get("speakers", []) or []:
            speaker = " ".join(str(speaker).split()).strip()
            key = normalize_name(speaker)
            if not speaker or not key or key in seen:
                continue
            seen.add(key)
            speakers.append(speaker)
    return speakers


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


def build_response_request(model, speakers):
    user_prompt = (
        "Parmi cette liste de speakers, lesquels sont vraiment des personnes ? "
        "Reponds uniquement par un tableau JSON de noms valides, sans commentaire.\n\n"
        + json.dumps(speakers, ensure_ascii=False, indent=2)
    )
    request_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return (
        {
            "model": model,
            "input": request_messages,
            "max_output_tokens": 512,
        },
        {
            "model": model,
            "messages": request_messages,
            "api": "responses.create",
            "max_output_tokens": 512,
        },
    )


def ask_gpt(client, model, speakers):
    body, request_log = build_response_request(model, speakers)
    if hasattr(client, "responses"):
        response = client.responses.create(**body)
        answer = response_text(response)
        return answer, request_log

    response = client.chat.completions.create(
        model=model,
        messages=body["input"],
        max_completion_tokens=body["max_output_tokens"],
    )
    answer = response.choices[0].message.content or ""
    request_log["api"] = "chat.completions.create"
    request_log["max_completion_tokens"] = body["max_output_tokens"]
    return answer, request_log


def parse_valid_speakers(answer, original_speakers):
    text = str(answer).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("["):
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            text = match.group(0)

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Reponse GPT inattendue: {answer!r}")

    original_by_key = {normalize_name(name): name for name in original_speakers}
    valid = []
    seen = set()
    for name in parsed:
        key = normalize_name(name)
        if not key or key in seen or key not in original_by_key:
            continue
        seen.add(key)
        valid.append(original_by_key[key])
    return valid


def filter_chunk_speakers(payload, valid_speakers):
    valid_keys = {normalize_name(name) for name in valid_speakers}
    corrected = deepcopy(payload)
    for chunk in corrected.get("chunks", []):
        meta_data = chunk.get("meta_data")
        if not isinstance(meta_data, dict):
            continue
        speakers = meta_data.get("speakers")
        if not isinstance(speakers, list):
            continue
        meta_data["speakers"] = [
            speaker
            for speaker in speakers
            if normalize_name(speaker) in valid_keys
        ]
    return corrected


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


def write_validation_outputs(model, source, target, log_target, speakers, valid_speakers, answer, request_log, payload):
    corrected = filter_chunk_speakers(payload, valid_speakers)
    corrected["speaker_validation"] = {
        "model": model,
        "source": source.name,
        "answer": answer,
        "reviewed": len(speakers),
        "kept": len(valid_speakers),
        "rejected": len(speakers) - len(valid_speakers),
        "valid_speakers": valid_speakers,
    }
    target.write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_payload = {
        "model": model,
        "source": source.name,
        "reviewed": len(speakers),
        "kept": len(valid_speakers),
        "rejected": len(speakers) - len(valid_speakers),
        "requests": [
            {
                "speakers": speakers,
                "request": request_log,
                "response": answer,
            }
        ] if speakers else [],
    }
    log_target.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[write] {target} ({len(valid_speakers)} speakers gardes, "
        f"{len(speakers) - len(valid_speakers)} rejetes)",
        flush=True,
    )
    print(f"[write] log -> {log_target}", flush=True)
    return target


def validate_file_live(client, model, video_path, force=False):
    source = chunks_path(video_path)
    target = corrected_chunks_path(video_path)
    log_target = validation_log_path(video_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] chunks introuvables: {source}")
        return None

    payload = load_json(source)
    speakers = unique_speakers(payload)
    if speakers:
        answer, request_log = ask_gpt(client, model, speakers)
        answer = answer.strip()
        valid_speakers = parse_valid_speakers(answer, speakers)
    else:
        body, request_log = build_response_request(model, speakers)
        _ = body
        answer = "[]"
        valid_speakers = []

    return write_validation_outputs(model, source, target, log_target, speakers, valid_speakers, answer, request_log, payload)


def submit_batch_validation(video_path, model, speakers):
    client = openai_client()
    video_chunks_dir = chunks_dir(video_path)
    video_chunks_dir.mkdir(parents=True, exist_ok=True)
    input_path = batch_input_path(video_path)
    state_path = batch_state_path(video_path)
    body, _request_log = build_response_request(model, speakers)

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
            "script": "22_CHUNK_validate_chunk_speakers.py",
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
    source = chunks_path(video_path)
    return source.name if source.exists() else CHUNKS_NAME


def finalize_batch_validation(video_path, model, source, target, log_target, speakers, payload, state):
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
    valid_speakers = parse_valid_speakers(answer, speakers) if speakers else []
    _body, request_log = build_response_request(model, speakers)
    return write_validation_outputs(model, source, target, log_target, speakers, valid_speakers, answer or "[]", request_log, payload)


def validate_file_batch(model, video_path, force=False, wait=False, poll_interval_seconds=30):
    source = chunks_path(video_path)
    target = corrected_chunks_path(video_path)
    log_target = validation_log_path(video_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] chunks introuvables: {source}")
        return None

    payload = load_json(source)
    speakers = unique_speakers(payload)
    if not speakers:
        _body, request_log = build_response_request(model, speakers)
        return write_validation_outputs(model, source, target, log_target, speakers, [], "[]", request_log, payload)

    if force:
        for artifact_path in (
            target,
            log_target,
            batch_state_path(video_path),
            batch_input_path(video_path),
            batch_output_path(video_path),
            batch_error_path(video_path),
        ):
            if artifact_path.exists():
                artifact_path.unlink()

    state = load_batch_state(video_path)
    if state is None:
        state_path = submit_batch_validation(video_path, model, speakers)
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

    return finalize_batch_validation(video_path, model, source, target, log_target, speakers, payload, state)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valide les speakers des chunks avec OpenAI et produit un JSON chunks corrige."
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
        help="Regenere le JSON chunks corrige meme s'il existe deja.",
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
        print(f"[info] mode OpenAI global={openai_mode}, step 22 executee en mode {args.mode}.", flush=True)
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

    print(f"{done} JSON chunks corriges generes.")


if __name__ == "__main__":
    main()
