import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PIPELINE_TRANSCRIPTS_DIR_NAME"] = "transcripts_ocr"

from pipeline.support.analysis import update_analysed_infos  # noqa: E402
from pipeline.support.paths import existing_transcripts_dir, relative_to_video_dir  # noqa: E402


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
DEFAULT_MODEL = "gpt-5.4-nano"
SOURCE_NAME = "ocr_subtitles_timecoded.txt"
TARGET_NAME = "ocr_subtitles_timecoded_corrected.txt"
BATCH_STATE_NAME = "ocr_spacing_batch_state.json"
BATCH_INPUT_NAME = "ocr_spacing_batch_input.jsonl"
BATCH_OUTPUT_NAME = "ocr_spacing_batch_output.jsonl"
BATCH_ERROR_NAME = "ocr_spacing_batch_error.jsonl"
DEFAULT_WAIT_FOR_BATCH = True
LEGACY_SOURCE_SUFFIX = "_ocr_subtitle_timecodes.txt"
LEGACY_TARGET_SUFFIX = "_ocr_subtitle_timecodes_corrected.txt"
SUBTITLE_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
SYSTEM_PROMPT = (
    "Tu corriges uniquement les espaces manquants ou superflus dans une ligne de sous-titre OCR en francais. "
    "Ne change pas les mots, ne corrige pas l'orthographe, ne reformule pas, ne modifies pas la ponctuation sauf si elle sert strictement a remettre un espace evident. "
    "Si le texte est deja correct, renvoie-le tel quel. "
    "Reponds uniquement avec la ligne corrigee, sans guillemets ni commentaire."
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


def subtitle_source_path(video_path):
    transcript_dir = existing_transcripts_dir(video_path)
    preferred = transcript_dir / SOURCE_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_SOURCE_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def subtitle_target_path(video_path):
    transcript_dir = existing_transcripts_dir(video_path)
    preferred = transcript_dir / TARGET_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_TARGET_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def batch_state_path(video_path):
    return existing_transcripts_dir(video_path) / BATCH_STATE_NAME


def batch_input_path(video_path):
    return existing_transcripts_dir(video_path) / BATCH_INPUT_NAME


def batch_output_path(video_path):
    return existing_transcripts_dir(video_path) / BATCH_OUTPUT_NAME


def batch_error_path(video_path):
    return existing_transcripts_dir(video_path) / BATCH_ERROR_NAME


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
            "max_output_tokens": 256,
        },
        {
            "model": model,
            "messages": request_messages,
            "api": "responses.create",
            "max_output_tokens": 256,
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


def write_corrected_output(video_path, source, target, output_lines, model, request_count, changed_count):
    rendered_lines = []
    for item in output_lines:
        if isinstance(item, dict):
            rendered_lines.append(corrected_line(item["timecode"], item["text"], item.get("corrected_text", item["text"])))
        else:
            rendered_lines.append(str(item))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rendered_lines).strip() + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "ocr_subtitle_spacing",
        {
            "status": "done",
            "source": relative_to_video_dir(source, video_path),
            "corrected_file": relative_to_video_dir(target, video_path),
            "line_count": len(rendered_lines),
            "request_count": request_count,
            "changed_count": changed_count,
            "model": model,
        },
    )
    print(f"[ok] {target}")
    return target


def process_video_live(client, model, video_path, force=False):
    source = subtitle_source_path(video_path)
    target = subtitle_target_path(video_path)
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

    return write_corrected_output(video_path, source, target, output_lines, model, request_count, changed_count)


def submit_batch_spacing(video_path, model, jobs):
    client = openai_client()
    transcript_dir = existing_transcripts_dir(video_path)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    input_path = batch_input_path(video_path)
    state_path = batch_state_path(video_path)

    with input_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            body, _request_log = build_response_request(model, job["text"])
            record = {
                "custom_id": job["custom_id"],
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
        "source": relative_to_video_dir(subtitle_source_path(video_path), video_path),
    }
    save_batch_state(state_path, state)
    print(f"[batch] submitted id={batch.id} status={batch.status} requests={len(jobs)}", flush=True)
    return state_path


def finalize_batch_spacing(video_path, model, source, target, jobs, output_lines, state):
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

    return write_corrected_output(video_path, source, target, output_lines, model, request_count, changed_count)


def process_video_batch(model, video_path, force=False, wait=False, poll_interval_seconds=30):
    source = subtitle_source_path(video_path)
    target = subtitle_target_path(video_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] sous-titres OCR timecodes introuvables: {source}")
        return None

    lines = source.read_text(encoding="utf-8").splitlines()
    jobs, output_lines = prepare_line_jobs(lines)
    if not jobs:
        return write_corrected_output(video_path, source, target, output_lines, model, 0, 0)

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
        state_path = submit_batch_spacing(video_path, model, jobs)
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

    return finalize_batch_spacing(video_path, model, source, target, jobs, output_lines, state)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Corrige les espaces manquants dans les sous-titres OCR timecodes avec OpenAI."
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
        "--limit-videos",
        type=int,
        help="Nombre maximum de videos a analyser.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le fichier corrige meme s'il existe deja.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modele OpenAI a utiliser. Defaut: {DEFAULT_MODEL}.",
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
    return parser.parse_args()


def main():
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    args = parse_args()
    openai_mode = current_openai_mode()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    if args.mode != openai_mode:
        print(f"[info] mode OpenAI global={openai_mode}, step 16 executee en mode {args.mode}.", flush=True)
    client = openai_client() if args.mode == "normal" else None
    done = 0
    for video_path in videos:
        if args.mode == "batch":
            result = process_video_batch(
                args.model,
                video_path,
                force=args.force,
                wait=args.wait,
                poll_interval_seconds=args.poll_interval_seconds,
            )
        else:
            result = process_video_live(client, args.model, video_path, force=args.force)
        if result:
            done += 1

    print(f"{done} fichier(s) de sous-titres corrige(s).")


if __name__ == "__main__":
    main()
