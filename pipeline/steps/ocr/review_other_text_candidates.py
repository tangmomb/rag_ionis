import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pipeline.support.paths import existing_ocr_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SOURCE_DIRNAME = "other_text_review_candidates"
LEGACY_SOURCE_DIRNAME = "ocr_processed_filtered_others_boxes"
SOURCE_MANIFEST_NAME = "review_candidates_manifest.json"
LEGACY_SOURCE_MANIFEST_NAME = "manifest.json"
OUTPUT_DIRNAME = "other_text_gpt_review"
LEGACY_OUTPUT_DIRNAME = "ocr_processed_filtered_others_boxes_review"
SUMMARY_NAME = "review_summary.json"
LEGACY_SUMMARY_NAME = "summary.json"
DEFAULT_MODEL = "gpt-5.6-luna"
REVIEW_MAX_OUTPUT_TOKENS = 400
DEFAULT_MODE = "batch"
DEFAULT_WAIT_FOR_BATCH = True
BATCH_STATE_NAME = "batch_state.json"
BATCH_INPUT_NAME = "batch_input.jsonl"
BATCH_OUTPUT_NAME = "batch_output.jsonl"
BATCH_ERROR_NAME = "batch_error.jsonl"
SYSTEM_PROMPT = (
    "Tu analyses une ou deux images completes provenant d'une video. "
    "Ta tache: 1) regarder d'abord l'ensemble des images pour juger le contexte visuel global, "
    "2) analyser ensuite uniquement le texte situe dans les zones encadrees en rouge, sans elargir ton attention a d'autres parties des images pour corriger le texte, "
    "3) dire si ce texte ressemble a du texte ajoute au montage "
    "(titre, lower third, intertitre, texte graphique, habillage, texte pose en post-production), attention il se peut que le texte soit en cours de fondu ou d'apparition donc pas hyper contraste, "
    "plutot qu'a du texte capture naturellement dans la scene, "
    "4) verifier enfin si le texte OCR fourni contient une erreur de lecture, et si oui proposer une correction. si le texte que tu proposes n'est pas une petite correction de l'ocr, alors considere qu'il ne s'agit pas d'un ajout au montage mais d'une erreur de lecture de l'ocr. "
    "Regle importante: si le texte encadre n'est pas parfaitement lisible, net, propre et clairement detache du decor, alors considere que ce n'est PAS du texte ajoute au montage. "
    "Autre regle importante: si le texte est bien net et passe par-dessus plusieurs elements differents (personnes, vetements, objets, decors, arriere-plan, etc.), alors considere que c'est FORCEMENT un ajout au montage. "
    "S'il y a deux images, elles se suivent dans le temps et montrent potentiellement un fondu, une apparition ou une animation du meme texte. "
    "Tu dois justement utiliser ces deux images successives pour distinguer un vrai texte ajoute au montage d'un simple element de decor ou d'image. "
    "Si une animation, apparition, transition ou evolution visuelle est visible entre les deux images pour la zone encadree, considere que cette box correspond FORCEMENT a un ajout au montage. "
    "Dans ce cas, applique la meme conclusion a toutes les boxes qui partagent clairement le meme look graphique, le meme style visuel ou le meme habillage. "
    "Reponds uniquement en JSON avec les cles: has_ocr_error (boolean), corrected_text (string), is_added_in_edit (boolean), confidence (number entre 0 et 1), reason (string court de 20 mots maximum). "
    "Si le texte OCR semble deja correct, corrected_text doit reprendre le texte OCR tel quel."
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
    candidates = sorted(
        path for path in parent_dir.iterdir() if path.is_dir() and any(video_files(path))
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


def normalize_openai_mode(value):
    normalized = str(value).strip().lower()
    if normalized in {"normal", "live"}:
        return "live"
    if normalized == "batch":
        return "batch"
    return DEFAULT_MODE


def source_dir(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / SOURCE_DIRNAME
    legacy = video_ocr_dir / LEGACY_SOURCE_DIRNAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def source_manifest_path(video_path):
    directory = source_dir(video_path)
    preferred = directory / SOURCE_MANIFEST_NAME
    legacy = directory / LEGACY_SOURCE_MANIFEST_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def output_dir(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / OUTPUT_DIRNAME
    legacy = video_ocr_dir / LEGACY_OUTPUT_DIRNAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def summary_path(video_path):
    directory = output_dir(video_path)
    preferred = directory / SUMMARY_NAME
    legacy = directory / LEGACY_SUMMARY_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def batch_state_path(video_path):
    return output_dir(video_path) / BATCH_STATE_NAME


def batch_input_path(video_path):
    return output_dir(video_path) / BATCH_INPUT_NAME


def batch_output_path(video_path):
    return output_dir(video_path) / BATCH_OUTPUT_NAME


def batch_error_path(video_path):
    return output_dir(video_path) / BATCH_ERROR_NAME


def openai_client():
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Le package 'openai' est requis pour cette step. Active le venv du projet ou installe les dependances."
        ) from error

    return OpenAI()


def load_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode_image_data_url(path):
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


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


def parse_json_answer(answer):
    text = str(answer).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Une sortie Responses peut etre tronquee (notamment au milieu de
        # `reason`). On recupere les champs simples deja emis et on laisse
        # l'appelant remettre le texte OCR original si corrected_text manque.
        def extract_bool(name, default=False):
            match = re.search(rf'"{re.escape(name)}"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
            return match.group(1).lower() == "true" if match else default

        def extract_number(name, default=0.0):
            match = re.search(rf'"{re.escape(name)}"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', text)
            try:
                return float(match.group(1)) if match else default
            except ValueError:
                return default

        def extract_string(name, default=""):
            match = re.search(rf'"{re.escape(name)}"\s*:\s*"((?:\\.|[^"\\])*)', text, flags=re.DOTALL)
            if not match:
                return default
            try:
                return json.loads('"' + match.group(1) + '"')
            except json.JSONDecodeError:
                return match.group(1)

        parsed = {
            "has_ocr_error": extract_bool("has_ocr_error"),
            "corrected_text": extract_string("corrected_text"),
            "is_added_in_edit": extract_bool("is_added_in_edit"),
            "confidence": extract_number("confidence"),
            "reason": extract_string("reason", "Reponse JSON tronquee; champs recuperes partiellement."),
        }
    corrected_text = " ".join(str(parsed.get("corrected_text", "")).split()).strip()
    return {
        "has_ocr_error": bool(parsed.get("has_ocr_error")),
        "corrected_text": corrected_text,
        "is_added_in_edit": bool(parsed.get("is_added_in_edit")),
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))),
        "reason": " ".join(str(parsed.get("reason", "")).split()).strip(),
    }


def build_user_prompt(item):
    return (
        "Tu recois le screenshot courant annote et parfois le screenshot precedent annote avec les memes boxes rouges. "
        "S'il y a deux images, elles se suivent et le fait de voir les deux doit t'aider a juger si la zone rouge pointe une animation, un fondu ou un texte stable ajoute au montage. "
        "Utilise explicitement ces deux images successives pour distinguer un simple contenu visuel d'un texte ajoute en post-production. "
        "S'il y a une animation, une apparition, un fondu ou une transition visible entre les deux images dans cette zone rouge, alors cette box doit etre consideree comme FORCEMENT ajoutee au montage. "
        "Et si cette box est ajoutee au montage a cause de ce comportement visuel, la meme conclusion doit s'appliquer a toutes les boxes au meme look graphique. "
        "Regarde d'abord les images completes pour comprendre la scene et le contexte general. "
        "Ensuite concentre-toi sur le texte dans la zone encadree en rouge. "
        "Decide d'abord si ce texte encadre est un ajout au montage ou non, en tenant compte du contexte global des images. "
        "Si le texte n'est pas parfaitement lisible ou net, reponds que ce n'est pas du montage. "
        "Si le texte est bien net et traverse visiblement plusieurs elements differents de l'image, reponds que c'est du montage. "
        "Apres cette decision, verifie si le texte OCR fourni correspond bien a ce qui est visible dans la zone rouge, et corrige-le si besoin. "
        "Question pratique a te poser: voici deux screenshots qui se suivent, la zone rouge pointe-t-elle une animation et y a-t-il une faute d'orthographe dans l'OCR ? "
        "Contexte OCR:\n"
        f"- timecode: {item.get('timecode', '')}\n"
        f"- texte OCR: {item.get('text', '')}\n"
        "Reponds uniquement avec le JSON demande."
    )


def build_messages(image_path, item, previous_image_path=None):
    user_prompt = build_user_prompt(item)
    user_content = [{"type": "input_text", "text": user_prompt}]
    if previous_image_path is not None:
        user_content.append({"type": "input_text", "text": "Image 1: screenshot precedent."})
        user_content.append(
            {"type": "input_image", "image_url": encode_image_data_url(previous_image_path), "detail": "high"}
        )
    user_content.append({"type": "input_text", "text": "Image 2: screenshot courant cible."})
    user_content.append({"type": "input_image", "image_url": encode_image_data_url(image_path), "detail": "high"})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    request_log = {
        "api": "responses.create",
        "model": None,
        "max_output_tokens": REVIEW_MAX_OUTPUT_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    return messages, request_log


def build_response_request(model, image_path, item, previous_image_path=None):
    messages, request_log = build_messages(image_path, item, previous_image_path=previous_image_path)
    request_log["model"] = model
    body = {
        "model": model,
        "input": messages,
        "max_output_tokens": REVIEW_MAX_OUTPUT_TOKENS,
    }
    return body, request_log


def review_one_image(client, model, image_path, item, previous_image_path=None):
    body, request_log = build_response_request(model, image_path, item, previous_image_path=previous_image_path)
    response = client.responses.create(**body)
    answer = response_text_from_payload(response.model_dump())
    return answer, request_log, response.model_dump()


def safe_stem(value):
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return cleaned.strip("_") or "item"


def write_text(path, content):
    path.write_text(str(content), encoding="utf-8")


def prepare_jobs(video_path, limit_images=None):
    source_manifest = source_manifest_path(video_path)
    if not source_manifest.exists():
        raise FileNotFoundError(f"manifest introuvable: {source_manifest}")

    source_payload = load_manifest(source_manifest)
    items = list(source_payload.get("items", []))
    if limit_images is not None:
        items = items[:limit_images]

    jobs = []
    for index, item in enumerate(items, start=1):
        crop_name = item.get("crop")
        previous_crop_name = item.get("previous_crop")
        if not crop_name:
            continue
        image_path = source_dir(video_path) / crop_name
        if not image_path.exists():
            print(f"[skip] crop introuvable: {image_path}", flush=True)
            continue

        previous_image_path = None
        if previous_crop_name:
            candidate_previous_path = source_dir(video_path) / previous_crop_name
            if candidate_previous_path.exists():
                previous_image_path = candidate_previous_path

        review_name = f"{index:03d}__{safe_stem(Path(crop_name).stem)}"
        jobs.append(
            {
                "index": index,
                "custom_id": f"review-{index:03d}",
                "review_name": review_name,
                "crop_name": crop_name,
                "previous_crop_name": previous_crop_name,
                "item": item,
                "image_path": image_path,
                "previous_image_path": previous_image_path,
            }
        )
    return source_manifest, source_payload, jobs


def reset_review_dir(review_dir):
    if review_dir.exists():
        for child in review_dir.iterdir():
            if child.is_file():
                child.unlink()
    review_dir.mkdir(parents=True, exist_ok=True)


def write_review_artifacts(target_dir, job, parsed, request_log, raw_answer, raw_payload):
    review_dir = target_dir / job["review_name"]
    reset_review_dir(review_dir)
    shutil.copy2(job["image_path"], review_dir / job["image_path"].name)
    if job["previous_image_path"] is not None:
        shutil.copy2(job["previous_image_path"], review_dir / job["previous_image_path"].name)

    write_text(review_dir / "model_prompt.txt", request_log["messages"][1]["content"])
    write_text(review_dir / "model_response.txt", raw_answer)
    (review_dir / "api_request.json").write_text(
        json.dumps(request_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (review_dir / "api_response.json").write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (review_dir / "review_decision.json").write_text(
        json.dumps(
            {
                "crop": job["crop_name"],
                "previous_crop": job["previous_crop_name"],
                "entry_id": job["item"].get("entry_id"),
                "timecode": job["item"].get("timecode"),
                "text": job["item"].get("text"),
                "image": job["item"].get("image"),
                "box": job["item"].get("box"),
                **parsed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "review_dir": job["review_name"],
        "crop": job["crop_name"],
        "previous_crop": job["previous_crop_name"],
        "entry_id": job["item"].get("entry_id"),
        "timecode": job["item"].get("timecode"),
        "text": job["item"].get("text"),
        "image": job["item"].get("image"),
        "box": job["item"].get("box"),
        **parsed,
    }


def write_summary(video_path, model, source_manifest, decisions):
    payload = {
        "model": model,
        "source_dir": relative_to_video_dir(source_dir(video_path), video_path),
        "source_manifest": relative_to_video_dir(source_manifest, video_path),
        "output_dir": relative_to_video_dir(output_dir(video_path), video_path),
        "reviewed_count": len(decisions),
        "added_in_edit_count": sum(1 for item in decisions if item["is_added_in_edit"]),
        "not_added_count": sum(1 for item in decisions if not item["is_added_in_edit"]),
        "items": decisions,
    }
    path = summary_path(video_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {path}", flush=True)
    return path


def review_video_live(video_path, model, force=False, limit_images=None):
    source_manifest, _, jobs = prepare_jobs(video_path, limit_images=limit_images)
    target_dir = output_dir(video_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = summary_path(video_path)
    if path.exists() and not force:
        print(f"[skip] {path.name} existe deja")
        return path

    client = openai_client()
    decisions = []
    for job in jobs:
        answer, request_log, raw_payload = review_one_image(
            client,
            model,
            job["image_path"],
            job["item"],
            previous_image_path=job["previous_image_path"],
        )
        parsed = parse_json_answer(answer)
        if not parsed["corrected_text"]:
            parsed["corrected_text"] = " ".join(str(job["item"].get("text", "")).split()).strip()
        decisions.append(write_review_artifacts(target_dir, job, parsed, request_log, answer, raw_payload))
        print(
            f"[review {job['index']}/{len(jobs)}] {job['crop_name']}: added={str(parsed['is_added_in_edit']).lower()} conf={parsed['confidence']:.2f}",
            flush=True,
        )

    return write_summary(video_path, model, source_manifest, decisions)


def save_batch_state(path, state):
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def submit_batch_review(video_path, model, jobs):
    client = openai_client()
    target_dir = output_dir(video_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    input_path = batch_input_path(video_path)
    state_path = batch_state_path(video_path)

    with input_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            body, _request_log = build_response_request(
                model,
                job["image_path"],
                job["item"],
                previous_image_path=job["previous_image_path"],
            )
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
            "script": "13_OCR_review_other_text_candidates.py",
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
        "source_manifest": relative_to_video_dir(source_manifest_path(video_path), video_path),
    }
    save_batch_state(state_path, state)
    print(f"[batch] submitted id={batch.id} status={batch.status} requests={len(jobs)}", flush=True)
    return state_path


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


def finalize_batch_review(video_path, model, jobs, state):
    client = openai_client()
    output_file_id = state.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch complete mais output_file_id absent.")

    output_response = client.files.content(output_file_id)
    output_response.write_to_file(batch_output_path(video_path))

    error_file_id = state.get("error_file_id")
    if error_file_id:
        error_response = client.files.content(error_file_id)
        error_response.write_to_file(batch_error_path(video_path))

    records = parse_batch_lines(batch_output_path(video_path))
    record_by_id = {record.get("custom_id"): record for record in records if record.get("custom_id")}
    target_dir = output_dir(video_path)
    source_manifest = source_manifest_path(video_path)
    decisions = []

    for job in jobs:
        record = record_by_id.get(job["custom_id"])
        if not record:
            print(f"[skip] resultat batch introuvable pour {job['custom_id']}", flush=True)
            continue
        response = record.get("response") or {}
        if response.get("status_code") != 200:
            print(f"[skip] batch {job['custom_id']} status={response.get('status_code')}", flush=True)
            continue
        raw_payload = response.get("body") or {}
        answer = response_text_from_payload(raw_payload)
        parsed = parse_json_answer(answer)
        if not parsed["corrected_text"]:
            parsed["corrected_text"] = " ".join(str(job["item"].get("text", "")).split()).strip()
        _body, request_log = build_response_request(
            model,
            job["image_path"],
            job["item"],
            previous_image_path=job["previous_image_path"],
        )
        decisions.append(write_review_artifacts(target_dir, job, parsed, request_log, answer, raw_payload))
        print(
            f"[batch-result {job['index']}/{len(jobs)}] {job['crop_name']}: added={str(parsed['is_added_in_edit']).lower()} conf={parsed['confidence']:.2f}",
            flush=True,
        )

    return write_summary(video_path, model, source_manifest, decisions)


def review_video_batch(video_path, model, force=False, limit_images=None, wait=False, poll_interval_seconds=30):
    source_manifest, _, jobs = prepare_jobs(video_path, limit_images=limit_images)
    target_dir = output_dir(video_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = summary_path(video_path)
    if path.exists() and not force:
        print(f"[skip] {path.name} existe deja")
        return path

    if force:
        for artifact_path in (
            path,
            batch_state_path(video_path),
            batch_input_path(video_path),
            batch_output_path(video_path),
            batch_error_path(video_path),
        ):
            if artifact_path.exists():
                artifact_path.unlink()

    state = load_batch_state(video_path)
    if state is None:
        state_path = submit_batch_review(video_path, model, jobs)
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

    return finalize_batch_review(video_path, model, jobs, state)


def review_video(video_path, model, mode="live", force=False, limit_images=None, wait=False, poll_interval_seconds=30):
    if mode == "batch":
        return review_video_batch(
            video_path,
            model,
            force=force,
            limit_images=limit_images,
            wait=wait,
            poll_interval_seconds=poll_interval_seconds,
        )
    return review_video_live(video_path, model, force=force, limit_images=limit_images)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demande a GPT si le texte dans la zone rouge des images 'others' ressemble a du texte ajoute au montage."
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
        default=os.getenv("OCR_OTHERS_REVIEW_MODEL", DEFAULT_MODEL),
        help=f"Modele OpenAI a utiliser. Defaut: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--mode",
        choices=("live", "batch"),
        default=normalize_openai_mode(os.getenv("PIPELINE_OPENAI_MODE", DEFAULT_MODE)),
        help=f"Mode d'execution OpenAI. Defaut: {DEFAULT_MODE}.",
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
        "--limit-images",
        type=int,
        help="Nombre maximum d'images annotees a soumettre par video.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere les reviews meme si elles existent deja.",
    )
    return parser.parse_args()


def main():
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if review_video(
            video_path,
            args.model,
            mode=args.mode,
            force=args.force,
            limit_images=args.limit_images,
            wait=args.wait,
            poll_interval_seconds=args.poll_interval_seconds,
        ):
            done += 1
    print(f"{done} review(s) GPT generee(s).")


if __name__ == "__main__":
    main()
