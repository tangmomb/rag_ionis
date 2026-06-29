import argparse
import base64
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_FILTER_MODEL = "gpt-5.4"
DEFAULT_FILTER_DETAIL = "low"
DEFAULT_INPUT_PRICE_PER_1M = 1.25
PROMPT_CACHE_KEY = "init-step05-filter-text-v1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def image_data_url(path):
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_files(video_images_dir):
    return sorted(
        (
            path
            for path in video_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )


def image_video_dirs(video_dir):
    candidates = []
    if (video_dir / "images").is_dir() and any(image_files(video_dir / "images")):
        candidates.append(video_dir)
    for child in sorted(video_dir.iterdir()):
        if child.is_dir() and (child / "images").is_dir() and any(image_files(child / "images")):
            candidates.append(child)
    return candidates


def chunks(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def build_request_body(model, batch, detail):
    content = [
        {
            "type": "input_text",
            "text": "Pour chaque image, reponds uniquement en JSON valide avec la forme {\"items\":[{\"image\":\"nom_du_fichier\",\"has_text\":true|false}]}. Marque has_text=true seulement si le texte est clairement ajoute au montage video et utile pour l'OCR: sous-titres, titres a l'ecran, slides, questions, annotations, incrustations ou panneaux explicitement presents comme partie du contenu video. Marque has_text=false si le texte appartient au decor ou a l'environnement, meme s'il est lisible: roll-up, affiche de fond, signaletique de lieu, packaging, etiquette, objet de scene, element de decor ou texte secondaire non incruste. Si un mot ou une suite de mots semble encore en train d'apparaitre, de se construire ou d'animer, considere l'image comme non pertinente: le texte complet devrait suivre dans une image plus tardive, donc mets false. Ignore aussi les logos decoratifs, icones, pictogrammes et elements de UI sans texte lisible. Si tu as un doute, mets false. Ne fais aucun commentaire.",
        }
    ]

    for image_path in batch:
        content.append({"type": "input_text", "text": f"Image: {image_path.name}"})
        content.append({"type": "input_image", "image_url": image_data_url(image_path), "detail": detail})

    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 1000,
        "temperature": 0,
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "prompt_cache_retention": "24h",
    }


def analyze_sync(client, model, batch, detail):
    body = build_request_body(model, batch, detail)
    response = client.responses.create(**body)
    return parse_json_response(response.output_text), getattr(response, "usage", None)


def build_batch_payload(request_id, model, batch, detail):
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": build_request_body(model, batch, detail),
    }


def analyze_batch(client, model, batch, detail, request_id):
    payload = build_batch_payload(request_id, model, batch, detail)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as tmp:
        tmp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        input_path = Path(tmp.name)

    try:
        with input_path.open("rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")

        batch_job = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        batch_job = wait_for_batch(client, batch_job.id)
        if batch_job.status != "completed":
            raise RuntimeError(f"Batch non termine correctement: {batch_job.status}")
        if not batch_job.output_file_id:
            raise RuntimeError("Batch termine mais output_file_id manquant")
        return parse_batch_output(client, batch_job.output_file_id, request_id)
    finally:
        try:
            input_path.unlink()
        except FileNotFoundError:
            pass


def wait_for_batch(client, batch_id, poll_seconds=5):
    while True:
        batch_job = client.batches.retrieve(batch_id)
        print(f"[batch] {batch_id}: {batch_job.status}", flush=True)
        if batch_job.status in {"completed", "failed", "expired", "cancelled"}:
            return batch_job
        time.sleep(poll_seconds)


def parse_batch_output(client, output_file_id, request_id):
    file_response = client.files.content(output_file_id)
    raw = file_response.text
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("custom_id") != request_id:
            continue
        error = row.get("error")
        if error:
            raise RuntimeError(f"Batch request error for {request_id}: {error}")
        response = row.get("response", {})
        body = response.get("body", {})
        usage = body.get("usage")
        output_text = body.get("output_text")
        if output_text:
            return parse_json_response(output_text), usage
        for item in body.get("output", []):
            content = item.get("content", [])
            for part in content:
                if part.get("type") == "output_text" and part.get("text"):
                    return parse_json_response(part["text"]), usage
        raise RuntimeError(f"Reponse batch introuvable pour {request_id}")


def usage_stats(usage):
    if not usage:
        return 0, 0, 0.0
    input_tokens = int(getattr(usage, "input_tokens", None) or getattr(usage, "get", lambda *_: 0)("input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", None) or getattr(usage, "get", lambda *_: 0)("output_tokens", 0) or 0)
    cost = (input_tokens / 1_000_000) * DEFAULT_INPUT_PRICE_PER_1M
    return input_tokens, output_tokens, cost


def parse_json_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtre les images qui contiennent du texte lisible avant l'OCR complet."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier contenant images/. Defaut: dernier sous-dossier de downloads/youtube avec images/",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent utilise si --video-dir est absent. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_FILTER_MODEL,
        help="Modele vision utilise pour le filtrage. Defaut: gpt-5.4-mini",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Nombre d'images envoyees par appel de filtrage. Defaut: 24",
    )
    parser.add_argument(
        "--detail",
        choices=("low", "high", "auto"),
        default=DEFAULT_FILTER_DETAIL,
        help="Niveau de detail image envoye au modele. Defaut: high",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        help="Nombre maximum de videos a analyser.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepte l'option pour compatibilite avec le pipeline. Sans effet.",
    )
    parser.add_argument(
        "--batch-api",
        action="store_true",
        help="Utilise la Batch API OpenAI au lieu des appels synchrones.",
    )
    return parser.parse_args()


def latest_video_dir(parent_dir):
    candidates = sorted(path for path in parent_dir.iterdir() if path.is_dir() and any(image_video_dirs(path)))
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier avec images trouve dans {parent_dir}")
    return candidates[-1]


def write_filtered_images(images_dir, images):
    output_dir = images_dir / "with_text"
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.iterdir():
        if existing.is_file():
            existing.unlink()
    copied = 0
    for path in images:
        if not path.exists():
            print(f"[skip] image introuvable: {path}", flush=True)
            continue
        shutil.copy2(path, output_dir / path.name)
        copied += 1
    print(f"[write] {copied} images -> {output_dir}", flush=True)


def main():
    load_dotenv(override=True)
    args = parse_args()
    client = OpenAI()
    batch_api = args.batch_api

    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(image_video_dirs(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"Modele filtrage: {args.model}", flush=True)
    print(f"Mode traitement: {'batch' if batch_api else 'synchrone'}", flush=True)

    for video_path in videos:
        images_dir = video_path / "images"
        images = image_files(images_dir)
        if not images:
            print(f"[skip] {video_path.name}: aucune image", flush=True)
            continue
        print(f"[analyse] {video_path.name}: {len(images)} images", flush=True)
        kept = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        total_batches = (len(images) + args.batch_size - 1) // args.batch_size
        for batch_index, batch in enumerate(chunks(images, args.batch_size), start=1):
            request_id = f"{video_path.name}_filter_{batch_index:04d}"
            print(f"[filter {batch_index}/{total_batches}] {request_id}: envoi", flush=True)
            if batch_api:
                result, usage = analyze_batch(client, args.model, batch, args.detail, request_id)
            else:
                result, usage = analyze_sync(client, args.model, batch, args.detail)
            input_tokens, output_tokens, cost = usage_stats(usage)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += cost
            print(
                f"[usage] in={input_tokens} out={output_tokens} cost=${cost:.6f} total=${total_cost:.6f}",
                flush=True,
            )
            for item in result.get("items", []):
                if item.get("has_text") is True and item.get("image"):
                    candidate = images_dir / item["image"]
                    if candidate.exists():
                        kept.append(candidate)
                    else:
                        print(f"[skip] image introuvable dans le lot: {candidate}", flush=True)
        kept = sorted({path.name: path for path in kept}.values(), key=lambda path: path.name)
        write_filtered_images(images_dir, kept)
        print(
            f"[usage total] in={total_input_tokens} out={total_output_tokens} cost=${total_cost:.6f}",
            flush=True,
        )
        print(f"[done] {video_path.name}: {len(kept)}/{len(images)} images gardees", flush=True)


if __name__ == "__main__":
    main()
