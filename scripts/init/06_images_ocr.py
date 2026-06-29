import argparse
import base64
import json
import os
import re
import sys
import time
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SECOND_PATTERN = re.compile(r"^seconde_(\d+(?:_\d+)?)$")
TIMECODE_PATTERN = re.compile(r"^(?:(\d{2})_)?(\d{2})_(\d{2})$")
DEFAULT_IMAGE_ANALYZE_MODEL = "gpt-5.4"
DEFAULT_IMAGE_ANALYZE_DETAIL = "low"
DEFAULT_INPUT_PRICE_PER_1M = 1.25
PROMPT_CACHE_KEY = "init-step06-ocr-v1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def image_data_url(path):
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_second(path):
    parsed = seconds_from_image_name(path.name)
    if parsed is not None:
        return parsed

    match = SECOND_PATTERN.match(path.stem)
    if not match:
        return float("inf")
    return float(match.group(1).replace("_", "."))


def seconds_from_image_name(name):
    stem = Path(name).stem
    timecode_match = TIMECODE_PATTERN.match(stem)
    if timecode_match:
        hours = int(timecode_match.group(1) or 0)
        minutes = int(timecode_match.group(2))
        seconds = int(timecode_match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    second_match = SECOND_PATTERN.match(stem)
    if second_match:
        return float(second_match.group(1).replace("_", "."))

    return None


def image_files(video_images_dir):
    return sorted(
        (
            path
            for path in video_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=image_second,
    )


def load_filtered_images(video_path, images_dir):
    filtered_dir = images_dir / "with_text"
    if not filtered_dir.is_dir():
        return image_files(images_dir)
    return image_files(filtered_dir)


def latest_video_dir(parent_dir):
    candidates = sorted(path for path in parent_dir.iterdir() if path.is_dir() and any(image_video_dirs(path)))
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier avec images trouve dans {parent_dir}")
    return candidates[-1]


def image_video_dirs(video_dir):
    candidates = []

    if (video_dir / "images").is_dir() and any(image_files(video_dir / "images")):
        candidates.append(video_dir)

    for child in sorted(video_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images").is_dir() and any(image_files(child / "images")):
            candidates.append(child)

    return candidates


def chunks(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def build_request_body(model, batch, detail):
    content = [
        {
            "type": "input_text",
            "text": "Analyse ces images extraites d'une video. Chaque image est nommee par seconde. Retranscris de facon litterale tous les textes ecrits visibles a l'ecran, y compris les sous-titres, les noms, les titres, les questions, les slides et les panneaux. Ignore les logos, meme s'ils contiennent du texte. S'il n'y a qu'un seul element de texte visible, considere-le comme un sous-titre. Si tu identifies plusieurs zones de texte ou plusieurs types de textes distincts sur une meme image, cree un element JSON par zone de texte, sans les fusionner. Si deux textes differents sont tres similaires, considere que le plus long des deux est un sous-titre. Ne fais pas de resume si le texte est lisible: recopie au plus pres le texte exact vu a l'ecran, ligne par ligne si besoin. Ignore les images sans texte lisible. Ne retourne une liste vide que si aucune image du lot ne contient de texte lisible. Reponds uniquement en JSON valide avec la forme {\"items\":[{\"image\":\"00_12.jpg\",\"text\":\"texte lu ou extrait litteral\",\"kind\":\"name|question|slide|title|subtitle|other\",\"confidence\":\"low|medium|high\"}]}. N'utilise jamais le kind \"logo\".",
        }
    ]

    for image_path in batch:
        content.append({"type": "input_text", "text": f"Image: {image_path.name}"})
        content.append({"type": "input_image", "image_url": image_data_url(image_path), "detail": detail})

    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 2000,
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
        raise RuntimeError(f"Réponse batch introuvable pour {request_id}")


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


def merge_result_items(items, result):
    for item in result.get("items", []):
        items.append(normalize_item_timecode(item))
    items.sort(key=lambda item: item.get("second", 0))
    return deduplicate_items(items)


def normalize_item_timecode(item):
    image = item.get("image", "")
    seconds = seconds_from_image_name(image)
    if seconds is not None:
        item["second"] = seconds
        item["timecode"] = format_timecode(seconds)
    return item


def normalize_detected_text(text):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def deduplicate_items(items):
    seen = set()
    seen_lines = set()
    deduplicated = []
    for item in items:
        lines = [line.strip() for line in str(item.get("text", "")).splitlines() if line.strip()]
        new_lines = []
        for line in lines:
            line_key = normalize_detected_text(line)
            if line_key and line_key not in seen_lines:
                new_lines.append(line)
        item["text"] = "\n".join(new_lines)

        key = normalize_detected_text(item.get("text", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        for line in new_lines:
            seen_lines.add(normalize_detected_text(line))
        deduplicated.append(item)
    return deduplicated


def write_outputs(transcript_dir, video_id, result):
    json_path = transcript_dir / f"{video_id}_ocr.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {len(result.get('items', []))} items -> {json_path}", flush=True)


def write_outputs_incrementally(transcript_dir, video_id, items):
    write_outputs(transcript_dir, video_id, {"items": items})


def format_timecode(seconds):
    seconds = int(seconds or 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecte les images contenant du texte ecrit et ecrit le resultat dans images/analyse."
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
        default=DEFAULT_IMAGE_ANALYZE_MODEL,
        help="Modele vision. Defaut: gpt-5.4-mini",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Nombre d'images envoyees par appel API. Defaut: 20",
    )
    parser.add_argument(
        "--detail",
        choices=("low", "high", "auto"),
        default=DEFAULT_IMAGE_ANALYZE_DETAIL,
        help="Niveau de detail image envoye au modele. Defaut: high",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        help="Nombre maximum de videos a analyser.",
    )
    parser.add_argument(
        "--images-manifest",
        help="Manifeste JSON optionnel produit par la step de filtrage. Si absent, toutes les images sont utilisees.",
    )
    parser.add_argument(
        "--batch-api",
        action="store_true",
        help="Utilise la Batch API OpenAI au lieu des appels synchrones.",
    )
    return parser.parse_args()


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
    print(f"Modele: {args.model}", flush=True)
    print(f"Mode traitement: {'batch' if batch_api else 'synchrone'}", flush=True)
    for video_path in videos:
        images_dir = video_path / "images"
        transcript_dir = video_path / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        images = load_filtered_images(video_path, images_dir)
        if not images:
            print(f"[skip] {video_path.name}: aucune image", flush=True)
            continue
        print(f"[analyse] {video_path.name}: {len(images)} images", flush=True)
        items = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        total_batches = (len(images) + args.batch_size - 1) // args.batch_size
        for batch_index, batch in enumerate(chunks(images, args.batch_size), start=1):
            request_id = f"{video_path.name}_batch_{batch_index:04d}"
            print(f"[batch {batch_index}/{total_batches}] {request_id}: envoi", flush=True)
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
            items = merge_result_items(items, result)
            write_outputs_incrementally(transcript_dir, video_path.name, items)
        print(
            f"[usage total] in={total_input_tokens} out={total_output_tokens} cost=${total_cost:.6f}",
            flush=True,
        )
        print(f"[done] {video_path.name}: {len(items)} items", flush=True)


if __name__ == "__main__":
    main()
