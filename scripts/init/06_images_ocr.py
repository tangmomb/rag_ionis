import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SECOND_PATTERN = re.compile(r"^seconde_(\d+(?:_\d+)?)$")
TIMECODE_PATTERN = re.compile(r"^(?:(\d{2})_)?(\d{2})_(\d{2})$")

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


def analyze_batch(client, model, batch, detail):
    content = [
        {
            "type": "input_text",
            "text": "Analyse ces images extraites d'une video. Chaque image est nommee par seconde. Retranscris de facon litterale tous les textes ecrits visibles a l'ecran, y compris les sous-titres, les noms, les titres, les questions, les slides, les panneaux et les logos texte importants. Ne fais pas de resume si le texte est lisible: recopie au plus pres le texte exact vu a l'ecran, ligne par ligne si besoin. Ignore les images sans texte lisible. Ne retourne une liste vide que si aucune image du lot ne contient de texte lisible. Reponds uniquement en JSON valide avec la forme {\"items\":[{\"image\":\"00_12.jpg\",\"text\":\"texte lu ou extrait litteral\",\"kind\":\"name|question|slide|title|subtitle|other\",\"confidence\":\"low|medium|high\"}]}.",
        }
    ]

    for image_path in batch:
        content.append({"type": "input_text", "text": f"Image: {image_path.name}"})
        content.append({"type": "input_image", "image_url": image_data_url(image_path), "detail": detail})

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=2000,
        temperature=0,
    )
    return parse_json_response(response.output_text)


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


def merge_results(results):
    items = []
    for result in results:
        for item in result.get("items", []):
            items.append(normalize_item_timecode(item))
    return {"items": deduplicate_items(sorted(items, key=lambda item: item.get("second", 0)))}


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
    print(f"[ok] {json_path}")


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
        default=os.getenv("OPENAI_IMAGE_ANALYZE_MODEL", "gpt-5.4-nano"),
        help="Modele vision. Defaut: OPENAI_IMAGE_ANALYZE_MODEL ou gpt-5.4-nano",
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
        default="low",
        help="Niveau de detail image envoye au modele. Defaut: low",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        help="Nombre maximum de videos a analyser.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    client = OpenAI()

    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(image_video_dirs(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"Modele: {args.model}")
    for video_path in videos:
        images_dir = video_path / "images"
        transcript_dir = video_path / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        images = image_files(images_dir)
        if not images:
            print(f"[skip] {video_path.name}: aucune image")
            continue
        print(f"[analyse] {video_path.name}: {len(images)} images")
        results = []
        for batch in chunks(images, args.batch_size):
            results.append(analyze_batch(client, args.model, batch, args.detail))
        merged = merge_results(results)
        write_outputs(transcript_dir, video_path.name, merged)


if __name__ == "__main__":
    main()
