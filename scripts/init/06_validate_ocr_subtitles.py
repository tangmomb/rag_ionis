import argparse
import json
import os
import re
import sys
import unicodedata
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
OCR_PROCESSED_SUFFIX = "_ocr_processed.json"
OCR_PROCESSED_CORRECTED_SUFFIX = "_ocr_processed_corrected.json"
DEFAULT_MODEL = "gpt-5.4-nano"
MAX_OUTPUT_TOKENS = 16
SYSTEM_PROMPT = (
    "Tu verifies des textes detectes par OCR dans des images de video. "
    "Pour chaque texte, dis seulement s'il s'agit vraiment d'un sous-titre affiche a l'ecran. "
    "Si c'est probablement du decor, un nom, un titre, une interface, un mot isole, un logo, "
    "ou une erreur OCR, reponds non. Reponds uniquement par oui ou non."
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


def processed_ocr_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{OCR_PROCESSED_SUFFIX}"


def corrected_ocr_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{OCR_PROCESSED_CORRECTED_SUFFIX}"


def validation_log_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}_ocr_subtitle_validation_log.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_model_name(model):
    compact = str(model).strip().lower().replace("_", "").replace("-", "")
    if compact == "gpt5.4nano":
        return "gpt-5.4-nano"
    return model


def normalize_answer(value):
    normalized = unicodedata.normalize("NFKD", str(value).strip().casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z]+", "", normalized)


def parse_yes_no(answer):
    normalized = normalize_answer(answer)
    if normalized.startswith("oui") or normalized.startswith("yes"):
        return True
    if normalized.startswith("non") or normalized.startswith("no"):
        return False
    raise ValueError(f"Reponse GPT inattendue: {answer!r}")


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


def ask_gpt(client, model, text):
    user_prompt = (
        "A ton avis c'est vraiment du sous titre ou erreur de l'ocr ? "
        "Reponds juste oui ou non.\n\n"
        f"{text}"
    )
    request_log = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=request_log["messages"],
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        answer = response_text(response)
        request_log["api"] = "responses.create"
        request_log["max_output_tokens"] = MAX_OUTPUT_TOKENS
        return answer, request_log

    response = client.chat.completions.create(
        model=model,
        messages=request_log["messages"],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    answer = response.choices[0].message.content or ""
    request_log["api"] = "chat.completions.create"
    request_log["max_completion_tokens"] = MAX_OUTPUT_TOKENS
    return answer, request_log


def openai_client():
    from openai import OpenAI

    return OpenAI()


def validate_items(model, payload):
    corrected_items = []
    logs = []
    reviewed = 0
    kept = 0
    rejected = 0
    client = None

    for item in payload.get("items", []):
        corrected = deepcopy(item)
        kind = str(corrected.get("kind", "")).strip().lower()
        if kind != "subtitle":
            corrected_items.append(corrected)
            continue

        text = str(corrected.get("text", ""))
        reviewed += 1
        if not text.strip():
            answer = "non"
            is_subtitle = False
        else:
            if client is None:
                client = openai_client()
            answer, request_log = ask_gpt(client, model, text)
            answer = answer.strip()
            logs.append(
                {
                    "index": reviewed,
                    "image": corrected.get("image"),
                    "second": corrected.get("second"),
                    "text": text,
                    "request": request_log,
                    "response": answer,
                }
            )
            is_subtitle = parse_yes_no(answer)

        corrected["subtitle_validation"] = {
            "model": model,
            "answer": answer,
            "is_subtitle": is_subtitle,
        }
        if is_subtitle:
            kept += 1
        else:
            rejected += 1
            corrected["previous_kind"] = corrected.get("kind")
            corrected["kind"] = "ocr_error"
            corrected["ocr_error_reason"] = "rejected_by_gpt_subtitle_validation"

        corrected_items.append(corrected)
        verdict = "oui" if is_subtitle else "non"
        print(f"[validate] {reviewed}: {verdict} -> {text}", flush=True)

    return corrected_items, {"reviewed": reviewed, "kept": kept, "rejected": rejected}, logs


def validate_file(model, video_path, force=False):
    source = processed_ocr_path(video_path)
    target = corrected_ocr_path(video_path)
    log_target = validation_log_path(video_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] OCR traite introuvable: {source}")
        return None

    payload = load_json(source)
    corrected_items, stats, logs = validate_items(model, payload)
    result = {
        **payload,
        "items": corrected_items,
        "subtitle_validation": {
            "model": model,
            "source": source.name,
            **stats,
        },
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_payload = {
        "model": model,
        "source": source.name,
        **stats,
        "requests": logs,
    }
    log_target.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[write] {target} ({stats['kept']} sous-titres gardes, {stats['rejected']} rejetes)",
        flush=True,
    )
    print(f"[write] log -> {log_target}", flush=True)
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valide les items OCR kind=subtitle avec OpenAI et produit un JSON OCR corrige."
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
        "--model",
        default=os.getenv("OCR_SUBTITLE_VALIDATION_MODEL", DEFAULT_MODEL),
        help=f"Modele OpenAI de validation. Defaut: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le JSON corrige meme s'il existe deja.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    args.model = normalize_model_name(args.model)
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"Modele validation OCR: {args.model}", flush=True)
    done = 0
    for video_path in videos:
        if validate_file(args.model, video_path, force=args.force):
            done += 1

    print(f"{done} JSON OCR corriges generes.")


if __name__ == "__main__":
    main()
