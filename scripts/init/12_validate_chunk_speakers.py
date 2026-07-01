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
CHUNKS_SUFFIX = "_chunks.json"
CHUNKS_CORRECTED_SUFFIX = "_chunks_corrected.json"
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


def chunks_path(video_path):
    return video_path.parent / "chunks" / f"{video_path.stem}{CHUNKS_SUFFIX}"


def corrected_chunks_path(video_path):
    return video_path.parent / "chunks" / f"{video_path.stem}{CHUNKS_CORRECTED_SUFFIX}"


def validation_log_path(video_path):
    return video_path.parent / "chunks" / f"{video_path.stem}_chunk_speaker_validation_log.json"


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


def openai_client():
    from openai import OpenAI

    return OpenAI()


def ask_gpt(client, model, speakers):
    user_prompt = (
        "Parmi cette liste de speakers, lesquels sont vraiment des personnes ? "
        "Reponds uniquement par un tableau JSON de noms valides, sans commentaire.\n\n"
        + json.dumps(speakers, ensure_ascii=False, indent=2)
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
            max_output_tokens=512,
        )
        answer = response_text(response)
        request_log["api"] = "responses.create"
        request_log["max_output_tokens"] = 512
        return answer, request_log

    response = client.chat.completions.create(
        model=model,
        messages=request_log["messages"],
        max_completion_tokens=512,
    )
    answer = response.choices[0].message.content or ""
    request_log["api"] = "chat.completions.create"
    request_log["max_completion_tokens"] = 512
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


def validate_file(model, video_path, force=False):
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
        answer, request_log = ask_gpt(openai_client(), model, speakers)
        answer = answer.strip()
        valid_speakers = parse_valid_speakers(answer, speakers)
        logs = [
            {
                "speakers": speakers,
                "request": request_log,
                "response": answer,
            }
        ]
    else:
        answer = "[]"
        valid_speakers = []
        logs = []

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
        "requests": logs,
    }
    log_target.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[write] {target} ({len(valid_speakers)} speakers gardes, "
        f"{len(speakers) - len(valid_speakers)} rejetes)",
        flush=True,
    )
    print(f"[write] log -> {log_target}", flush=True)
    return target


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
        "--force",
        action="store_true",
        help="Regenere le JSON chunks corrige meme s'il existe deja.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    args.model = normalize_model_name(args.model)
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"Modele validation speakers: {args.model}", flush=True)
    done = 0
    for video_path in videos:
        if validate_file(args.model, video_path, force=args.force):
            done += 1

    print(f"{done} JSON chunks corriges generes.")


if __name__ == "__main__":
    main()
