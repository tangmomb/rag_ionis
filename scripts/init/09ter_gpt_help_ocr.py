import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
OCR_DIR_NAME = "ocr"
FILTERED_OCR_NAME = "ocr_processed_filtered.json"
FILTERED_GPT_OCR_NAME = "ocr_processed_filtered_gpt.json"
GPT_HELP_DIR_NAME = "gpt_help"
MANIFEST_NAME = "manifest.json"
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_CONTEXT_WINDOW_SECONDS = 5
SYSTEM_PROMPT = (
    "Tu aides a valider des mots-cles OCR animes extraits d'une video interview. "
    "Pour chaque mot-cle, commence par proposer une correction orthographique courte et plausible en francais, "
    "en gardant le sens attendu. Pour cette correction, tu peux ajouter, modifier ou supprimer au maximum 3 caracteres. "
    "Ensuite dis si le mot-cle corrige parait coherent avec ce que la personne dit "
    "dans l'extrait fourni. La coherence signifie que la personne a bien plus ou moins exprime l'idee mise en avant "
    "par le mot-cle. Reponds uniquement en JSON avec les champs corrected_keyword, coherent, confidence, rationale."
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


def filtered_ocr_path(video_path):
    return video_path.parent / OCR_DIR_NAME / FILTERED_OCR_NAME


def gpt_help_dir(video_path):
    return video_path.parent / OCR_DIR_NAME / GPT_HELP_DIR_NAME


def filtered_gpt_ocr_path(video_path):
    return video_path.parent / OCR_DIR_NAME / FILTERED_GPT_OCR_NAME


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_timecode(value):
    parts = [int(part) for part in str(value).strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Timecode invalide: {value}")


def subtitle_items(payload):
    subtitles = payload.get("kinds", {}).get("subtitle", {})
    if not isinstance(subtitles, dict):
        return []
    items = []
    for timecode, text in subtitles.items():
        cleaned_text = " ".join(str(text).split()).strip()
        if not cleaned_text:
            continue
        items.append(
            {
                "timecode": str(timecode).strip(),
                "second": parse_timecode(timecode),
                "text": cleaned_text,
            }
        )
    return sorted(items, key=lambda item: (item["second"], item["timecode"]))


def others_items(payload):
    others = payload.get("kinds", {}).get("others", {})
    if not isinstance(others, dict):
        return []
    items = []
    for timecode, text in others.items():
        cleaned_text = " ".join(str(text).split()).strip()
        if not cleaned_text:
            continue
        items.append(
            {
                "timecode": str(timecode).strip(),
                "second": parse_timecode(timecode),
                "text": cleaned_text,
            }
        )
    return sorted(items, key=lambda item: (item["second"], item["timecode"]))


def context_subtitles(subtitles, center_second, window_seconds):
    selected = [
        item
        for item in subtitles
        if abs(item["second"] - center_second) <= window_seconds
    ]
    selected.sort(key=lambda item: (item["second"], item["timecode"]))
    return selected


def build_question(keyword_item, subtitle_context, window_seconds):
    if subtitle_context:
        context_lines = "\n".join(f"- {item['text']}" for item in subtitle_context)
    else:
        context_lines = "- Aucun sous-titre trouve dans cette fenetre"
    return (
        f"Voici un mot cle anime apparu a l'ecran: \"{keyword_item['text']}\".\n"
        f"Voici ce que la personne dit lors de l'apparition de ce mot cle, dans une fenetre de plus ou moins {window_seconds} secondes:\n"
        f"{context_lines}\n\n"
        "Commence par corriger l'orthographe du mot cle. Pour cette correction, tu peux ajouter, modifier ou supprimer au maximum 3 caracteres. "
        "Puis dis moi si ca parait coherent. "
        "La personne doit avoir plus ou moins dit ce que le mot cle met en avant.\n"
        "Reponds uniquement en JSON avec les champs corrected_keyword, coherent, confidence, rationale."
    )


def openai_client():
    from openai import OpenAI

    return OpenAI()


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


def ask_gpt(client, model, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    request_log = {
        "model": model,
        "messages": messages,
    }
    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=messages,
            max_output_tokens=512,
        )
        answer = response_text(response)
        request_log["api"] = "responses.create"
        request_log["max_output_tokens"] = 512
        return answer, request_log

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=512,
    )
    answer = response.choices[0].message.content or ""
    request_log["api"] = "chat.completions.create"
    request_log["max_completion_tokens"] = 512
    return answer, request_log


def parse_json_answer(answer):
    text = str(answer).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Reponse GPT inattendue: {answer!r}")
    return parsed


def safe_slug(value):
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")
    return slug or "item"


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return bool(value)


def cleaned_keyword_text(value, fallback):
    text = " ".join(str(value or "").split()).strip()
    return text or fallback


def build_gpt_filtered_payload(source_payload, entries):
    output_payload = json.loads(json.dumps(source_payload))
    kinds = output_payload.setdefault("kinds", {})
    source_others = kinds.get("others", {})
    if not isinstance(source_others, dict):
        source_others = {}

    kept_others = {}
    removed_count = 0
    corrected_count = 0
    for entry in entries:
        timecode = entry["keyword_timecode"]
        original_text = str(source_others.get(timecode, entry["keyword_text"]))
        coherent = as_bool(entry.get("coherent"))
        if not coherent:
            removed_count += 1
            continue
        corrected_text = cleaned_keyword_text(entry.get("corrected_keyword"), original_text)
        if corrected_text != original_text:
            corrected_count += 1
        kept_others[timecode] = corrected_text

    kinds["others"] = kept_others
    output_payload["source"] = FILTERED_OCR_NAME
    output_payload["gpt_help"] = {
        "source": "gpt_help/manifest.json",
        "removed_false_count": removed_count,
        "corrected_true_count": corrected_count,
        "kept_count": len(kept_others),
    }
    return output_payload


def write_help_for_video(model, video_path, window_seconds, force=False):
    source = filtered_ocr_path(video_path)
    if not source.exists():
        print(f"[skip] OCR filtered introuvable: {source}")
        return None

    payload = load_json(source)
    other_items = others_items(payload)
    if not other_items:
        print(f"[skip] aucun kind=others dans {source.name}")
        return None

    subtitles = subtitle_items(payload)
    output_dir = gpt_help_dir(video_path)
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists() and not force:
        print(f"[skip] {manifest_path} existe deja")
        return manifest_path

    output_dir.mkdir(parents=True, exist_ok=True)
    client = openai_client()
    entries = []
    for index, item in enumerate(other_items, start=1):
        subtitle_context = context_subtitles(subtitles, item["second"], window_seconds)
        question = build_question(item, subtitle_context, window_seconds)
        answer, request_log = ask_gpt(client, model, question)
        parsed_answer = parse_json_answer(answer)
        file_name = f"{index:03d}_{item['timecode'].replace(':', '-')}_{safe_slug(item['text'])}.json"
        target = output_dir / file_name
        record = {
            "model": model,
            "keyword": item,
            "window_seconds": window_seconds,
            "subtitle_context": subtitle_context,
            "question": question,
            "response_text": answer.strip(),
            "response_json": parsed_answer,
            "request": request_log,
        }
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        entries.append(
            {
                "keyword_timecode": item["timecode"],
                "keyword_text": item["text"],
                "file": target.name,
                "corrected_keyword": parsed_answer.get("corrected_keyword"),
                "coherent": parsed_answer.get("coherent"),
                "confidence": parsed_answer.get("confidence"),
            }
        )
        print(f"[ok] {video_path.name}: {item['timecode']} {item['text']} -> {target.name}", flush=True)

    filtered_gpt_path = filtered_gpt_ocr_path(video_path)
    filtered_gpt_payload = build_gpt_filtered_payload(payload, entries)
    filtered_gpt_path.write_text(json.dumps(filtered_gpt_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "source": source.name,
        "model": model,
        "window_seconds": window_seconds,
        "reviewed_count": len(other_items),
        "filtered_gpt_file": filtered_gpt_path.name,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] filtered -> {filtered_gpt_path}", flush=True)
    print(f"[ok] manifest -> {manifest_path}", flush=True)
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demande a GPT un avis sur les mots-cles OCR kind=others a partir du contexte subtitle."
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
        default=os.getenv("OCR_GPT_HELP_MODEL", DEFAULT_MODEL),
        help=f"Modele OpenAI utilise. Defaut: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW_SECONDS,
        help=f"Fenetre de contexte subtitle autour du mot cle. Defaut: {DEFAULT_CONTEXT_WINDOW_SECONDS}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le dossier gpt_help meme s'il existe deja.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"Modele GPT help: {args.model}", flush=True)
    done = 0
    for video_path in videos:
        if write_help_for_video(args.model, video_path, args.window_seconds, force=args.force):
            done += 1

    print(f"{done} dossier(s) gpt_help generes.")


if __name__ == "__main__":
    main()
