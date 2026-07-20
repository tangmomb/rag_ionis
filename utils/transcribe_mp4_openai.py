import json
import sys
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_SEARCH_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = {".mp4"}
OUTPUT_DIR = Path(__file__).resolve().parent / "transcriptions_openai"
MODEL_OPTIONS = [
    {
        "label": "whisper-1",
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
        "json_suffix": "_openai_whisper_transcript.verbose.json",
    },
    {
        "label": "gpt-4o-transcribe",
        "model": "gpt-4o-transcribe",
        "response_format": "json",
        "timestamp_granularities": None,
        "json_suffix": "_openai_gpt4o_transcribe.json",
    },
    {
        "label": "gpt-4o-mini-transcribe",
        "model": "gpt-4o-mini-transcribe",
        "response_format": "json",
        "timestamp_granularities": None,
        "json_suffix": "_openai_gpt4o_mini_transcribe.json",
    },
    {
        "label": "gpt-4o-transcribe-diarize",
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "timestamp_granularities": None,
        "json_suffix": "_openai_gpt4o_transcribe_diarize.json",
        "chunking_strategy": "auto",
    },
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_mp4_files(search_dir):
    if not search_dir.exists():
        return []
    return sorted(
        (
            path
            for path in search_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )


def prompt_video_path(candidates):
    if candidates:
        print("MP4 trouves :")
        for index, path in enumerate(candidates, start=1):
            print(f"  {index}. {path}")
        print()

    prompt = "Choisis un numero de fichier MP4 ou colle un chemin complet: "
    while True:
        raw_value = input(prompt).strip().strip('"')
        if not raw_value:
            print("Saisie vide, recommence.")
            continue

        if raw_value.isdigit() and candidates:
            selected_index = int(raw_value)
            if 1 <= selected_index <= len(candidates):
                return candidates[selected_index - 1]
            print("Numero hors liste, recommence.")
            continue

        candidate_path = Path(raw_value)
        if candidate_path.exists() and candidate_path.is_file() and candidate_path.suffix.lower() == ".mp4":
            return candidate_path
        print("Fichier MP4 introuvable, recommence.")


def prompt_model_choices():
    print("Modeles disponibles :")
    print("  0. tous les modeles")
    for index, option in enumerate(MODEL_OPTIONS, start=1):
        print(f"  {index}. {option['label']}")
    print()

    prompt = "Choisis un numero de modele, ou 0 pour tous (defaut: 1): "
    while True:
        raw_value = input(prompt).strip()
        if not raw_value:
            return [MODEL_OPTIONS[0]]
        if raw_value.isdigit():
            selected_index = int(raw_value)
            if selected_index == 0:
                return MODEL_OPTIONS
            if 1 <= selected_index <= len(MODEL_OPTIONS):
                return [MODEL_OPTIONS[selected_index - 1]]
        print("Numero hors liste, recommence.")


def transcribe_video(client, video_path, model_option):
    request = {
        "model": model_option["model"],
        "language": "fr",
        "response_format": model_option["response_format"],
    }
    if model_option["timestamp_granularities"]:
        request["timestamp_granularities"] = model_option["timestamp_granularities"]
    if model_option.get("chunking_strategy"):
        request["chunking_strategy"] = model_option["chunking_strategy"]

    with video_path.open("rb") as video_file:
        return client.audio.transcriptions.create(
            file=video_file,
            **request,
        )


def response_text(payload):
    if hasattr(payload, "text") and payload.text:
        return str(payload.text).strip()

    if isinstance(payload, dict):
        text = payload.get("text")
        if text:
            return str(text).strip()

    text = getattr(payload, "text", None)
    return str(text).strip() if text else ""


def response_jsonable(payload):
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if isinstance(payload, dict):
        return payload
    return {"text": response_text(payload)}


def write_outputs(video_path, payload, model_option):
    transcript_text = response_text(payload)
    model_slug = model_option["model"].replace("-", "_")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text_path = OUTPUT_DIR / f"{video_path.stem}_openai_{model_slug}_transcript.txt"
    json_path = OUTPUT_DIR / f"{video_path.stem}{model_option['json_suffix']}"
    text_path.write_text(transcript_text + ("\n" if transcript_text else ""), encoding="utf-8")
    json_path.write_text(
        json.dumps(response_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return text_path, json_path


def main():
    from openai import OpenAI

    load_dotenv(override=True)
    client = OpenAI()
    search_dir = DEFAULT_SEARCH_DIR
    candidates = find_mp4_files(search_dir)

    if not candidates:
        print(f"Aucun MP4 trouve sous {search_dir}.")
        print("Tu peux quand meme coller un chemin complet vers un fichier .mp4.")

    video_path = prompt_video_path(candidates)
    model_options = prompt_model_choices()
    print(f"[transcription] {video_path}")
    failures = 0
    for model_option in model_options:
        print(
            f"[modele] {model_option['model']} "
            f"(response_format={model_option['response_format']})"
        )
        try:
            payload = transcribe_video(client, video_path, model_option)
            text_path, json_path = write_outputs(video_path, payload, model_option)
        except Exception as error:
            failures += 1
            print(f"[erreur] {model_option['model']}: {error}", file=sys.stderr)
            continue
        print(f"[ok] transcription texte: {text_path}")
        print(f"[ok] transcription json: {json_path}")

    if failures:
        raise SystemExit(failures)


if __name__ == "__main__":
    main()
