import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
INIT_DIR = CURRENT_DIR.parent
if str(INIT_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_DIR))
os.environ["PIPELINE_TRANSCRIPTS_DIR_NAME"] = "transcripts_ocr"

from common.pipeline_analysis import update_analysed_infos  # noqa: E402
from common.pipeline_paths import existing_transcripts_dir, relative_to_video_dir  # noqa: E402


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
DEFAULT_MODEL = "gpt-5.4-nano"
SOURCE_NAME = "ocr_subtitles_timecoded.txt"
TARGET_NAME = "ocr_subtitles_timecoded_corrected.txt"
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


def request_spacing_fix(client, model, text):
    user_prompt = (
        "Corrige uniquement les espaces manquants ou en trop dans cette ligne. "
        "Si rien n'est a changer, renvoie exactement le meme texte.\n\n"
        f"Ligne: {text}"
    )
    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=256,
        )
        return response_text(response)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=256,
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


def process_video(client, model, video_path, force=False):
    source = subtitle_source_path(video_path)
    target = subtitle_target_path(video_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] sous-titres OCR timecodes introuvables: {source}")
        return None

    lines = source.read_text(encoding="utf-8").splitlines()
    output_lines = []
    changed_count = 0
    request_count = 0
    for index, line in enumerate(lines, start=1):
        match = SUBTITLE_LINE.match(line)
        if not match:
            if line.strip():
                output_lines.append(line.strip())
            continue
        timecode, text = match.groups()
        cleaned_text = " ".join(str(text).split()).strip()
        if not cleaned_text:
            output_lines.append(f"[{timecode}]")
            continue
        fixed_text = request_spacing_fix(client, model, cleaned_text)
        normalized_fixed = normalize_answer(fixed_text)
        if normalized_fixed != cleaned_text:
            changed_count += 1
        output_lines.append(corrected_line(timecode, cleaned_text, normalized_fixed))
        request_count += 1
        print(f"[line {index}/{len(lines)}] {video_path.name}", flush=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(output_lines).strip() + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "ocr_subtitle_spacing",
        {
            "status": "done",
            "source": relative_to_video_dir(source, video_path),
            "corrected_file": relative_to_video_dir(target, video_path),
            "line_count": len(output_lines),
            "request_count": request_count,
            "changed_count": changed_count,
            "model": model,
        },
    )
    print(f"[ok] {target}")
    return target


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
    return parser.parse_args()


def main():
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)
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
    if openai_mode == "batch":
        print("[warn] mode OpenAI global=batch, mais cette step utilise actuellement le mode normal.", flush=True)
    client = openai_client()
    done = 0
    for video_path in videos:
        if process_video(client, args.model, video_path, force=args.force):
            done += 1

    print(f"{done} fichier(s) de sous-titres corrige(s).")


if __name__ == "__main__":
    main()
