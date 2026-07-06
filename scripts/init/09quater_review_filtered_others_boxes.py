import argparse
import base64
import json
import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SOURCE_DIRNAME = "ocr_processed_filtered_others_boxes"
SOURCE_MANIFEST_NAME = "manifest.json"
OUTPUT_DIRNAME = "ocr_processed_filtered_others_boxes_review"
DEFAULT_MODEL = "gpt-5.2"
SYSTEM_PROMPT = (
    "Tu analyses une image complete provenant d'une video. "
    "Ta tache: 1) regarder d'abord l'ensemble de l'image pour juger le contexte visuel global, "
    "2) analyser ensuite le texte situe dans la zone encadree en rouge, "
    "3) dire si ce texte ressemble a du texte ajoute au montage "
    "(titre, lower third, intertitre, texte graphique, habillage, texte pose en post-production) "
    "plutot qu'a du texte capture naturellement dans la scene, "
    "4) verifier enfin si le texte OCR fourni contient une erreur de lecture, et si oui proposer une correction. "
    "Regle importante: si le texte encadre n'est pas parfaitement lisible, net, propre et clairement detache du decor, "
    "alors considere que ce n'est PAS du texte ajoute au montage. "
    "Autre regle importante: si le texte est bien net et passe par-dessus plusieurs elements differents "
    "(personnes, vetements, objets, decors, arriere-plan, etc.), alors considere que c'est FORCEMENT un ajout au montage. "
    "Reponds uniquement en JSON avec les cles: "
    "has_ocr_error (boolean), corrected_text (string), is_added_in_edit (boolean), confidence (number entre 0 et 1), reason (string court). "
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


def source_dir(video_path):
    return video_path.parent / "ocr" / SOURCE_DIRNAME


def source_manifest_path(video_path):
    return source_dir(video_path) / SOURCE_MANIFEST_NAME


def output_dir(video_path):
    return video_path.parent / "ocr" / OUTPUT_DIRNAME


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
    corrected_text = " ".join(str(parsed.get("corrected_text", "")).split()).strip()
    return {
        "has_ocr_error": bool(parsed.get("has_ocr_error")),
        "corrected_text": corrected_text,
        "is_added_in_edit": bool(parsed.get("is_added_in_edit")),
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))),
        "reason": " ".join(str(parsed.get("reason", "")).split()).strip(),
    }


def review_one_image(client, model, image_path, item):
    user_prompt = (
        "Regarde d'abord l'image complete pour comprendre la scene et le contexte general. "
        "Ensuite concentre-toi sur le texte dans la zone encadree en rouge. "
        "Decide d'abord si ce texte encadre est un ajout au montage ou non, en tenant compte du contexte global de l'image. "
        "Si le texte n'est pas parfaitement lisible ou net, reponds que ce n'est pas du montage. "
        "Si le texte est bien net et traverse visiblement plusieurs elements differents de l'image, reponds que c'est du montage. "
        "Apres cette decision, verifie si le texte OCR fourni correspond bien a ce qui est visible dans la zone rouge, et corrige-le si besoin. "
        "Contexte OCR:\n"
        f"- timecode: {item.get('timecode', '')}\n"
        f"- texte OCR: {item.get('text', '')}\n"
        "Reponds uniquement avec le JSON demande."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": encode_image_data_url(image_path), "detail": "high"},
            ],
        },
    ]
    response = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=200,
    )
    answer = response_text(response).strip()
    return answer, {
        "api": "responses.create",
        "model": model,
        "max_output_tokens": 200,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def safe_stem(value):
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return cleaned.strip("_") or "item"


def write_text(path, content):
    path.write_text(str(content), encoding="utf-8")


def review_video(video_path, model, force=False, limit_images=None):
    source_manifest = source_manifest_path(video_path)
    if not source_manifest.exists():
        print(f"[skip] manifest introuvable: {source_manifest}")
        return None

    source_payload = load_manifest(source_manifest)
    items = list(source_payload.get("items", []))
    if limit_images is not None:
        items = items[:limit_images]

    target_dir = output_dir(video_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] {summary_path.name} existe deja")
        return summary_path

    client = openai_client()
    decisions = []
    for index, item in enumerate(items, start=1):
        crop_name = item.get("crop")
        if not crop_name:
            continue
        image_path = source_dir(video_path) / crop_name
        if not image_path.exists():
            print(f"[skip] crop introuvable: {image_path}", flush=True)
            continue

        review_name = f"{index:03d}__{safe_stem(Path(crop_name).stem)}"
        review_dir = target_dir / review_name
        if force and review_dir.exists():
            for child in review_dir.iterdir():
                if child.is_file():
                    child.unlink()
        review_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, review_dir / image_path.name)

        answer, request_log = review_one_image(client, model, image_path, item)
        parsed = parse_json_answer(answer)
        if not parsed["corrected_text"]:
            parsed["corrected_text"] = " ".join(str(item.get("text", "")).split()).strip()
        write_text(review_dir / "prompt.txt", request_log["messages"][1]["content"])
        write_text(review_dir / "response.txt", answer)
        (review_dir / "request.json").write_text(
            json.dumps(request_log, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (review_dir / "decision.json").write_text(
            json.dumps(
                {
                    "crop": crop_name,
                    "timecode": item.get("timecode"),
                    "text": item.get("text"),
                    "image": item.get("image"),
                    "box": item.get("box"),
                    **parsed,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        decisions.append(
            {
                "review_dir": review_name,
                "crop": crop_name,
                "timecode": item.get("timecode"),
                "text": item.get("text"),
                "image": item.get("image"),
                "box": item.get("box"),
                **parsed,
            }
        )
        print(
            f"[review {index}/{len(items)}] {crop_name}: added={str(parsed['is_added_in_edit']).lower()} conf={parsed['confidence']:.2f}",
            flush=True,
        )

    summary_payload = {
        "model": model,
        "source_dir": f"ocr/{SOURCE_DIRNAME}",
        "source_manifest": f"ocr/{SOURCE_DIRNAME}/{SOURCE_MANIFEST_NAME}",
        "output_dir": f"ocr/{OUTPUT_DIRNAME}",
        "reviewed_count": len(decisions),
        "added_in_edit_count": sum(1 for item in decisions if item["is_added_in_edit"]),
        "not_added_count": sum(1 for item in decisions if not item["is_added_in_edit"]),
        "items": decisions,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {summary_path}", flush=True)
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demande a GPT-5.4-nano si le texte dans la zone rouge des images 'others' ressemble a du texte ajoute au montage."
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
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if review_video(video_path, args.model, force=args.force, limit_images=args.limit_images):
            done += 1
    print(f"{done} review(s) GPT generee(s).")


if __name__ == "__main__":
    main()
