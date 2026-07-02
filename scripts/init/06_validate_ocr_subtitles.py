import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
OCR_PROCESSED_SUFFIX = "_ocr_processed.json"
OCR_PROCESSED_CORRECTED_SUFFIX = "_ocr_processed_corrected.json"
SPACY_FRENCH_MODEL = os.environ.get("SPACY_FRENCH_MODEL", "fr_dep_news_trf")
SPACY_REQUIRE_GPU = os.environ.get("SPACY_REQUIRE_GPU", "1").lower() not in {"0", "false", "no"}
VERB_POS = {"AUX", "VERB"}

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


def subtitle_candidates_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}_ocr_subtitle_candidates.txt"


def validation_log_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}_ocr_subtitle_validation_log.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_french_spacy_model():
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "Le package spaCy est absent. Installe les requirements du projet."
        ) from exc

    if SPACY_REQUIRE_GPU:
        try:
            spacy.require_gpu()
        except Exception as exc:
            raise RuntimeError(
                "SPACY_REQUIRE_GPU=1 mais le GPU spaCy n'est pas accessible. "
                "Utilise SPACY_REQUIRE_GPU=0 pour autoriser le CPU."
            ) from exc
    else:
        spacy.prefer_gpu()

    try:
        return spacy.load(SPACY_FRENCH_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"Modele spaCy introuvable: {SPACY_FRENCH_MODEL}. "
            "Reinstalle-le avec `python -m spacy download fr_dep_news_trf`."
        ) from exc


def normalize_text(text):
    return " ".join(str(text).split()).strip()


def format_timecode(seconds):
    seconds = int(seconds or 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def subtitle_candidate_items(payload):
    items = []
    for index, item in enumerate(payload.get("items", []), start=1):
        kind = str(item.get("kind", "")).strip().lower()
        if kind != "subtitle":
            continue
        text = normalize_text(item.get("text", ""))
        items.append({"index": index, "item": item, "text": text})
    return items


def write_subtitle_candidates(path, candidates):
    lines = []
    for candidate in candidates:
        item = candidate["item"]
        second = item.get("second")
        prefix = f"[{format_timecode(second)}] " if second is not None else ""
        lines.append(f"{prefix}{candidate['text']}")
    path.write_text("\n".join(lines).strip() + ("\n" if lines else ""), encoding="utf-8")


def verb_tokens(doc):
    return [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
        }
        for token in doc
        if token.pos_ in VERB_POS
    ]


def validate_items(nlp, payload):
    candidates = subtitle_candidate_items(payload)
    docs = list(nlp.pipe(candidate["text"] for candidate in candidates)) if candidates else []
    by_index = {
        candidate["index"]: {
            "candidate": candidate,
            "doc": doc,
            "verbs": verb_tokens(doc),
        }
        for candidate, doc in zip(candidates, docs)
    }

    corrected_items = []
    logs = []
    reviewed = 0
    kept = 0
    rejected = 0

    for index, item in enumerate(payload.get("items", []), start=1):
        corrected = deepcopy(item)
        kind = str(corrected.get("kind", "")).strip().lower()
        if kind != "subtitle":
            corrected_items.append(corrected)
            continue

        reviewed += 1
        analysis = by_index.get(index, {})
        verbs = analysis.get("verbs", [])
        has_verb = bool(verbs)
        text = normalize_text(corrected.get("text", ""))

        if has_verb:
            kept += 1
        else:
            rejected += 1
            corrected["kind"] = "other"

        logs.append(
            {
                "index": reviewed,
                "image": corrected.get("image"),
                "second": corrected.get("second"),
                "text": text,
                "has_verb": has_verb,
                "verbs": verbs,
            }
        )
        verdict = "subtitle" if has_verb else "other"
        print(f"[validate] {reviewed}: {verdict} -> {text}", flush=True)
        corrected_items.append(corrected)

    return corrected_items, {"reviewed": reviewed, "kept": kept, "rejected": rejected}, logs, candidates


def validate_file(nlp, video_path, force=False):
    source = processed_ocr_path(video_path)
    target = corrected_ocr_path(video_path)
    candidates_target = subtitle_candidates_path(video_path)
    log_target = validation_log_path(video_path)
    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] OCR traite introuvable: {source}")
        return None

    payload = load_json(source)
    corrected_items, stats, logs, candidates = validate_items(nlp, payload)
    write_subtitle_candidates(candidates_target, candidates)
    result = {
        **payload,
        "items": corrected_items,
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_payload = {
        "engine": "spacy",
        "model": SPACY_FRENCH_MODEL,
        "rule": "has_verb",
        "source": source.name,
        "candidates": candidates_target.name,
        **stats,
        "items": logs,
    }
    log_target.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[write] {target} ({stats['kept']} sous-titres gardes, {stats['rejected']} reclasses other)",
        flush=True,
    )
    print(f"[write] candidates -> {candidates_target}", flush=True)
    print(f"[write] log -> {log_target}", flush=True)
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valide les items OCR kind=subtitle avec spaCy has_verb et produit un JSON OCR corrige."
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
        help="Regenere le JSON corrige meme s'il existe deja.",
    )
    return parser.parse_args()


def main():
    global SPACY_FRENCH_MODEL, SPACY_REQUIRE_GPU

    load_dotenv(override=True)
    SPACY_FRENCH_MODEL = os.environ.get("SPACY_FRENCH_MODEL", SPACY_FRENCH_MODEL)
    SPACY_REQUIRE_GPU = os.environ.get("SPACY_REQUIRE_GPU", "1").lower() not in {"0", "false", "no"}
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"Validation OCR subtitles: spaCy {SPACY_FRENCH_MODEL}, rule=has_verb", flush=True)
    nlp = load_french_spacy_model()
    done = 0
    for video_path in videos:
        if validate_file(nlp, video_path, force=args.force):
            done += 1

    print(f"{done} JSON OCR corriges generes.")


if __name__ == "__main__":
    main()
