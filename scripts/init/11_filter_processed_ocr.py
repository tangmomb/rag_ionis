import argparse
import json
import sys
from pathlib import Path

from pipeline_analysis import update_analysed_infos
from ocr_processed_filtering import (
    build_filtered_payload,
    filter_overlay_items,
    filtered_ocr_path,
    group_items_by_kind,
    load_groupable_items,
    load_overlay_items,
    normalize_text,
    processed_ocr_source_path,
)
from pipeline_paths import existing_ocr_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
OCR_FOOTAGE_NAME = "raw_ocr_footage_frames.json"
LEGACY_OCR_FOOTAGE_NAME = "ocr_footage.json"

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


def footage_ocr_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / OCR_FOOTAGE_NAME
    legacy = video_ocr_dir / LEGACY_OCR_FOOTAGE_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_footage_occurrences(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    occurrences_by_text = {}

    for item in payload.get("items", []):
        image_name = item.get("image")
        if not image_name:
            continue
        for raw in item.get("raw", []):
            polys = list(raw.get("dt_polys", []))
            texts = list(raw.get("rec_texts", []))
            scores = list(raw.get("rec_scores", []))
            count = min(len(polys), len(texts))
            for index in range(count):
                text = " ".join(str(texts[index]).split()).strip()
                normalized = normalize_text(text)
                if not normalized:
                    continue
                box = polys[index]
                occurrence = {
                    "image": image_name,
                    "text": text,
                    "box": box,
                }
                if index < len(scores):
                    try:
                        occurrence["score"] = float(scores[index])
                    except (TypeError, ValueError):
                        pass
                occurrences_by_text.setdefault(normalized, []).append(occurrence)

    return occurrences_by_text


def enrich_others_with_footage_occurrences(grouped_kind_details, video_path):
    others_details = grouped_kind_details.get("others")
    if not isinstance(others_details, dict) or not others_details:
        return

    footage_path = footage_ocr_path(video_path)
    if not footage_path.exists():
        print(f"[skip] OCR footage introuvable: {footage_path}")
        return

    occurrences_by_text = load_footage_occurrences(footage_path)
    for timecode, detail in others_details.items():
        if not isinstance(detail, dict):
            continue
        text = " ".join(str(detail.get("text", "")).split()).strip()
        normalized = normalize_text(text)
        occurrences = list(occurrences_by_text.get(normalized, []))
        detail["all_occurrences"] = occurrences
        detail["all_occurrence_count"] = len(occurrences)
        detail["all_occurrences_source"] = relative_to_video_dir(footage_path, video_path)


def filter_processed_ocr(video_path, force=False):
    source = processed_ocr_source_path(video_path)
    target = filtered_ocr_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] OCR processed introuvable: {source}")
        return None

    all_items, payload = load_groupable_items(source)
    overlay_items, _ = load_overlay_items(source)
    filtered_items = filter_overlay_items(overlay_items)
    grouped_kinds, grouped_kind_details = group_items_by_kind(all_items, filtered_items)
    enrich_others_with_footage_occurrences(grouped_kind_details, video_path)
    filtered_payload = build_filtered_payload(
        payload,
        source.name,
        filtered_items,
        grouped_kinds,
        grouped_kind_details,
    )
    target.write_text(json.dumps(filtered_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "filter_ocr_processed",
        {
            "status": "done",
            "source": relative_to_video_dir(source, video_path),
            "filtered_file": relative_to_video_dir(target, video_path),
            "filtered_item_count": len(filtered_items),
            "kind_count": len(grouped_kinds),
        },
    )
    print(f"[ok] {target}")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtre les overlays OCR et produit filtered_ocr_overlays.json."
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
        "--force",
        action="store_true",
        help="Regenere les fichiers filtres meme s'ils existent deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if filter_processed_ocr(video_path, force=args.force):
            done += 1

    print(f"{done} fichiers OCR filtered generes.")


if __name__ == "__main__":
    main()
