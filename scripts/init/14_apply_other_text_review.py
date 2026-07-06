import argparse
import json
import sys
from pathlib import Path

from pipeline_paths import existing_ocr_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SOURCE_FILTERED_NAME = "filtered_ocr_overlays.json"
LEGACY_SOURCE_FILTERED_NAME = "ocr_processed_filtered.json"
SOURCE_REVIEW_DIRNAME = "other_text_gpt_review"
LEGACY_SOURCE_REVIEW_DIRNAME = "ocr_processed_filtered_others_boxes_review"
SOURCE_REVIEW_SUMMARY_NAME = "review_summary.json"
LEGACY_SOURCE_REVIEW_SUMMARY_NAME = "summary.json"
OUTPUT_FILTERED_NAME = "reviewed_ocr_overlays.json"
LEGACY_OUTPUT_FILTERED_NAME = "ocr_processed_filtered_reviewed.json"

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


def filtered_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / SOURCE_FILTERED_NAME
    legacy = video_ocr_dir / LEGACY_SOURCE_FILTERED_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def review_summary_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / SOURCE_REVIEW_DIRNAME / SOURCE_REVIEW_SUMMARY_NAME
    legacy = video_ocr_dir / LEGACY_SOURCE_REVIEW_DIRNAME / LEGACY_SOURCE_REVIEW_SUMMARY_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def output_path(video_path):
    video_ocr_dir = existing_ocr_dir(video_path)
    preferred = video_ocr_dir / OUTPUT_FILTERED_NAME
    legacy = video_ocr_dir / LEGACY_OUTPUT_FILTERED_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def review_items_by_entry_id(summary_payload):
    indexed = {}
    for item in summary_payload.get("items", []):
        entry_id = str(item.get("entry_id", "")).strip()
        if entry_id:
            indexed[entry_id] = item
            continue
        timecode = str(item.get("timecode", "")).strip()
        if timecode:
            indexed[timecode] = item
    return indexed


def apply_review_to_filtered(filtered_payload, summary_payload, video_path):
    review_by_entry_id = review_items_by_entry_id(summary_payload)
    kinds = filtered_payload.get("kinds", {})
    kinds_details = filtered_payload.get("kinds_details", {})
    others = dict(kinds.get("others", {}))
    others_details = dict(kinds_details.get("others", {}))

    removed = []
    corrected = []
    untouched = []

    for entry_id, text in list(others.items()):
        review = review_by_entry_id.get(entry_id)
        if not review:
            untouched.append(entry_id)
            continue

        detail = others_details.get(entry_id)
        if not isinstance(detail, dict):
            detail = {"timecode": entry_id, "text": text}
            others_details[entry_id] = detail

        is_added = bool(review.get("is_added_in_edit"))
        corrected_text = " ".join(str(review.get("corrected_text", "")).split()).strip()
        has_ocr_error = bool(review.get("has_ocr_error"))

        if not is_added:
            others.pop(entry_id, None)
            others_details.pop(entry_id, None)
            removed.append(entry_id)
            continue

        if has_ocr_error and corrected_text:
            others[entry_id] = corrected_text
            detail["text"] = corrected_text
            corrected.append(entry_id)

        detail["review"] = {
            "has_ocr_error": has_ocr_error,
            "corrected_text": corrected_text or str(detail.get("text", text)),
            "is_added_in_edit": is_added,
            "confidence": review.get("confidence"),
            "reason": review.get("reason"),
            "review_dir": review.get("review_dir"),
            "crop": review.get("crop"),
        }

        if entry_id not in corrected:
            untouched.append(entry_id)

    filtered_payload["kinds"]["others"] = others
    filtered_payload["kinds_details"]["others"] = others_details
    filtered_payload["others_review"] = {
        "source_summary": relative_to_video_dir(review_summary_path(video_path), video_path),
        "output_file": relative_to_video_dir(output_path(video_path), video_path),
        "reviewed_count": len(review_by_entry_id),
        "removed_count": len(removed),
        "corrected_count": len(corrected),
        "kept_count": len(others),
        "removed_entry_ids": removed,
        "corrected_entry_ids": corrected,
    }
    return filtered_payload


def apply_review(video_path, force=False):
    source_filtered = filtered_path(video_path)
    source_summary = review_summary_path(video_path)
    target = output_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source_filtered.exists():
        print(f"[skip] filtered introuvable: {source_filtered}")
        return None
    if not source_summary.exists():
        print(f"[skip] summary introuvable: {source_summary}")
        return None

    filtered_payload = load_json(source_filtered)
    summary_payload = load_json(source_summary)
    output_payload = apply_review_to_filtered(filtered_payload, summary_payload, video_path)
    target.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {target}")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produit reviewed_ocr_overlays.json en appliquant les decisions GPT sur les items others."
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
        help="Regenere le fichier reviewed meme s'il existe deja.",
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
        if apply_review(video_path, force=args.force):
            done += 1

    print(f"{done} fichier(s) OCR reviewed generes.")


if __name__ == "__main__":
    main()
