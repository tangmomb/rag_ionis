import argparse
import json
import sys
from pathlib import Path

from analysed_infos import update_analysed_infos
from ocr_processed_filtering import (
    build_filtered_payload,
    filter_overlay_items,
    filtered_ocr_path,
    group_items_by_kind,
    load_groupable_items,
    load_overlay_items,
    processed_ocr_source_path,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")

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
    grouped_kinds = group_items_by_kind(all_items, filtered_items)
    filtered_payload = build_filtered_payload(payload, source.name, filtered_items, grouped_kinds)
    target.write_text(json.dumps(filtered_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "filter_ocr_processed",
        {
            "status": "done",
            "source": f"ocr/{source.name}",
            "filtered_file": f"ocr/{target.name}",
            "filtered_item_count": len(filtered_items),
            "kind_count": len(grouped_kinds),
        },
    )
    print(f"[ok] {target}")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtre les overlays OCR et produit un ocr_processed_filtered.json."
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
