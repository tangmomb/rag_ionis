import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from ocr_processed_filtering import filtered_ocr_path
from local_paddle_ocr import box_bounds, configure_stdio


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
OUTPUT_DIRNAME = "ocr_processed_filtered_others_boxes"
MANIFEST_NAME = "manifest.json"
BOX_PADDING_PX = 10
BOX_OUTLINE_COLOR = (255, 0, 0)
BOX_OUTLINE_WIDTH = 4


configure_stdio()

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


def output_dir(video_path):
    return video_path.parent / "ocr" / OUTPUT_DIRNAME


def manifest_path(video_path):
    return output_dir(video_path) / MANIFEST_NAME


def load_others_entries(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = payload.get("kinds_details", {}).get("others", {})
    entries = []
    for timecode, item in sorted(details.items()):
        if not isinstance(item, dict):
            continue
        image_name = item.get("image")
        box = item.get("box")
        if not image_name or not box:
            continue
        entries.append(
            {
                "timecode": timecode,
                "text": item.get("text", ""),
                "image": image_name,
                "box": box,
                "all_occurrences": list(item.get("all_occurrences", [])),
            }
        )
    return entries


def padded_box_bounds(poly, image_size):
    bounds = box_bounds(poly)
    if not bounds:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = bounds
    left = max(0, min(width, int(x1) - BOX_PADDING_PX))
    top = max(0, min(height, int(y1) - BOX_PADDING_PX))
    right = max(0, min(width, int(x2 + 0.999999) + BOX_PADDING_PX))
    bottom = max(0, min(height, int(y2 + 0.999999) + BOX_PADDING_PX))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def safe_stem(value):
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return cleaned.strip("_") or "item"


def annotate_image(image_path, box):
    with Image.open(image_path) as image:
        bounds = padded_box_bounds(box, image.size)
        if bounds is None:
            return None
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(bounds, outline=BOX_OUTLINE_COLOR, width=BOX_OUTLINE_WIDTH)
        return annotated


def extract_for_video(video_path, force=False):
    source = filtered_ocr_path(video_path)
    target_dir = output_dir(video_path)
    target_manifest = manifest_path(video_path)
    if target_manifest.exists() and not force:
        print(f"[skip] {target_manifest.name} existe deja")
        return target_manifest
    if not source.exists():
        print(f"[skip] OCR filtered introuvable: {source}")
        return None

    entries = load_others_entries(source)
    if force and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    images_dir = video_path.parent / "images"
    manifest_items = []
    written = 0
    for index, entry in enumerate(entries, start=1):
        group_dir_name = f"{index:03d}__{safe_stem(entry['text'])}"
        group_dir = target_dir / group_dir_name
        group_dir.mkdir(parents=True, exist_ok=True)

        occurrences = list(entry.get("all_occurrences", []))
        if not occurrences and entry.get("image") and entry.get("box"):
            occurrences = [
                {
                    "timecode": entry["timecode"],
                    "text": entry["text"],
                    "image": entry["image"],
                    "box": entry["box"],
                }
            ]

        first_occurrence = occurrences[0] if occurrences else None
        if not first_occurrence:
            continue
        occurrence_image_name = first_occurrence.get("image")
        occurrence_box = first_occurrence.get("box")
        if not occurrence_image_name or not occurrence_box:
            continue
        occurrence_image_path = images_dir / occurrence_image_name
        if not occurrence_image_path.exists():
            print(f"[skip] image occurrence introuvable: {occurrence_image_path}", flush=True)
            continue
        annotated = annotate_image(occurrence_image_path, occurrence_box)
        if annotated is None:
            continue
        occurrence_output_name = (
            f"001__{safe_stem(Path(occurrence_image_name).stem)}"
            f"__{safe_stem(first_occurrence.get('text', entry['text']))}.png"
        )
        occurrence_output_path = group_dir / occurrence_output_name
        annotated.save(occurrence_output_path)

        manifest_items.append(
            {
                "timecode": entry["timecode"],
                "text": entry["text"],
                "image": entry["image"],
                "box": entry["box"],
                "group_dir": group_dir_name,
                "crop": f"{group_dir_name}/{occurrence_output_name}",
                "selected_occurrence": {
                    "timecode": first_occurrence.get("timecode"),
                    "text": first_occurrence.get("text", entry["text"]),
                    "image": occurrence_image_name,
                    "box": occurrence_box,
                    "score": first_occurrence.get("score"),
                    "annotation": f"{group_dir_name}/{occurrence_output_name}",
                },
                "all_occurrence_count": len(occurrences),
            }
        )
        written += 1

    payload = {
        "source": f"ocr/{source.name}",
        "kind": "others",
        "output_dir": f"ocr/{OUTPUT_DIRNAME}",
        "image_count": written,
        "all_occurrence_image_count": sum(item["all_occurrence_count"] for item in manifest_items),
        "rendering": {
            "mode": "full_image_with_red_box",
            "padding_px": BOX_PADDING_PX,
            "outline_color": list(BOX_OUTLINE_COLOR),
            "outline_width": BOX_OUTLINE_WIDTH,
        },
        "items": manifest_items,
    }
    target_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {target_dir} ({written} image(s) annotee(s))", flush=True)
    return target_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exporte les images entieres des items kind=others avec une box rouge autour de la zone OCR."
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
        help="Regenere les images annotees meme si elles existent deja.",
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
        if extract_for_video(video_path, force=args.force):
            done += 1
    print(f"{done} export(s) d'images annotees genere(s).")


if __name__ == "__main__":
    main()
