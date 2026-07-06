import argparse
import json
import statistics
from pathlib import Path

from analysed_infos import analysed_infos_path, update_analysed_infos
from local_paddle_ocr import (
    anchored_subtitle_match,
    box_bounds,
    box_geometry,
    configure_stdio,
    graphic_kind_for_image,
    image_size,
    image_video_dirs,
    infer_subtitle_anchors,
    latest_video_dir,
    seconds_from_image_name,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
LOCATION_NAME = "ocr_location.json"
BOXES_DIRNAME = "ocr_boxes_images"
LEGACY_BOXES_DIRNAME = "ocr_boxes"


configure_stdio()


def location_path(ocr_dir):
    new_path = ocr_dir / BOXES_DIRNAME / LOCATION_NAME
    legacy_boxes_path = ocr_dir / LEGACY_BOXES_DIRNAME / LOCATION_NAME
    legacy_path = ocr_dir / LOCATION_NAME
    if new_path.exists():
        return new_path
    if legacy_boxes_path.exists():
        return legacy_boxes_path
    if not legacy_path.exists():
        return new_path
    return legacy_path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_has_subtitles(video_path, has_subtitles):
    return update_analysed_infos(video_path, "08_detect_ocr_subtitles", {"has_subtitles": bool(has_subtitles)})


def subtitle_entries_from_boxes(payload, images_dir):
    entries = []
    sizes = {}

    for item in payload.get("items", []):
        image_name = item.get("image")
        if not image_name or graphic_kind_for_image(image_name):
            continue

        image_path = images_dir / image_name
        if image_name not in sizes:
            sizes[image_name] = image_size(image_path)
        size = sizes[image_name]
        second = seconds_from_image_name(Path(image_name).name)

        for poly in item.get("boxes", []):
            bounds = box_bounds(poly)
            if not bounds:
                continue
            geometry = box_geometry(bounds, size)
            entries.append(
                {
                    "image": image_name,
                    "geometry": geometry,
                    "second": second,
                }
            )

    return entries


def has_stable_subtitle_anchor(entries):
    anchors = infer_subtitle_anchors(entries)
    if len(anchors) < 2:
        return False

    anchor_cx = statistics.median(anchor["cx"] for anchor in anchors)
    anchor_cy = statistics.median(anchor["cy"] for anchor in anchors)
    anchor_height = statistics.median(anchor["relative_height"] for anchor in anchors)
    anchor_widths = [anchor["relative_width"] for anchor in anchors]
    x_tolerance = max(0.06, min(0.16, statistics.median(anchor_widths) * 0.25))
    y_tolerance = max(0.04, min(0.075, anchor_height * 1.6))
    matching_seconds = {
        entry["second"]
        for entry in entries
        if entry["second"] is not None
        and anchored_subtitle_match(
            entry["geometry"],
            anchor_cx,
            anchor_cy,
            x_tolerance,
            y_tolerance,
        )
    }
    return len(matching_seconds) >= 3


def detect_for_video(video_path, force=False):
    ocr_dir = video_path / "ocr"
    images_dir = video_path / "images"
    source = location_path(ocr_dir)
    target = analysed_infos_path(video_path)

    if target.exists() and not force:
        try:
            if "has_subtitles" in load_json(target):
                print(f"[skip] {video_path.name}: has_subtitles existe deja")
                return None
        except Exception:
            pass

    if not source.exists():
        print(f"[skip] OCR location introuvable: {source}")
        return None

    payload = load_json(source)
    entries = subtitle_entries_from_boxes(payload, images_dir)
    has_subtitles = has_stable_subtitle_anchor(entries)
    write_has_subtitles(video_path, has_subtitles)
    print(f"[ok] {video_path.name}: has_subtitles={str(has_subtitles).lower()}", flush=True)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecte la presence probable de sous-titres OCR depuis ocr_location.json."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier contenant images/. Defaut: dernier sous-dossier de downloads/youtube avec images/.",
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
        help="Recalcule has_subtitles meme si la valeur existe deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(image_video_dirs(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if detect_for_video(video_path, force=args.force):
            done += 1
    print(f"{done} detection(s) de sous-titres OCR ecrite(s).")


if __name__ == "__main__":
    main()
