import argparse
import json
from pathlib import Path

from analysed_infos import analysed_infos_path, update_analysed_infos
from local_paddle_ocr import (
    box_bounds,
    box_geometry,
    collapse_answer_overlay_items,
    collapse_graphic_sequence_items,
    configure_stdio,
    deduplicate_items,
    filter_decor_items,
    image_video_dirs,
    image_size,
    latest_video_dir,
    non_subtitle_kind,
    mark_last_graphic_sequence_as_outro,
    ocr_items_from_raw_result,
    refine_subtitle_kinds,
    subtitle_text_signal,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DEFAULT_MIN_CONFIDENCE = 0.9
RAW_SUFFIX = "_ocr_brut.json"
PROCESSED_SUFFIX = "_ocr_processed.json"


configure_stdio()


def raw_path(transcript_dir, video_id):
    return transcript_dir / f"{video_id}{RAW_SUFFIX}"


def processed_path(transcript_dir, video_id):
    return transcript_dir / f"{video_id}{PROCESSED_SUFFIX}"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analysed_has_subtitles(video_path):
    path = analysed_infos_path(video_path)
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    value = payload.get("has_subtitles")
    return value if isinstance(value, bool) else None


def strip_subtitle_kind(items, images_dir):
    sizes = {}
    sanitized = []
    for item in items:
        if item.get("kind") != "subtitle":
            sanitized.append(item)
            continue

        image_name = item.get("image")
        if image_name and image_name not in sizes:
            sizes[image_name] = image_size(images_dir / image_name)
        geometry = box_geometry(box_bounds(item.get("box")), sizes.get(image_name))
        word_count, has_sentence_punctuation, _has_subtitle_signal = subtitle_text_signal(
            item.get("text", ""),
            geometry["relative_width"],
        )
        replacement = dict(item)
        replacement["kind"] = non_subtitle_kind(
            item.get("text", ""),
            geometry,
            word_count,
            has_sentence_punctuation,
        )
        sanitized.append(replacement)
    return sanitized


def write_outputs(transcript_dir, video_id, result):
    json_path = processed_path(transcript_dir, video_id)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {len(result.get('items', []))} items -> {json_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transforme l'OCR brut en OCR processed sans relancer PaddleOCR."
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
        "--min-confidence",
        type=float,
        help="rec_score minimum pour garder une detection OCR. Defaut: valeur du JSON brut ou 0.9.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le JSON OCR processed meme s'il existe deja.",
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
        images_dir = video_path / "images"
        transcript_dir = video_path / "transcript"
        has_subtitles = analysed_has_subtitles(video_path)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        source = raw_path(transcript_dir, video_path.name)
        target = processed_path(transcript_dir, video_path.name)
        if target.exists() and not args.force:
            print(f"[skip] {target.name} existe deja")
            continue
        if not source.exists():
            print(f"[skip] OCR brut introuvable: {source}")
            continue

        payload = load_json(source)
        min_confidence = args.min_confidence
        if min_confidence is None:
            min_confidence = float(payload.get("min_confidence", DEFAULT_MIN_CONFIDENCE))

        items = []
        raw_items = payload.get("items", [])
        for index, raw_item in enumerate(raw_items, start=1):
            image_name = raw_item.get("image")
            if not image_name:
                continue
            image_path = images_dir / image_name
            image_items = ocr_items_from_raw_result(
                raw_item.get("raw"),
                image_name=image_name,
                image_path=image_path,
                min_confidence=min_confidence,
            )
            items.extend(image_items)
            print(f"[process {index}/{len(raw_items)}] {image_name}: {len(image_items)} texte(s)", flush=True)

        refined_items = refine_subtitle_kinds(items, images_dir)
        if has_subtitles is False:
            refined_items = strip_subtitle_kind(refined_items, images_dir)
        filtered_items = filter_decor_items(refined_items, images_dir)
        graphic_collapsed_items = collapse_graphic_sequence_items(filtered_items)
        outro_marked_items = mark_last_graphic_sequence_as_outro(graphic_collapsed_items, images_dir)
        answer_collapsed_items = collapse_answer_overlay_items(outro_marked_items, images_dir)
        processed_items = deduplicate_items(answer_collapsed_items, images_dir)
        write_outputs(
            transcript_dir,
            video_path.name,
            {
                "source": source.name,
                "min_confidence": min_confidence,
                "items": processed_items,
            },
        )
        update_analysed_infos(
            video_path,
            "ocr_processed",
            {
                "status": "done",
                "source": f"transcript/{source.name}",
                "processed_file": f"transcript/{target.name}",
                "item_count": len(processed_items),
                "min_confidence": min_confidence,
            },
        )
        print(f"[done] {video_path.name}: {len(items)} items intermediaires", flush=True)
        done += 1

    print(f"{done} JSON OCR processed generes.")


if __name__ == "__main__":
    main()
