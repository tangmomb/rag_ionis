import argparse
import json
from pathlib import Path

from analysed_infos import update_analysed_infos
from local_paddle_ocr import (
    boxes_from_raw_result,
    configure_stdio,
    image_video_dirs,
    latest_video_dir,
)


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
RAW_SUFFIX = "_ocr_brut.json"
BOXES_SUFFIX = "_ocr_boxes.json"


configure_stdio()


def raw_path(transcript_dir, video_id):
    return transcript_dir / f"{video_id}{RAW_SUFFIX}"


def boxes_path(transcript_dir, video_id):
    return transcript_dir / f"{video_id}{BOXES_SUFFIX}"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_outputs(transcript_dir, video_id, result):
    json_path = boxes_path(transcript_dir, video_id)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] boxes -> {json_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrait uniquement les emplacements OCR boxes a partir du JSON OCR brut."
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
        help="Regenere le JSON OCR boxes meme s'il existe deja.",
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
        transcript_dir = video_path / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        source = raw_path(transcript_dir, video_path.name)
        target = boxes_path(transcript_dir, video_path.name)
        if target.exists() and not args.force:
            print(f"[skip] {target.name} existe deja")
            continue
        if not source.exists():
            print(f"[skip] OCR brut introuvable: {source}")
            continue

        payload = load_json(source)
        raw_items = payload.get("items", [])
        box_items = []
        total_boxes = 0
        for index, raw_item in enumerate(raw_items, start=1):
            image_name = raw_item.get("image")
            if not image_name:
                continue
            boxes = boxes_from_raw_result(raw_item.get("raw"))
            total_boxes += len(boxes)
            box_items.append(
                {
                    "image": image_name,
                    "boxes": boxes,
                }
            )
            print(f"[boxes {index}/{len(raw_items)}] {image_name}: {len(boxes)} box(es)", flush=True)

        write_outputs(
            transcript_dir,
            video_path.name,
            {
                "source": source.name,
                "items": box_items,
            },
        )
        update_analysed_infos(
            video_path,
            "ocr_boxes",
            {
                "status": "done",
                "source": f"transcript/{source.name}",
                "boxes_file": f"transcript/{target.name}",
                "image_count": len(box_items),
                "box_count": total_boxes,
            },
        )
        print(f"[done] {video_path.name}: {total_boxes} box(es)", flush=True)
        done += 1

    print(f"{done} JSON OCR boxes generes.")


if __name__ == "__main__":
    main()
