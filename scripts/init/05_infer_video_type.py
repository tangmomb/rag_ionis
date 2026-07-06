import argparse
import json
import sys
from pathlib import Path

from analysed_infos import update_analysed_infos

DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
MANIFEST_NAME = "manifest.json"


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


def manifest_path(video_path):
    return video_path.parent / "images" / MANIFEST_NAME


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def infer_video_type_from_manifest(payload):
    labels = {
        str(item.get("pred_label", "")).strip().lower()
        for item in payload.get("items", [])
        if str(item.get("pred_label", "")).strip()
    }
    if labels and labels.issubset({"graphic", "mixture"}):
        return "motion_design"
    return "video_recording"


def write_analysed_infos(video_path, video_type):
    return update_analysed_infos(video_path, "05_infer_video_type", {"video_type": video_type})


def infer_for_video(video_path, force=False):
    source = manifest_path(video_path)
    if not source.exists():
        print(f"[skip] manifest introuvable: {source}")
        return None

    payload = load_json(source)
    video_type = infer_video_type_from_manifest(payload)
    write_analysed_infos(video_path, video_type)
    print(f"[ok] {video_path.name}: video_type={video_type}", flush=True)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduit le type de video a partir du manifest de classification images."
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
        help="Reecrit analysed_infos meme si les donnees existent deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if infer_for_video(video_path, force=args.force):
            done += 1
    print(f"{done} type(s) de video inferes.")


if __name__ == "__main__":
    main()
