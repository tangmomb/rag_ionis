import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.support.analysis import update_analysed_infos
from imageio_ffmpeg import get_ffmpeg_exe
from pipeline.support.paths import images_dir, relative_to_video_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
BIN_DIR = Path("downloads/bin")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def ffmpeg_exe():
    source = Path(get_ffmpeg_exe())
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "ffmpeg.exe"
    if not target.exists():
        shutil.copy2(source, target)
    return target


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
        path
        for path in parent_dir.iterdir()
        if path.is_dir() and any(video_files(path))
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


def extract_images(video_path, interval_seconds, force=False):
    video_images_dir = images_dir(video_path)
    existing = sorted(video_images_dir.glob("*.jpg")) if video_images_dir.exists() else []
    if existing and not force:
        print(f"[skip] {video_path.name}: {len(existing)} images existent deja")
        return len(existing)

    if force and video_images_dir.exists():
        shutil.rmtree(video_images_dir)
    video_images_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = video_images_dir / "frame_%05d.jpg"
    command = [
        str(ffmpeg_exe()),
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_seconds}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    print(f"[images] {video_path.name} -> {video_images_dir}")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rename_frames_with_timecodes(video_images_dir, interval_seconds)
    count = len(list(video_images_dir.glob("*.jpg")))
    print(f"[ok] {count} images")
    update_analysed_infos(
        video_path,
        "extract_images",
        {
            "status": "done",
            "interval_seconds": interval_seconds,
            "image_count": count,
            "images_dir": relative_to_video_dir(video_images_dir, video_path),
        },
    )
    return count


def format_timecode(value):
    total_milliseconds = int(round(value * 1000))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    suffix = f"_{milliseconds:03d}" if milliseconds else ""
    if hours:
        return f"{hours:02d}_{minutes:02d}_{seconds:02d}{suffix}"
    return f"{minutes:02d}_{seconds:02d}{suffix}"


def rename_frames_with_timecodes(video_images_dir, interval_seconds):
    frames = sorted(video_images_dir.glob("frame_*.jpg"))
    for index, frame in enumerate(frames):
        second = index * interval_seconds
        second = math.floor(second * 1000) / 1000
        target = video_images_dir / f"{format_timecode(second)}.jpg"
        frame.rename(target)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrait une image toutes les N secondes pour chaque video."
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
        "--interval",
        type=float,
        default=0.5,
        help="Intervalle en secondes entre deux images. Defaut: 0.5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Nombre maximum de videos a traiter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Supprime et regenere les images deja extraites.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval doit etre superieur a 0")

    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit is not None:
        videos = videos[: args.limit]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")

    total = 0
    for video_path in videos:
        total += extract_images(video_path, args.interval, force=args.force)

    print(f"{total} images extraites.")


if __name__ == "__main__":
    main()
