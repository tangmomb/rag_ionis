import math
import shutil
import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe
from pipeline.support.paths import images_dir


BIN_DIR = Path("downloads/bin")


def ffmpeg_exe():
    source = Path(get_ffmpeg_exe())
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "ffmpeg.exe"
    if not target.exists():
        shutil.copy2(source, target)
    return target


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
