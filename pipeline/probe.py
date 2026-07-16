from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio_ffmpeg


def probe_video(video_path: str | Path) -> dict[str, Any]:
    video = Path(video_path)
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()

    source_size = metadata.get("source_size") or metadata.get("size") or (None, None)
    width, height = source_size
    duration = metadata.get("duration")
    fps = metadata.get("fps")
    return {
        "path": video.resolve().as_posix(),
        "filename": video.name,
        "extension": video.suffix.lower(),
        "size_bytes": video.stat().st_size,
        "duration_seconds": float(duration) if duration is not None else None,
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "fps": float(fps) if fps is not None else None,
        "video_codec": metadata.get("codec"),
        "pixel_format": metadata.get("pix_fmt"),
        "audio_codec": metadata.get("audio_codec"),
        "has_audio": bool(metadata.get("audio_codec")),
        "rotation": int(metadata.get("rotate") or 0),
        "probe": "imageio_ffmpeg",
    }
