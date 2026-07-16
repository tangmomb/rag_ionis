from __future__ import annotations

from pathlib import Path

from .context import VIDEO_EXTENSIONS, video_in_directory


def discover_videos(root: str | Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dossier videos introuvable: {directory}")
    videos: list[Path] = []
    for child in sorted(directory.iterdir()):
        if child.is_dir():
            candidates = [
                path
                for path in child.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ]
            if len(candidates) == 1:
                videos.append(candidates[0])
        elif child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(child)
    return videos


def select_videos(root: str | Path, selector: str) -> list[Path]:
    candidate = Path(selector)
    if candidate.exists():
        return [video_in_directory(candidate) if candidate.is_dir() else candidate]

    videos = discover_videos(root)
    normalized = selector.strip()
    if normalized.lower() == "all":
        return videos
    if normalized.isdigit():
        count = int(normalized)
        if count <= 0:
            raise ValueError("Le nombre de videos doit etre positif.")
        return videos[:count]

    matches = [
        video
        for video in videos
        if video.stem == normalized or video.parent.name == normalized
    ]
    if not matches:
        raise FileNotFoundError(f"Video introuvable pour le selecteur: {selector}")
    return matches
