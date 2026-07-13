from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_JSON_BYTES = 20 * 1024 * 1024
MAX_PREVIEW_CHARS = 1_500


def slugify(value: str, limit: int = 80) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized or "pipeline")[:limit].rstrip("-")


def command_argument(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _read_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size > MAX_JSON_BYTES:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "oui" if value else "non"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _json_item_count(payload: Any) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return None
    for key in ("items", "chunks", "frames", "results", "detections", "overlays"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    kinds = payload.get("kinds")
    if isinstance(kinds, dict):
        return sum(len(value) for value in kinds.values() if isinstance(value, (list, dict)))
    return None


def _text_samples(payload: Any, limit: int = 8) -> list[str]:
    samples: list[str] = []

    def visit(value: Any, key: str | None = None) -> None:
        if len(samples) >= limit:
            return
        if isinstance(value, dict):
            for child_key, child in value.items():
                if child_key in {"text", "content", "corrected_text"} and isinstance(child, str):
                    cleaned = " ".join(child.split())
                    if cleaned and cleaned not in samples:
                        samples.append(cleaned[:240])
                elif child_key == "kinds" or key == "kinds":
                    visit(child, child_key)
                if len(samples) >= limit:
                    break
        elif isinstance(value, list):
            for child in value[:50]:
                visit(child, key)
                if len(samples) >= limit:
                    break
        elif key == "kinds" and isinstance(value, str):
            cleaned = " ".join(value.split())
            if cleaned and cleaned not in samples:
                samples.append(cleaned[:240])

    kinds = payload.get("kinds") if isinstance(payload, dict) else None
    if isinstance(kinds, dict):
        for values in kinds.values():
            if isinstance(values, dict):
                for text in values.values():
                    if isinstance(text, str) and text not in samples:
                        samples.append(" ".join(text.split())[:240])
                        if len(samples) >= limit:
                            return samples
    visit(payload)
    return samples


def resolve_artifact_scopes(command: list[str]) -> list[Path]:
    video_dir_value = command_argument(command, "--video-dir")
    if video_dir_value:
        path = Path(video_dir_value).resolve()
        if not path.exists():
            return []
        direct_video = any(
            child.is_file() and child.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
            for child in path.iterdir()
        )
        if direct_video:
            return [path]
        child_video_dirs = sorted(
            child
            for child in path.iterdir()
            if child.is_dir()
            and any(
                candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
                for candidate in child.iterdir()
            )
        )
        return child_video_dirs or [path]

    download_dir_value = command_argument(command, "--download-dir")
    if not download_dir_value:
        return []
    download_dir = Path(download_dir_value).resolve()
    if not download_dir.exists():
        return []

    runs = sorted(path for path in download_dir.iterdir() if path.is_dir() and path.name.endswith("_init"))
    if not runs:
        return []
    latest_run = runs[-1]
    video_dirs = sorted(
        path
        for path in latest_run.iterdir()
        if path.is_dir() and any(child.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} for child in path.iterdir())
    )
    return video_dirs or [latest_run]


def progress_for_label(label: str) -> float | None:
    match = re.search(r"\bStep\s+(\d{1,2})\b", label, flags=re.IGNORECASE)
    if not match:
        return None
    # Step 23 is the last per-video processing step. Steps 24 and 25 only
    # finalize publication to S3 and SQL, so the video itself is ready at 100%.
    return min(100.0, int(match.group(1)) / 23 * 100)


def is_video_report_step(label: str) -> bool:
    match = re.search(r"\bStep\s+(\d{1,2})\b", label, flags=re.IGNORECASE)
    return bool(match and int(match.group(1)) == 23)


def build_video_markdown(video_dir: Path, label: str, command: list[str], elapsed_seconds: float) -> str:
    metadata = _read_json(video_dir / "metadata" / "youtube_video_metadata.json") or {}
    analysis_path = _first_existing(
        (
            video_dir / "metadata" / "pipeline_analysis.json",
            video_dir / "outputs" / "metadata" / "pipeline_analysis.json",
        )
    )
    analysis = _read_json(analysis_path) if analysis_path else {}
    analysis = analysis if isinstance(analysis, dict) else {}

    video_id = str(metadata.get("youtube_video_id") or video_dir.name)
    title = str(metadata.get("title") or video_id)
    duration = metadata.get("duration_seconds")
    lines = [
        f"# {title}",
        "",
        f"**Étape terminée :** {label}  ",
        f"**Vidéo :** `{video_id}`  ",
        f"**Durée de l'étape :** {elapsed_seconds:.1f} s  ",
        f"**Dossier :** `{video_dir}`",
        "",
        "## État courant",
        "",
        "| Information | Valeur |",
        "|---|---:|",
        f"| Durée vidéo | {_markdown_cell(duration)} s |",
        f"| Sous-titres détectés | {_markdown_cell(analysis.get('has_subtitles'))} |",
        f"| Type de vidéo | {_markdown_cell(analysis.get('video_type'))} |",
    ]

    images_dir = video_dir / "outputs" / "images"
    if images_dir.exists():
        image_files = [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        group_counts: Counter[str] = Counter()
        for path in image_files:
            relative = path.relative_to(images_dir)
            group = relative.parts[0] if len(relative.parts) > 1 else "non classées"
            group_counts[group] += 1
        lines.extend(["", "## Images", "", f"**Total : {len(image_files)}**"])
        if group_counts:
            lines.extend(["", "| Groupe | Images |", "|---|---:|"])
            lines.extend(f"| {_markdown_cell(group)} | {count} |" for group, count in sorted(group_counts.items()))

    classification = _read_json(images_dir / "frame_classification_manifest.json")
    if isinstance(classification, dict) and isinstance(classification.get("class_counts"), dict):
        lines.extend(["", "## Classification", "", "| Classe | Frames |", "|---|---:|"])
        lines.extend(
            f"| {_markdown_cell(name)} | {count} |"
            for name, count in classification["class_counts"].items()
        )

    ocr_dir = video_dir / "outputs" / "ocr"
    if ocr_dir.exists():
        ocr_files = sorted(ocr_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        lines.extend(["", "## OCR", "", "| Fichier | Éléments | Taille |", "|---|---:|---:|"])
        for path in ocr_files[-8:]:
            payload = _read_json(path)
            item_count = _json_item_count(payload)
            lines.append(
                f"| `{path.name}` | {_markdown_cell(item_count)} | {path.stat().st_size / 1024:.1f} Ko |"
            )
        if ocr_files:
            samples = _text_samples(_read_json(ocr_files[-1]))
            if samples:
                lines.extend(["", "### Aperçu OCR", ""])
                lines.extend(f"- {sample}" for sample in samples)

    transcript_dirs = sorted(path for path in (video_dir / "outputs").glob("transcripts*") if path.is_dir())
    transcript_files = sorted(
        path
        for directory in transcript_dirs
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    if transcript_files:
        lines.extend(["", "## Transcripts", "", "| Fichier | Taille |", "|---|---:|"])
        for path in transcript_files:
            lines.append(f"| `{path.name}` | {path.stat().st_size / 1024:.1f} Ko |")
        preferred = _first_existing(
            [
                *(directory / "video_summary.md" for directory in transcript_dirs),
                *(directory / "plain_transcript.txt" for directory in transcript_dirs),
            ]
        ) or transcript_files[-1]
        try:
            preview = preferred.read_text(encoding="utf-8")[:MAX_PREVIEW_CHARS].strip()
        except (OSError, UnicodeDecodeError):
            preview = ""
        if preview:
            lines.extend(["", f"### Aperçu — `{preferred.name}`", "", preview])

    chunks_dir = video_dir / "outputs" / "chunks"
    chunks_path = _first_existing(
        (
            chunks_dir / "transcript_chunks_speaker_validated.json",
            chunks_dir / "transcript_chunks.json",
        )
    )
    chunks_payload = _read_json(chunks_path) if chunks_path else None
    chunks = chunks_payload.get("chunks") if isinstance(chunks_payload, dict) else chunks_payload
    if isinstance(chunks, list):
        speakers: set[str] = set()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            metadata_value = chunk.get("meta_data") or {}
            if isinstance(metadata_value, dict):
                speakers.update(str(value) for value in metadata_value.get("speakers", []) if value)
        embedding_count = len(list(chunks_dir.glob("*_embedding.json")))
        lines.extend(
            [
                "",
                "## Chunks et embeddings",
                "",
                f"- Chunks : **{len(chunks)}**",
                f"- Embeddings : **{embedding_count}**",
                f"- Speakers : **{', '.join(sorted(speakers)) or 'aucun'}**",
            ]
        )
        previews = [
            " ".join(str(chunk.get("content") or chunk.get("text") or "").split())[:350]
            for chunk in chunks[:3]
            if isinstance(chunk, dict)
        ]
        if any(previews):
            lines.extend(["", "### Aperçu des chunks", ""])
            lines.extend(f"- {preview}" for preview in previews if preview)

    lines.extend(["", "## Commande", "", f"`{' '.join(map(str, command))}`"])
    return "\n".join(lines)


def artifact_records(label: str, command: list[str], elapsed_seconds: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    progress = progress_for_label(label)
    publish_report = is_video_report_step(label)
    for scope in resolve_artifact_scopes(command):
        metadata = _read_json(scope / "metadata" / "youtube_video_metadata.json") or {}
        video_id = str(metadata.get("youtube_video_id") or scope.name)
        title = str(metadata.get("title") or video_id)
        records.append(
            {
                "key": slugify(f"rapport-video-{video_id}"),
                "progress_key": slugify(f"progress-{video_id}"),
                "progress": progress,
                "progress_description": f"{title} — {label}",
                "publish_report": publish_report,
                "description": f"Rapport final — {title} ({video_id})",
                "markdown": (
                    build_video_markdown(scope, label, command, elapsed_seconds)
                    if publish_report
                    else None
                ),
            }
        )
    return records
