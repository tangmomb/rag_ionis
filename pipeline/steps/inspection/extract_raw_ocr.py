from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pipeline.support.json_io import write_json
from pipeline.support.paddle_ocr import (
    LocalPaddleOCR,
    image_files,
)
from pipeline.support.paths import existing_images_dir, ocr_raw_dir as output_ocr_raw_dir


DEFAULT_MIN_CONFIDENCE = 0.9
IMAGE_GROUPS = ("footage", "graphic", "mixture")


@dataclass(frozen=True)
class RawOcrOptions:
    device: str = "gpu:0"
    lang: str = "fr"
    min_confidence: float = DEFAULT_MIN_CONFIDENCE


class RawOcrRecognizer(Protocol):
    def recognize_raw(self, image_path: Path) -> object: ...


def raw_ocr_name(group_name):
    return f"raw_ocr_{group_name}_frames.json"


def existing_raw_ocr_path(ocr_dir, group_name):
    preferred = ocr_dir / raw_ocr_name(group_name)
    legacy = ocr_dir.parent / f"ocr_{group_name}.json"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def extract_for_video_isolated(
    video_path: Path,
    *,
    device: str = "gpu:0",
    lang: str = "fr",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    force: bool = False,
) -> list[Path]:
    """Execute PaddleOCR outside the process that may already have loaded PyTorch.

    PyTorch and Paddle ship different cuDNN DLL builds on Windows. Once one of
    them is loaded, importing the other in the same process can fail with
    WinError 127. A dedicated process gives Paddle a clean DLL namespace while
    preserving the normal pipeline command and its checkpointing behavior.
    """
    command = [
        sys.executable,
        "-m",
        "pipeline.workers.raw_ocr",
        "--video-path",
        str(Path(video_path).resolve()),
        "--device",
        device,
        "--lang",
        lang,
        "--min-confidence",
        str(min_confidence),
    ]
    if force:
        command.append("--force")

    print("[isolation] OCR Paddle dans un processus dedie", flush=True)
    subprocess.run(command, check=True)

    output_directory = output_ocr_raw_dir(video_path)
    return [
        existing_raw_ocr_path(output_directory, group_name)
        for group_name in IMAGE_GROUPS
    ]


def write_group_raw_outputs(ocr_dir, group_name, raw_result):
    json_path = ocr_dir / raw_ocr_name(group_name)
    write_json(json_path, raw_result)
    print(f"[write] raw {group_name} -> {json_path}", flush=True)
    return json_path


def base_payload(
    options: RawOcrOptions,
    items: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "device": options.device,
        "lang": options.lang,
        "min_confidence": options.min_confidence,
        "items": items,
    }


def print_step_progress(label, current, total):
    total = max(1, int(total or 0))
    current = min(max(0, int(current or 0)), total)
    percent = int((current / total) * 100)
    print(f"\r[{label}] {percent:3d}% ({current}/{total})", end="", flush=True)
    if current >= total:
        print(flush=True)


def extract_for_video(
    video_path: Path,
    *,
    device: str = "gpu:0",
    lang: str = "fr",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    force: bool = False,
    ocr: RawOcrRecognizer | None = None,
) -> list[Path]:
    options = RawOcrOptions(
        device=device,
        lang=lang,
        min_confidence=min_confidence,
    )
    images_dir = existing_images_dir(video_path)
    ocr_dir = output_ocr_raw_dir(video_path)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    group_output_paths = {
        group_name: existing_raw_ocr_path(ocr_dir, group_name)
        for group_name in IMAGE_GROUPS
    }
    if all(path.exists() for path in group_output_paths.values()) and not force:
        print(f"[skip] OCR brut deja genere pour {video_path.name}")
        return list(group_output_paths.values())

    images = image_files(images_dir)
    if not images:
        print(f"[skip] {video_path.name}: aucune image", flush=True)
        empty_payload = base_payload(options, [])
        return [
            write_group_raw_outputs(ocr_dir, group_name, empty_payload)
            for group_name in IMAGE_GROUPS
        ]

    if ocr is None:
        ocr = LocalPaddleOCR(
            device=device,
            lang=lang,
            min_confidence=min_confidence,
        )

    print(f"[analyse] {video_path.name}: {len(images)} images", flush=True)
    raw_items_by_group = {group_name: [] for group_name in IMAGE_GROUPS}
    total_images = len(images)
    for index, image_path in enumerate(images, start=1):
        image_name = image_path.relative_to(images_dir).as_posix()
        raw_result = ocr.recognize_raw(image_path)
        item = {
            "image": image_name,
            "raw": raw_result,
        }
        group_name = Path(image_name).parts[0] if Path(image_name).parts else ""
        if group_name in raw_items_by_group:
            raw_items_by_group[group_name].append(item)
        print_step_progress(f"ocr {video_path.name}", index, total_images)

    paths = []
    for group_name in IMAGE_GROUPS:
        group_path = write_group_raw_outputs(
            ocr_dir,
            group_name,
            base_payload(options, raw_items_by_group[group_name]),
        )
        paths.append(group_path)
    print(
        f"[done] {video_path.name}: {sum(len(items) for items in raw_items_by_group.values())} images OCR brutes",
        flush=True,
    )
    return paths
