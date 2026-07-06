import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SOURCE_DIR_NAMES = ("footage",)
OUTPUT_DIR_NAME = "is_interview"
MANIFEST_NAME = "manifest.json"
DEFAULT_MAX_INTERVIEW_SEQUENCES = 5


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


def seconds_from_image_name(name):
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        if len(parts[2]) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 1000.0
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) + int(parts[3]) / 1000.0
    return None


def image_second(path):
    parsed = seconds_from_image_name(path.name)
    if parsed is not None:
        return parsed
    return float("inf")


def candidate_image_files(images_dir, source_dirs):
    candidates = []
    for dir_name in source_dirs:
        source_dir = images_dir / dir_name
        if not source_dir.exists():
            continue
        for path in source_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                candidates.append(path)
    return sorted(candidates, key=lambda path: (image_second(path), path.name, path.as_posix()))


def ensure_clean_dir(path):
    directory = Path(path)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def dct_matrix(size):
    indices = np.arange(size, dtype=np.float64)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[0, :] = 1.0 / np.sqrt(size)
    for k in range(1, size):
        matrix[k, :] = np.sqrt(2.0 / size) * np.cos((np.pi * (2 * indices + 1) * k) / (2.0 * size))
    return matrix


PHASH_DCT = dct_matrix(32)


def grayscale_array(path, size, cache):
    key = (str(path), size)
    if key not in cache:
        image = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
        cache[key] = np.asarray(image, dtype=np.float32)
    return cache[key]


def compute_phash(path, gray_cache, phash_cache):
    cache_key = str(path)
    if cache_key in phash_cache:
        return phash_cache[cache_key]
    pixels = grayscale_array(path, 32, gray_cache)
    transformed = PHASH_DCT @ pixels @ PHASH_DCT.T
    low_freq = transformed[:8, :8]
    median = float(np.median(low_freq[1:, :]))
    phash_cache[cache_key] = low_freq > median
    return phash_cache[cache_key]


def phash_distance(hash_a, hash_b):
    return int(np.count_nonzero(hash_a != hash_b))


def compute_ssim(path_a, path_b, gray_cache):
    a = grayscale_array(path_a, 128, gray_cache) / 255.0
    b = grayscale_array(path_b, 128, gray_cache) / 255.0
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov_ab = float(((a - mu_a) * (b - mu_b)).mean())
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def classify_pair(frame_a, frame_b, args, gray_cache, phash_cache):
    hash_a = compute_phash(frame_a, gray_cache, phash_cache)
    hash_b = compute_phash(frame_b, gray_cache, phash_cache)
    distance = phash_distance(hash_a, hash_b)
    if distance <= args.phash_similar_max:
        return {
            "is_similar": True,
            "method": "phash",
            "phash_distance": distance,
            "ssim": None,
        }
    if distance >= args.phash_ambiguous_max:
        return {
            "is_similar": False,
            "method": "phash",
            "phash_distance": distance,
            "ssim": None,
        }

    ssim_score = compute_ssim(frame_a, frame_b, gray_cache)
    return {
        "is_similar": ssim_score >= args.ssim_min,
        "method": "ssim",
        "phash_distance": distance,
        "ssim": round(float(ssim_score), 6),
    }


def bridge_single_gaps(pair_results, args):
    if args.max_gap_pairs <= 0 or len(pair_results) < 3:
        return pair_results

    bridged = [dict(item) for item in pair_results]
    limit = len(bridged)
    for index, item in enumerate(bridged):
        if item["is_similar"]:
            continue

        gap_size = 1
        while index + gap_size < limit and gap_size <= args.max_gap_pairs and not bridged[index + gap_size]["is_similar"]:
            gap_size += 1
        if gap_size > args.max_gap_pairs:
            continue

        left_index = index - 1
        right_index = index + gap_size
        if left_index < 0 or right_index >= limit:
            continue
        if not bridged[left_index]["is_similar"] or not bridged[right_index]["is_similar"]:
            continue

        for gap_index in range(index, index + gap_size):
            bridged[gap_index]["is_similar"] = True
            bridged[gap_index]["bridge_gap"] = True
            bridged[gap_index]["bridge_original_is_similar"] = False
    return bridged


def build_sequences(image_paths, args):
    if len(image_paths) < 2:
        return [], []

    raw_pair_results = []
    gray_cache = {}
    phash_cache = {}

    for index in range(len(image_paths) - 1):
        frame_a = image_paths[index]
        frame_b = image_paths[index + 1]
        decision = classify_pair(frame_a, frame_b, args, gray_cache, phash_cache)
        item = {
            "left_image": frame_a.name,
            "right_image": frame_b.name,
            "left_second": image_second(frame_a),
            "right_second": image_second(frame_b),
            **decision,
        }
        raw_pair_results.append(item)

    pair_results = bridge_single_gaps(raw_pair_results, args)
    current = None
    sequences = []

    for index, item in enumerate(pair_results):
        if item["is_similar"]:
            frame_a = image_paths[index]
            frame_b = image_paths[index + 1]
            if current is None:
                current = [frame_a, frame_b]
            elif current[-1] == frame_a:
                current.append(frame_b)
            else:
                current = [frame_a, frame_b]
        elif current is not None:
            if len(current) >= args.min_run_frames:
                sequences.append(current)
            current = None

    if current is not None and len(current) >= args.min_run_frames:
        sequences.append(current)

    serialized = []
    for seq_index, sequence in enumerate(sequences, start=1):
        start_second = image_second(sequence[0])
        end_second = image_second(sequence[-1])
        serialized.append(
            {
                "sequence_id": seq_index,
                "frame_count": len(sequence),
                "start_image": sequence[0].name,
                "end_image": sequence[-1].name,
                "start_second": start_second,
                "end_second": end_second,
                "duration_seconds": round(max(0.0, end_second - start_second), 3),
                "frames": [frame.name for frame in sequence],
            }
        )
    return pair_results, serialized


def write_outputs(video_path, image_paths, pair_results, sequences, args):
    output_dir = ensure_clean_dir(video_path.parent / OUTPUT_DIR_NAME)

    selected_names = []
    seen = set()
    source_by_name = {path.name: path for path in image_paths}
    for sequence in sequences:
        frame_items = []
        for frame_name in sequence["frames"]:
            if frame_name not in seen:
                seen.add(frame_name)
                selected_names.append(frame_name)
            source_path = source_by_name[frame_name]
            frame_items.append(
                {
                    "name": frame_name,
                    "source": source_path.relative_to(video_path.parent).as_posix(),
                }
            )
        sequence["frames"] = frame_items

    manifest = {
        "method": "consecutive_similarity",
        "is_interview": len(sequences) < args.max_interview_sequences,
        "source_dirs": list(args.source_dirs),
        "frame_count": len(image_paths),
        "selected_frame_count": len(selected_names),
        "sequence_count": len(sequences),
        "thresholds": {
            "phash_similar_max": args.phash_similar_max,
            "phash_ambiguous_max": args.phash_ambiguous_max,
            "ssim_min": args.ssim_min,
            "min_run_frames": args.min_run_frames,
            "max_gap_pairs": args.max_gap_pairs,
            "max_interview_sequences": args.max_interview_sequences,
        },
        "sequences": sequences,
        "pairs": pair_results,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def detect_for_video(video_path, args):
    images_dir = video_path.parent / "images"
    output_manifest = video_path.parent / OUTPUT_DIR_NAME / MANIFEST_NAME
    if output_manifest.exists() and not args.force:
        print(f"[skip] {video_path.name}: {output_manifest} existe deja")
        return output_manifest

    image_paths = candidate_image_files(images_dir, args.source_dirs)
    if not image_paths:
        print(f"[skip] {video_path.name}: aucune image candidate dans {images_dir}")
        return None

    print(f"[analyse] {video_path.name}: {len(image_paths)} images candidates", flush=True)
    pair_results, sequences = build_sequences(image_paths, args)
    manifest_path = write_outputs(video_path, image_paths, pair_results, sequences, args)
    print(
        f"[ok] {video_path.name}: sequences={len(sequences)}, frames={sum(seq['frame_count'] for seq in sequences)} -> {manifest_path}",
        flush=True,
    )
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecte les interviews via des frames consecutives visuellement similaires."
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
        "--source-dirs",
        nargs="+",
        default=list(SOURCE_DIR_NAMES),
        help="Sous-dossiers images a analyser. Defaut: footage",
    )
    parser.add_argument(
        "--phash-similar-max",
        type=int,
        default=6,
        help="Distance pHash max pour declarer une paire similaire sans SSIM. Defaut: 6",
    )
    parser.add_argument(
        "--phash-ambiguous-max",
        type=int,
        default=14,
        help="Distance pHash a partir de laquelle une paire est consideree differente. Entre les deux, on calcule le SSIM. Defaut: 14",
    )
    parser.add_argument(
        "--ssim-min",
        type=float,
        default=0.92,
        help="Score SSIM minimal pour valider une paire ambigue. Defaut: 0.92",
    )
    parser.add_argument(
        "--min-run-frames",
        type=int,
        default=6,
        help="Nombre minimum de frames consecutives similaires pour retenir une sequence. Defaut: 6",
    )
    parser.add_argument(
        "--max-gap-pairs",
        type=int,
        default=1,
        help="Nombre max de paires non similaires isolees a ponter au milieu d'une sequence stable. Defaut: 1",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere le dossier is_interview meme si le manifeste existe deja.",
    )
    parser.add_argument(
        "--max-interview-sequences",
        type=int,
        default=DEFAULT_MAX_INTERVIEW_SEQUENCES,
        help="Nombre de sequences detectees en-dessous duquel la video est consideree comme interview. Defaut: 5",
    )
    args = parser.parse_args()
    if args.phash_similar_max < 0:
        raise ValueError("--phash-similar-max doit etre >= 0")
    if args.phash_ambiguous_max < args.phash_similar_max:
        raise ValueError("--phash-ambiguous-max doit etre >= --phash-similar-max")
    if args.min_run_frames < 2:
        raise ValueError("--min-run-frames doit etre >= 2")
    if args.max_gap_pairs < 0:
        raise ValueError("--max-gap-pairs doit etre >= 0")
    if args.max_interview_sequences < 0:
        raise ValueError("--max-interview-sequences doit etre >= 0")
    return args


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
        if detect_for_video(video_path, args):
            done += 1
    print(f"{done} detection(s) interview creee(s).")


if __name__ == "__main__":
    main()
