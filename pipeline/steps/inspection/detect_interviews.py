import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pipeline.steps.inspection.classify_frames import (
    load_cached_dino_embeddings,
)
from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import existing_images_dir, existing_interview_dir, interview_dir, relative_to_video_dir


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SOURCE_DIR_NAMES = ("footage",)
OUTPUT_DIR_NAME = "interview"
MANIFEST_NAME = "interview_detection_manifest.json"
LEGACY_MANIFEST_NAME = "manifest.json"
CLASSIFICATION_MANIFEST_NAME = "frame_classification_manifest.json"
DEFAULT_MAX_INTERVIEW_SEQUENCES = 5


@dataclass(frozen=True)
class InterviewDetectionOptions:
    source_dirs: tuple[str, ...] = SOURCE_DIR_NAMES
    phash_similar_max: int = 6
    phash_ambiguous_max: int = 14
    ssim_min: float = 0.92
    ssim_high_phash_min: float = 0.97
    analysis_crop_bottom: float = 0.18
    analysis_blur_radius: float = 1.0
    cluster_similarity_min: float = 0.88
    dominant_cluster_ratio_min: float = 0.70
    min_run_frames: int = 6
    max_gap_pairs: int = 1
    max_interview_sequences: int = DEFAULT_MAX_INTERVIEW_SEQUENCES
    force: bool = False


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


def classification_manifest_path(video_path):
    images_dir = existing_images_dir(video_path)
    preferred = images_dir / CLASSIFICATION_MANIFEST_NAME
    legacy = images_dir / LEGACY_MANIFEST_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_json(path):
    return read_json(path)


def load_classification_counts(video_path):
    source = classification_manifest_path(video_path)
    if not source.exists():
        return None

    payload = load_json(source)
    class_counts = payload.get("class_counts")
    if isinstance(class_counts, dict):
        footage_count = int(class_counts.get("footage", 0) or 0)
        graphic_count = int(class_counts.get("graphic", 0) or 0)
    else:
        footage_count = int(payload.get("footage_count", 0) or 0)
        graphic_count = int(payload.get("graphic_count", 0) or 0)

    return {
        "source": relative_to_video_dir(source, video_path),
        "footage_count": footage_count,
        "graphic_count": graphic_count,
        "graphic_exceeds_footage": graphic_count > footage_count,
    }


def dct_matrix(size):
    indices = np.arange(size, dtype=np.float64)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[0, :] = 1.0 / np.sqrt(size)
    for k in range(1, size):
        matrix[k, :] = np.sqrt(2.0 / size) * np.cos((np.pi * (2 * indices + 1) * k) / (2.0 * size))
    return matrix


PHASH_DCT = dct_matrix(32)


def grayscale_array(path, size, cache, args):
    key = (
        str(path),
        size,
        args.analysis_crop_bottom,
        args.analysis_blur_radius,
    )
    if key not in cache:
        image = Image.open(path).convert("L")
        crop_bottom = max(0.0, min(0.45, args.analysis_crop_bottom))
        if crop_bottom:
            visible_height = max(
                1,
                round(image.height * (1.0 - crop_bottom)),
            )
            image = image.crop((0, 0, image.width, visible_height))
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        if args.analysis_blur_radius > 0:
            image = image.filter(
                ImageFilter.GaussianBlur(args.analysis_blur_radius)
            )
        cache[key] = np.asarray(image, dtype=np.float32)
    return cache[key]


def compute_phash(path, args, gray_cache, phash_cache):
    cache_key = (
        str(path),
        args.analysis_crop_bottom,
        args.analysis_blur_radius,
    )
    if cache_key in phash_cache:
        return phash_cache[cache_key]
    pixels = grayscale_array(path, 32, gray_cache, args)
    transformed = PHASH_DCT @ pixels @ PHASH_DCT.T
    low_freq = transformed[:8, :8]
    median = float(np.median(low_freq[1:, :]))
    phash_cache[cache_key] = low_freq > median
    return phash_cache[cache_key]


def phash_distance(hash_a, hash_b):
    return int(np.count_nonzero(hash_a != hash_b))


def compute_ssim(path_a, path_b, args, gray_cache):
    a = grayscale_array(path_a, 128, gray_cache, args) / 255.0
    b = grayscale_array(path_b, 128, gray_cache, args) / 255.0
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
    hash_a = compute_phash(frame_a, args, gray_cache, phash_cache)
    hash_b = compute_phash(frame_b, args, gray_cache, phash_cache)
    distance = phash_distance(hash_a, hash_b)
    if distance <= args.phash_similar_max:
        return {
            "is_similar": True,
            "method": "phash",
            "phash_distance": distance,
            "ssim": None,
        }
    ssim_score = compute_ssim(frame_a, frame_b, args, gray_cache)
    high_phash = distance >= args.phash_ambiguous_max
    required_ssim = (
        args.ssim_high_phash_min
        if high_phash
        else args.ssim_min
    )
    return {
        "is_similar": ssim_score >= required_ssim,
        "method": "ssim_high_phash" if high_phash else "ssim",
        "phash_distance": distance,
        "ssim": round(float(ssim_score), 6),
        "required_ssim": required_ssim,
    }


def dominant_visual_cluster(image_paths, args, descriptors):
    if not image_paths:
        return {
            "representative": None,
            "frame_count": 0,
            "frame_ratio": 0.0,
            "frames": [],
        }

    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or len(descriptors) != len(image_paths):
        raise ValueError(
            "Un embedding DINO est requis pour chaque frame candidate."
        )
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = np.divide(
        descriptors,
        norms,
        out=np.zeros_like(descriptors),
        where=norms > 0,
    )
    similarities = descriptors @ descriptors.T
    membership = similarities >= args.cluster_similarity_min
    counts = membership.sum(axis=1)
    mean_similarities = np.divide(
        (similarities * membership).sum(axis=1),
        counts,
        out=np.zeros(len(image_paths), dtype=np.float64),
        where=counts > 0,
    )
    representative_index = max(
        range(len(image_paths)),
        key=lambda index: (
            int(counts[index]),
            float(mean_similarities[index]),
        ),
    )
    member_indexes = np.flatnonzero(
        membership[representative_index]
    ).tolist()
    member_similarities = similarities[
        representative_index,
        member_indexes,
    ]
    return {
        "representative": image_paths[representative_index],
        "frame_count": len(member_indexes),
        "frame_ratio": len(member_indexes) / len(image_paths),
        "mean_similarity": float(member_similarities.mean()),
        "min_similarity": float(member_similarities.min()),
        "frames": [image_paths[index] for index in member_indexes],
    }


def bridge_transient_gaps(
    pair_results,
    image_paths,
    args,
    gray_cache,
    phash_cache,
):
    if args.max_gap_pairs <= 0 or len(pair_results) < 3:
        return pair_results

    bridged = [dict(item) for item in pair_results]
    limit = len(bridged)
    index = 0
    while index < limit:
        if bridged[index]["is_similar"]:
            index += 1
            continue

        gap_size = 1
        while (
            index + gap_size < limit
            and not bridged[index + gap_size]["is_similar"]
        ):
            gap_size += 1
        if gap_size > args.max_gap_pairs:
            index += gap_size
            continue

        left_index = index - 1
        right_index = index + gap_size
        if left_index < 0 or right_index >= limit:
            index += gap_size
            continue
        if (
            not bridged[left_index]["is_similar"]
            or not bridged[right_index]["is_similar"]
        ):
            index += gap_size
            continue

        context_left_index = index - 1
        context_right_index = index + gap_size + 1
        context_decision = classify_pair(
            image_paths[context_left_index],
            image_paths[context_right_index],
            args,
            gray_cache,
            phash_cache,
        )
        if not context_decision["is_similar"]:
            index += gap_size
            continue

        context = {
            "left_image": image_paths[context_left_index].name,
            "right_image": image_paths[context_right_index].name,
            **context_decision,
        }
        for gap_index in range(index, index + gap_size):
            bridged[gap_index]["is_similar"] = True
            bridged[gap_index]["bridge_gap"] = True
            bridged[gap_index]["bridge_original_is_similar"] = False
            bridged[gap_index]["bridge_context"] = context
        index += gap_size
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

    pair_results = bridge_transient_gaps(
        raw_pair_results,
        image_paths,
        args,
        gray_cache,
        phash_cache,
    )
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


def sequence_count_is_interview(sequence_count, max_interview_sequences):
    return 1 <= sequence_count <= max_interview_sequences


def write_outputs(
    video_path,
    image_paths,
    dominant_cluster,
    args,
    classification_counts=None,
):
    output_dir = ensure_clean_dir(interview_dir(video_path))
    cluster_ratio = dominant_cluster["frame_ratio"]
    is_interview = cluster_ratio >= args.dominant_cluster_ratio_min
    representative = dominant_cluster["representative"]
    cluster_frames = [
        {
            "name": path.name,
            "source": relative_to_video_dir(path, video_path),
        }
        for path in dominant_cluster["frames"]
    ]
    manifest = {
        "method": "dominant_dino_embedding_cluster",
        "is_interview": is_interview,
        "source_dirs": list(args.source_dirs),
        "frame_count": len(image_paths),
        "selected_frame_count": dominant_cluster["frame_count"],
        "dominant_cluster_ratio": round(cluster_ratio, 6),
        "dominant_cluster": {
            "representative": (
                {
                    "name": representative.name,
                    "source": relative_to_video_dir(
                        representative,
                        video_path,
                    ),
                }
                if representative is not None
                else None
            ),
            "frame_count": dominant_cluster["frame_count"],
            "frame_ratio": round(cluster_ratio, 6),
            "mean_similarity": round(
                dominant_cluster.get("mean_similarity", 0.0),
                6,
            ),
            "min_similarity": round(
                dominant_cluster.get("min_similarity", 0.0),
                6,
            ),
            "frames": cluster_frames,
        },
        "thresholds": {
            "analysis_crop_bottom": args.analysis_crop_bottom,
            "analysis_blur_radius": args.analysis_blur_radius,
            "cluster_similarity_min": args.cluster_similarity_min,
            "dominant_cluster_ratio_min": (
                args.dominant_cluster_ratio_min
            ),
        },
        "classification_counts": classification_counts,
    }
    manifest_path = output_dir / MANIFEST_NAME
    write_json(manifest_path, manifest)
    return manifest_path


def detect_for_video(video_path, args):
    images_dir = existing_images_dir(video_path)
    output_dir = existing_interview_dir(video_path)
    output_manifest = output_dir / MANIFEST_NAME
    legacy_output_manifest = output_dir / LEGACY_MANIFEST_NAME
    if legacy_output_manifest.exists() and not output_manifest.exists():
        output_manifest = legacy_output_manifest
    if output_manifest.exists() and not args.force:
        print(f"[skip] {video_path.name}: {output_manifest} existe deja")
        return output_manifest

    image_paths = candidate_image_files(images_dir, args.source_dirs)
    if not image_paths:
        print(f"[skip] {video_path.name}: aucune image candidate dans {images_dir}")
        return None

    classification_counts = load_classification_counts(video_path)

    print(f"[analyse] {video_path.name}: {len(image_paths)} images candidates", flush=True)
    descriptors = load_cached_dino_embeddings(
        image_paths,
        images_dir,
    )
    cluster = dominant_visual_cluster(
        image_paths,
        args,
        descriptors,
    )
    manifest_path = write_outputs(
        video_path,
        image_paths,
        cluster,
        args,
        classification_counts,
    )
    print(
        (
            f"[ok] {video_path.name}: cluster="
            f"{cluster['frame_count']}/{len(image_paths)} "
            f"({cluster['frame_ratio']:.1%}), "
            f"is_interview="
            f"{cluster['frame_ratio'] >= args.dominant_cluster_ratio_min} "
            f"-> {manifest_path}"
        ),
        flush=True,
    )
    return manifest_path


def detect_video(
    video_path,
    *,
    source_dirs=SOURCE_DIR_NAMES,
    phash_similar_max: int = 6,
    phash_ambiguous_max: int = 14,
    ssim_min: float = 0.92,
    ssim_high_phash_min: float = 0.97,
    analysis_crop_bottom: float = 0.18,
    analysis_blur_radius: float = 1.0,
    cluster_similarity_min: float = 0.88,
    dominant_cluster_ratio_min: float = 0.70,
    min_run_frames: int = 6,
    max_gap_pairs: int = 1,
    max_interview_sequences: int = DEFAULT_MAX_INTERVIEW_SEQUENCES,
    force: bool = False,
):
    """API Python nommee pour detecter les sequences d'interview."""

    return detect_for_video(
        video_path,
        InterviewDetectionOptions(
            source_dirs=tuple(source_dirs),
            phash_similar_max=phash_similar_max,
            phash_ambiguous_max=phash_ambiguous_max,
            ssim_min=ssim_min,
            ssim_high_phash_min=ssim_high_phash_min,
            analysis_crop_bottom=analysis_crop_bottom,
            analysis_blur_radius=analysis_blur_radius,
            cluster_similarity_min=cluster_similarity_min,
            dominant_cluster_ratio_min=dominant_cluster_ratio_min,
            min_run_frames=min_run_frames,
            max_gap_pairs=max_gap_pairs,
            max_interview_sequences=max_interview_sequences,
            force=force,
        ),
    )
