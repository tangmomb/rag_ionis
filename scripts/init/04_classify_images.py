import argparse
import json
import re
import shutil
import sys
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
DEFAULT_CLUSTERS = 2
DEFAULT_BLUR_KERNEL = 31
DEFAULT_FEATURE_SIZE = 64
DEFAULT_MIN_CLUSTER_IMAGES = 5
DEFAULT_MIN_MAJORITY_RATIO = 0.70
DEFAULT_MIN_SILHOUETTE = 0.12
DEFAULT_GRAPHIC_DOMINANT_HUE_RATIO = 0.70
DEFAULT_GRAPHIC_MAX_EDGE_RATIO = 0.002
DEFAULT_FLAT_REGION_MIN_AREA = 500
DEFAULT_FLAT_DELTA_E_THRESH = 3.0
DEFAULT_FLAT_GRAD_THRESH = 0.015
DEFAULT_FLAT_TILE_SIZE = 32
DEFAULT_MIN_FLAT_REGION_RATIO = 0.08
DEFAULT_MIN_FLAT_COMPONENT_RATIO = 0.03
DEFAULT_MIN_FLAT_IMAGES = 2
DEFAULT_BOUNDARY_SECONDS = 12.0
DEFAULT_BOUNDARY_MAX_RATIO = 0.12
DEFAULT_BOUNDARY_MAX_RAW_EDGE_RATIO = 0.08
DEFAULT_BOUNDARY_MAX_GRAY_ENTROPY = 0.60
DEFAULT_BOUNDARY_MIN_SEQUENCE_IMAGES = 2
ANSWERS_DIR_NAME = "answers"
GRAPHIC_DIR_NAME = "graphic"
NO_CLUSTER_DIR_NAME = "no_cluster"
MANIFEST_NAME = "manifest.json"
FEATURES_NAME = "cv_features.json"
OLD_EMBEDDINGS_NAME = "dinov2_embeddings.json"
STAGING_DIR_NAME = ".cluster_tmp"
SECOND_PATTERN = re.compile(r"^seconde_(\d+(?:_\d+)?)$")
TIMECODE_PATTERN = re.compile(r"^(?:(\d{2})_)?(\d{2})_(\d{2})$")


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


def image_files(images_dir):
    if not images_dir.exists():
        return []
    candidates = (
        path
        for path in images_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and STAGING_DIR_NAME not in path.parts
    )
    return sorted(candidates, key=lambda path: (image_second(path), path.name, path.as_posix()))


def seconds_from_image_name(name):
    stem = Path(name).stem
    timecode_match = TIMECODE_PATTERN.match(stem)
    if timecode_match:
        hours = int(timecode_match.group(1) or 0)
        minutes = int(timecode_match.group(2))
        seconds = int(timecode_match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    second_match = SECOND_PATTERN.match(stem)
    if second_match:
        return float(second_match.group(1).replace("_", "."))

    return None


def image_second(path):
    parsed = seconds_from_image_name(path.name)
    if parsed is not None:
        return parsed
    return float("inf")


def odd_kernel(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def lab_plane_residual_spread(tile_lab):
    import numpy as np

    height, width = tile_lab.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    design = np.column_stack(
        [
            xx.reshape(-1).astype(np.float32),
            yy.reshape(-1).astype(np.float32),
            np.ones(height * width, dtype=np.float32),
        ]
    )
    pixels = tile_lab.reshape(-1, 3).astype(np.float32)
    fitted = np.empty_like(pixels)
    for channel in range(3):
        coefficients, *_ = np.linalg.lstsq(design, pixels[:, channel], rcond=None)
        fitted[:, channel] = design @ coefficients
    residual_delta_e = np.linalg.norm(pixels - fitted, axis=1)
    return float(np.percentile(residual_delta_e, 95))


def detect_flat_color_regions(
    image_rgb,
    min_area=DEFAULT_FLAT_REGION_MIN_AREA,
    delta_e_thresh=DEFAULT_FLAT_DELTA_E_THRESH,
    grad_thresh=DEFAULT_FLAT_GRAD_THRESH,
    tile_size=DEFAULT_FLAT_TILE_SIZE,
    max_dim=640,
):
    import cv2
    import numpy as np

    height, width = image_rgb.shape[:2]
    scale = min(1.0, float(max_dim) / max(height, width))
    if scale < 1.0:
        work_rgb = cv2.resize(
            image_rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_rgb = image_rgb

    work = work_rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(work, cv2.COLOR_RGB2Lab)
    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y) / 4.0

    work_height, work_width = work_rgb.shape[:2]
    scaled_min_area = max(16, int(round(float(min_area) * scale * scale)))
    tile_size = max(8, int(tile_size))
    flat_mask = np.zeros((work_height, work_width), dtype=np.uint8)

    for y in range(0, work_height, tile_size):
        y2 = min(work_height, y + tile_size)
        for x in range(0, work_width, tile_size):
            x2 = min(work_width, x + tile_size)
            area = (y2 - y) * (x2 - x)
            if area < scaled_min_area:
                continue

            tile_lab = lab[y:y2, x:x2]
            pixels = tile_lab.reshape(-1, 3)
            mean_lab = pixels.mean(axis=0)
            delta_e = np.linalg.norm(pixels - mean_lab, axis=1)
            color_spread = float(np.percentile(delta_e, 95))
            smooth_spread = color_spread
            if color_spread >= delta_e_thresh:
                smooth_spread = lab_plane_residual_spread(tile_lab)

            mean_grad = float(grad[y:y2, x:x2].mean())
            if min(color_spread, smooth_spread) < delta_e_thresh and mean_grad < grad_thresh:
                flat_mask[y:y2, x:x2] = 255

    kernel_size = max(3, tile_size // 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if flat_mask.shape != (height, width):
        flat_mask = cv2.resize(flat_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return flat_mask


def flat_region_stats(flat_mask):
    import cv2
    import numpy as np

    foreground = flat_mask > 0
    total = int(flat_mask.size)
    if total == 0 or not foreground.any():
        return 0.0, 0.0

    _, _, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    return float(foreground.mean()), float(largest_area / total)


def image_cv_features(
    path,
    feature_size,
    blur_kernel,
    flat_region_min_area,
    flat_delta_e_thresh,
    flat_grad_thresh,
    flat_tile_size,
):
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible: {path}")

    kernel = odd_kernel(blur_kernel)
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0) if kernel > 1 else image
    raw_small = cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA)
    raw_gray = cv2.cvtColor(raw_small, cv2.COLOR_BGR2GRAY)
    raw_edges = cv2.Canny(raw_gray, 80, 160)
    raw_edge_ratio = float(np.mean(raw_edges > 0))
    histogram = cv2.calcHist([raw_gray], [0], None, [32], [0, 256]).ravel()
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]
    gray_entropy = float(-(probabilities * np.log2(probabilities)).sum() / 5.0)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(blurred, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
    small_float = small.astype(np.float32) / 255.0

    gray_std = float(gray.std() / 255.0)
    color_std = float(small_float.reshape(-1, 3).std(axis=0).mean())
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.mean(edges > 0))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    chromatic_mask = (saturation >= 0.20) & (value >= 0.18)
    if chromatic_mask.any():
        hue_bins = 18
        bins = np.floor(hsv[..., 0][chromatic_mask].astype(np.float32) / 180.0 * hue_bins).astype(np.int32) % hue_bins
        histogram = np.bincount(bins, minlength=hue_bins)
        smoothed = histogram + np.roll(histogram, 1) + np.roll(histogram, -1)
        dominant_hue_ratio = float(smoothed.max() / (feature_size * feature_size))
    else:
        dominant_hue_ratio = 0.0

    brightness = gray.astype(np.float32) / 255.0
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    flat_mask = detect_flat_color_regions(
        image_rgb,
        min_area=flat_region_min_area,
        delta_e_thresh=flat_delta_e_thresh,
        grad_thresh=flat_grad_thresh,
        tile_size=flat_tile_size,
    )
    flat_region_ratio, largest_flat_region_ratio = flat_region_stats(flat_mask)

    return {
        "gray_std": gray_std,
        "color_std": color_std,
        "edge_ratio": edge_ratio,
        "raw_edge_ratio": raw_edge_ratio,
        "dominant_hue_ratio": dominant_hue_ratio,
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "gray_entropy": gray_entropy,
        "flat_region_ratio": flat_region_ratio,
        "largest_flat_region_ratio": largest_flat_region_ratio,
    }


def features_for_images(paths, args):
    features = []
    for index, path in enumerate(paths, start=1):
        features.append(
            image_cv_features(
                path,
                args.feature_size,
                args.blur_kernel,
                args.flat_region_min_area,
                args.flat_delta_e_thresh,
                args.flat_grad_thresh,
                args.flat_tile_size,
            )
        )
        if index % 50 == 0 or index == len(paths):
            print(f"[features] {index}/{len(paths)} images", flush=True)
    return features


def feature_matrix(features):
    import numpy as np

    keys = (
        "gray_std",
        "color_std",
        "edge_ratio",
        "dominant_hue_ratio",
        "brightness_std",
        "flat_region_ratio",
        "largest_flat_region_ratio",
    )
    matrix = np.array([[item[key] for key in keys] for item in features], dtype=np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return (matrix - mean) / std, keys, mean, std


def kmeans(points, clusters, iterations, seed):
    import cv2
    import numpy as np

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(iterations), 1e-4)
    cv2.setRNGSeed(int(seed))
    compactness, labels, centers = cv2.kmeans(
        points.astype(np.float32),
        int(clusters),
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.flatten().astype(np.int32), centers.astype(np.float32), float(compactness)


def cluster_counts(labels):
    counts = {}
    for label in labels:
        label = int(label)
        counts[label] = counts.get(label, 0) + 1
    return counts


def choose_answer_cluster(labels):
    counts = cluster_counts(labels)
    return max(sorted(counts), key=lambda label: counts[label])


def silhouette_score(points, labels):
    import numpy as np

    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) < 2:
        return 0.0

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    scores = []
    for index, label in enumerate(labels):
        same_mask = labels == label
        same_indices = np.flatnonzero(same_mask)
        if len(same_indices) <= 1:
            scores.append(0.0)
            continue

        other_means = []
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_indices = np.flatnonzero(labels == other_label)
            if len(other_indices) > 0:
                other_means.append(float(distances[index, other_indices].mean()))
        if not other_means:
            scores.append(0.0)
            continue

        own_indices = same_indices[same_indices != index]
        own_mean = float(distances[index, own_indices].mean()) if len(own_indices) else 0.0
        other_mean = min(other_means)
        denominator = max(own_mean, other_mean)
        scores.append((other_mean - own_mean) / denominator if denominator > 0.0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def graphic_override_indices(features, args):
    return {
        index
        for index, feature in enumerate(features)
        if feature["dominant_hue_ratio"] >= args.graphic_dominant_hue_ratio
        and feature["edge_ratio"] <= args.graphic_max_edge_ratio
    }


def flat_graphic_candidate_indices(features, args):
    return {
        index
        for index, feature in enumerate(features)
        if feature["flat_region_ratio"] >= args.min_flat_region_ratio
        and feature["largest_flat_region_ratio"] >= args.min_flat_component_ratio
    }


def contiguous_index_runs(indices):
    runs = []
    start = None
    previous = None
    for index in sorted(indices):
        if start is None:
            start = previous = index
            continue
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous))
        start = previous = index
    if start is not None:
        runs.append((start, previous))
    return runs


def flat_graphic_evidence_indices(features, args):
    candidates = flat_graphic_candidate_indices(features, args)
    kept = set()
    for start, end in contiguous_index_runs(candidates):
        if end - start + 1 >= args.min_flat_images:
            kept.update(range(start, end + 1))
    return kept


def boundary_window_seconds(image_paths, args):
    seconds = [image_second(path) for path in image_paths]
    finite_seconds = [second for second in seconds if second != float("inf")]
    if len(finite_seconds) < 2:
        return args.boundary_seconds
    duration = max(finite_seconds) - min(finite_seconds)
    return min(args.boundary_seconds, max(6.0, duration * args.boundary_max_ratio))


def boundary_graphic_indices(image_paths, features, args):
    seconds = [image_second(path) for path in image_paths]
    finite_seconds = [second for second in seconds if second != float("inf")]
    if not finite_seconds:
        edge_count = max(1, int(round(len(image_paths) * args.boundary_max_ratio)))
        boundary_indices = set(range(edge_count)) | set(range(max(0, len(image_paths) - edge_count), len(image_paths)))
    else:
        first_second = min(finite_seconds)
        last_second = max(finite_seconds)
        window = boundary_window_seconds(image_paths, args)
        boundary_indices = {
            index
            for index, second in enumerate(seconds)
            if second != float("inf") and (second <= first_second + window or second >= last_second - window)
        }

    candidates = {
        index
        for index in boundary_indices
        if features[index]["raw_edge_ratio"] <= args.boundary_max_raw_edge_ratio
        and features[index]["gray_entropy"] <= args.boundary_max_gray_entropy
    }
    kept = set()
    for start, end in contiguous_index_runs(candidates):
        if end - start + 1 >= args.boundary_min_sequence_images:
            kept.update(range(start, end + 1))
    return kept


def cluster_decision(labels, points, image_paths, features, args, kmeans_run):
    counts = cluster_counts(labels)
    total = len(labels)
    answer_cluster = choose_answer_cluster(labels)
    raw_graphic_clusters = [cluster for cluster in sorted(counts) if cluster != answer_cluster]
    graphic_cluster = raw_graphic_clusters[0] if raw_graphic_clusters else answer_cluster
    largest_count = counts[answer_cluster]
    smallest_count = min(counts.values()) if counts else 0
    majority_ratio = largest_count / total if total else 1.0
    silhouette = silhouette_score(points, labels)

    flat_candidate_indices = flat_graphic_candidate_indices(features, args)
    flat_evidence_indices = flat_graphic_evidence_indices(features, args)
    reasons = []
    if not kmeans_run and total >= 2 and not flat_evidence_indices:
        reasons.append("no_flat_regions")
    else:
        if len(counts) < 2:
            reasons.append("single_cluster")
        if smallest_count < args.min_cluster_images:
            reasons.append("small_cluster")
        if majority_ratio < args.min_majority_ratio:
            reasons.append("balanced_clusters")
        if silhouette < args.min_silhouette:
            reasons.append("weak_separation")
        if not flat_evidence_indices:
            reasons.append("no_flat_regions")

    boundary_indices = boundary_graphic_indices(image_paths, features, args)
    identifiable = not reasons
    boundary_fallback = bool(boundary_indices) and reasons and all(
        reason in {"balanced_clusters", "weak_separation"} for reason in reasons
    )
    override_indices = graphic_override_indices(features, args) if identifiable else set()

    if identifiable:
        roles = [
            "graphic" if index in boundary_indices or index in override_indices or int(label) != answer_cluster else "answer"
            for index, label in enumerate(labels)
        ]
    elif boundary_fallback:
        identifiable = True
        roles = ["graphic" if index in boundary_indices else "answer" for index in range(total)]
    else:
        roles = ["no_cluster" for _ in labels]

    return {
        "answer_cluster": int(answer_cluster),
        "graphic_cluster": int(graphic_cluster),
        "kmeans_run": bool(kmeans_run),
        "cluster_identifiable": identifiable,
        "no_graphic_reasons": [] if identifiable else reasons,
        "kmeans_rejection_reasons": reasons,
        "identification_strategy": "kmeans" if not boundary_fallback else "boundary_fallback",
        "roles": roles,
        "flat_graphic_candidate_indices": sorted(flat_candidate_indices),
        "flat_graphic_evidence_indices": sorted(flat_evidence_indices),
        "boundary_graphic_indices": sorted(boundary_indices),
        "graphic_override_indices": sorted(override_indices),
        "raw_cluster_counts": {str(cluster): int(count) for cluster, count in sorted(counts.items())},
        "largest_cluster_ratio": majority_ratio,
        "smallest_cluster_count": int(smallest_count),
        "silhouette": silhouette,
        "thresholds": {
            "min_cluster_images": int(args.min_cluster_images),
            "min_majority_ratio": float(args.min_majority_ratio),
            "min_silhouette": float(args.min_silhouette),
            "graphic_dominant_hue_ratio": float(args.graphic_dominant_hue_ratio),
            "graphic_max_edge_ratio": float(args.graphic_max_edge_ratio),
            "flat_region_min_area": int(args.flat_region_min_area),
            "flat_delta_e_thresh": float(args.flat_delta_e_thresh),
            "flat_grad_thresh": float(args.flat_grad_thresh),
            "flat_tile_size": int(args.flat_tile_size),
            "min_flat_region_ratio": float(args.min_flat_region_ratio),
            "min_flat_component_ratio": float(args.min_flat_component_ratio),
            "min_flat_images": int(args.min_flat_images),
            "boundary_seconds": float(args.boundary_seconds),
            "boundary_max_ratio": float(args.boundary_max_ratio),
            "boundary_max_raw_edge_ratio": float(args.boundary_max_raw_edge_ratio),
            "boundary_max_gray_entropy": float(args.boundary_max_gray_entropy),
            "boundary_min_sequence_images": int(args.boundary_min_sequence_images),
        },
    }


def role_target(images_dir, role):
    if role == "answer":
        return images_dir / ANSWERS_DIR_NAME
    if role == "no_cluster":
        return images_dir / NO_CLUSTER_DIR_NAME
    return images_dir / GRAPHIC_DIR_NAME


def graphic_sequence_map(image_paths, roles):
    sequence_by_index = {}
    sequences = []
    current_sequence = None
    previous_second = None

    for index, (path, role) in enumerate(zip(image_paths, roles)):
        if role != "graphic":
            current_sequence = None
            previous_second = None
            continue

        current_second = image_second(path)
        starts_sequence = (
            current_sequence is None
            or current_second is None
            or previous_second is None
            or current_second <= previous_second
            or current_second - previous_second > 1.5
        )
        if starts_sequence:
            current_sequence = f"graphic_{len(sequences) + 1:02d}"
            sequences.append({"name": current_sequence, "images": []})

        relative_name = f"{current_sequence}/{path.name}"
        sequence_by_index[index] = current_sequence
        sequences[-1]["images"].append(relative_name)
        previous_second = current_second

    return sequence_by_index, sequences


def stage_images(images_dir, image_paths):
    staging_dir = images_dir / STAGING_DIR_NAME
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    staged_paths = []
    for path in image_paths:
        target = staging_dir / path.name
        if target.exists():
            target = staging_dir / f"{path.stem}_{len(staged_paths):05d}{path.suffix}"
        shutil.move(path, target)
        staged_paths.append(target)
    return staged_paths


def clear_cluster_dirs(images_dir):
    for path in (
        images_dir / "answer",
        images_dir / "answer_variants",
        images_dir / ANSWERS_DIR_NAME,
        images_dir / GRAPHIC_DIR_NAME,
        images_dir / NO_CLUSTER_DIR_NAME,
        images_dir / "question_intertitles",
    ):
        if path.exists():
            shutil.rmtree(path)
    old_embeddings_path = images_dir / OLD_EMBEDDINGS_NAME
    if old_embeddings_path.exists():
        old_embeddings_path.unlink()


def write_outputs(video_path, image_paths, labels, centers, compactness, images_dir, force, features, normalization, decision):
    manifest_path = images_dir / MANIFEST_NAME
    if manifest_path.exists() and not force:
        print(f"[skip] {video_path.name}: {manifest_path} existe deja")
        return manifest_path

    answer_cluster = decision["answer_cluster"]
    graphic_cluster = decision["graphic_cluster"]
    roles = decision["roles"]
    role_counts = {
        "answer": roles.count("answer"),
        "graphic": roles.count("graphic"),
        "no_cluster": roles.count("no_cluster"),
    }
    graphic_override_index_set = set(decision["graphic_override_indices"])
    boundary_graphic_index_set = set(decision["boundary_graphic_indices"])
    flat_candidate_index_set = set(decision["flat_graphic_candidate_indices"])
    flat_evidence_index_set = set(decision["flat_graphic_evidence_indices"])
    graphic_sequences_by_index, graphic_sequences = graphic_sequence_map(image_paths, roles)

    staged_paths = stage_images(images_dir, image_paths)
    clear_cluster_dirs(images_dir)
    if decision["cluster_identifiable"]:
        (images_dir / ANSWERS_DIR_NAME).mkdir(parents=True, exist_ok=True)
        (images_dir / GRAPHIC_DIR_NAME).mkdir(parents=True, exist_ok=True)
    else:
        (images_dir / NO_CLUSTER_DIR_NAME).mkdir(parents=True, exist_ok=True)

    items = []
    role_images = {"answer": [], "graphic": [], "no_cluster": []}
    for index, path in enumerate(staged_paths):
        role = roles[index]
        target_dir = role_target(images_dir, role)
        if role == "graphic":
            target_dir = target_dir / graphic_sequences_by_index[index]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.move(path, target)
        relative_image = target.relative_to(images_dir).as_posix()
        role_images[role].append(relative_image.split("/", 1)[1] if role == "graphic" else path.name)
        relative_target = target.relative_to(video_path.parent).as_posix()
        feature_payload = {key: round(float(value), 6) for key, value in features[index].items()}
        items.append(
            {
                "image": relative_image,
                "role": role,
                "cluster": int(labels[index]),
                "flat_graphic_candidate": index in flat_candidate_index_set,
                "flat_graphic_evidence": index in flat_evidence_index_set,
                "boundary_graphic": index in boundary_graphic_index_set,
                "graphic_override": index in graphic_override_index_set,
                **feature_payload,
                "target": relative_target,
            }
        )

    staging_dir = images_dir / STAGING_DIR_NAME
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    features_path = images_dir / FEATURES_NAME
    features_payload = {
        "method": "opencv_features_kmeans",
        "source": "images",
        "feature_count": len(image_paths),
        "feature_keys": list(normalization["keys"]),
        "normalization": {
            "mean": [round(float(value), 6) for value in normalization["mean"]],
            "std": [round(float(value), 6) for value in normalization["std"]],
        },
        "items": sorted(items, key=lambda item: item["image"]),
    }
    features_path.write_text(json.dumps(features_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "method": "opencv_features_kmeans",
        "source": "images",
        "output_dir": "images",
        "clusters_count": 2,
        "answer_cluster": int(answer_cluster),
        "graphic_cluster": int(graphic_cluster),
        "kmeans_run": decision["kmeans_run"],
        "cluster_identifiable": decision["cluster_identifiable"],
        "identification_strategy": decision["identification_strategy"],
        "no_graphic_reasons": decision["no_graphic_reasons"],
        "kmeans_rejection_reasons": decision["kmeans_rejection_reasons"],
        "answer_count": role_counts["answer"],
        "graphic_count": role_counts["graphic"],
        "no_cluster_count": role_counts["no_cluster"],
        "flat_graphic_candidate_count": len(decision["flat_graphic_candidate_indices"]),
        "flat_graphic_evidence_count": len(decision["flat_graphic_evidence_indices"]),
        "boundary_graphic_count": len(decision["boundary_graphic_indices"]),
        "graphic_sequences_count": len(graphic_sequences),
        "graphic_override_count": len(decision["graphic_override_indices"]),
        "features": features_path.relative_to(video_path.parent).as_posix(),
        "compactness": round(compactness, 6),
        "silhouette": round(decision["silhouette"], 6),
        "largest_cluster_ratio": round(decision["largest_cluster_ratio"], 6),
        "smallest_cluster_count": decision["smallest_cluster_count"],
        "thresholds": {
            "min_cluster_images": decision["thresholds"]["min_cluster_images"],
            "min_majority_ratio": decision["thresholds"]["min_majority_ratio"],
            "min_silhouette": decision["thresholds"]["min_silhouette"],
            "graphic_dominant_hue_ratio": decision["thresholds"]["graphic_dominant_hue_ratio"],
            "graphic_max_edge_ratio": decision["thresholds"]["graphic_max_edge_ratio"],
            "flat_region_min_area": decision["thresholds"]["flat_region_min_area"],
            "flat_delta_e_thresh": decision["thresholds"]["flat_delta_e_thresh"],
            "flat_grad_thresh": decision["thresholds"]["flat_grad_thresh"],
            "flat_tile_size": decision["thresholds"]["flat_tile_size"],
            "min_flat_region_ratio": decision["thresholds"]["min_flat_region_ratio"],
            "min_flat_component_ratio": decision["thresholds"]["min_flat_component_ratio"],
            "min_flat_images": decision["thresholds"]["min_flat_images"],
            "boundary_seconds": decision["thresholds"]["boundary_seconds"],
            "boundary_max_ratio": decision["thresholds"]["boundary_max_ratio"],
            "boundary_max_raw_edge_ratio": decision["thresholds"]["boundary_max_raw_edge_ratio"],
            "boundary_max_gray_entropy": decision["thresholds"]["boundary_max_gray_entropy"],
            "boundary_min_sequence_images": decision["thresholds"]["boundary_min_sequence_images"],
        },
        "flat_graphic_candidate_indices": decision["flat_graphic_candidate_indices"],
        "flat_graphic_evidence_indices": decision["flat_graphic_evidence_indices"],
        "raw_cluster_counts": decision["raw_cluster_counts"],
        "clusters": (
            [
                {
                    "role": "answer",
                    "cluster": int(answer_cluster),
                    "count": role_counts["answer"],
                    "images": role_images["answer"],
                },
                {
                    "role": "graphic",
                    "cluster": int(graphic_cluster),
                    "count": role_counts["graphic"],
                    "sequences": graphic_sequences,
                    "images": role_images["graphic"],
                },
            ]
            if decision["cluster_identifiable"]
            else [
                {
                    "role": "no_cluster",
                    "cluster": None,
                    "count": role_counts["no_cluster"],
                    "images": role_images["no_cluster"],
                }
            ]
        ),
        "items": sorted(items, key=lambda item: item["image"]),
        "kmeans_centers": [
            {"cluster": index, "center": [round(float(value), 6) for value in center]}
            for index, center in enumerate(centers.tolist())
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[ok] {video_path.name}: answers={role_counts['answer']}, "
        f"graphic={role_counts['graphic']}, no_cluster={role_counts['no_cluster']} -> {manifest_path}",
        flush=True,
    )
    if not decision["cluster_identifiable"]:
        print(
            f"[no-graphic] {video_path.name}: cluster non identifiable "
            f"({', '.join(decision['no_graphic_reasons'])})",
            flush=True,
        )
    return manifest_path


def classify_video_images(video_path, args):
    images_dir = video_path.parent / "images"
    paths = image_files(images_dir)
    if not paths:
        print(f"[skip] {video_path.name}: aucune image dans {images_dir}")
        return None

    manifest_path = images_dir / MANIFEST_NAME
    if manifest_path.exists() and not args.force:
        print(f"[skip] {video_path.name}: {manifest_path} existe deja")
        return manifest_path

    print(f"[analyse] {video_path.name}: {len(paths)} images, k=2", flush=True)
    features = features_for_images(paths, args)
    points, keys, mean, std = feature_matrix(features)
    flat_evidence_indices = flat_graphic_evidence_indices(features, args)
    if len(paths) < 2:
        import numpy as np

        labels = np.zeros(len(paths), dtype=np.int32)
        centers = points[:1]
        compactness = 0.0
        kmeans_run = False
    elif not flat_evidence_indices:
        import numpy as np

        labels = np.zeros(len(paths), dtype=np.int32)
        centers = points[:1]
        compactness = 0.0
        kmeans_run = False
        print(
            f"[preflight] {video_path.name}: aucun aplat/degrade stable detecte, k-means ignore",
            flush=True,
        )
    else:
        labels, centers, compactness = kmeans(points, args.clusters, args.iterations, args.seed)
        kmeans_run = True
    decision = cluster_decision(labels, points, paths, features, args, kmeans_run)
    normalization = {"keys": keys, "mean": mean, "std": std}
    return write_outputs(
        video_path,
        paths,
        labels,
        centers,
        compactness,
        images_dir,
        args.force,
        features,
        normalization,
        decision,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classe les images extraites en answers/graphic avec des features OpenCV simples et k-means."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier contenant images/. Defaut: dernier sous-dossier de downloads/youtube avec videos.",
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
        "--clusters",
        type=int,
        default=DEFAULT_CLUSTERS,
        help=f"Nombre de clusters k-means. Defaut: {DEFAULT_CLUSTERS}",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=DEFAULT_BLUR_KERNEL,
        help=f"Taille du flou gaussien applique avant mesure. Defaut: {DEFAULT_BLUR_KERNEL}",
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=DEFAULT_FEATURE_SIZE,
        help=f"Taille carree de l'image reduite pour les statistiques couleur. Defaut: {DEFAULT_FEATURE_SIZE}",
    )
    parser.add_argument(
        "--min-cluster-images",
        type=int,
        default=DEFAULT_MIN_CLUSTER_IMAGES,
        help=(
            "Nombre minimum d'images dans le petit cluster pour accepter un cluster graphic. "
            f"Defaut: {DEFAULT_MIN_CLUSTER_IMAGES}"
        ),
    )
    parser.add_argument(
        "--min-majority-ratio",
        type=float,
        default=DEFAULT_MIN_MAJORITY_RATIO,
        help=(
            "Part minimale du plus gros cluster pour accepter un cluster graphic. "
            f"Defaut: {DEFAULT_MIN_MAJORITY_RATIO}"
        ),
    )
    parser.add_argument(
        "--min-silhouette",
        type=float,
        default=DEFAULT_MIN_SILHOUETTE,
        help=(
            "Score silhouette minimum pour considerer que les deux clusters sont separes. "
            f"Defaut: {DEFAULT_MIN_SILHOUETTE}"
        ),
    )
    parser.add_argument(
        "--graphic-dominant-hue-ratio",
        type=float,
        default=DEFAULT_GRAPHIC_DOMINANT_HUE_RATIO,
        help=(
            "Seuil de couleur dominante pour rattacher une image au cluster graphic meme si k-means l'a mise avec answers. "
            f"Defaut: {DEFAULT_GRAPHIC_DOMINANT_HUE_RATIO}"
        ),
    )
    parser.add_argument(
        "--graphic-max-edge-ratio",
        type=float,
        default=DEFAULT_GRAPHIC_MAX_EDGE_RATIO,
        help=(
            "Densite maximum de contours apres flou pour l'override graphic par couleur dominante. "
            f"Defaut: {DEFAULT_GRAPHIC_MAX_EDGE_RATIO}"
        ),
    )
    parser.add_argument(
        "--flat-region-min-area",
        type=int,
        default=DEFAULT_FLAT_REGION_MIN_AREA,
        help=(
            "Surface minimale d'une zone locale analysee pour detecter un aplat ou degrade. "
            f"Defaut: {DEFAULT_FLAT_REGION_MIN_AREA}"
        ),
    )
    parser.add_argument(
        "--flat-delta-e-thresh",
        type=float,
        default=DEFAULT_FLAT_DELTA_E_THRESH,
        help=(
            "Dispersion couleur Lab maximum pour considerer une zone comme un aplat. "
            f"Defaut: {DEFAULT_FLAT_DELTA_E_THRESH}"
        ),
    )
    parser.add_argument(
        "--flat-grad-thresh",
        type=float,
        default=DEFAULT_FLAT_GRAD_THRESH,
        help=(
            "Gradient moyen maximum pour considerer une zone comme faiblement texturee. "
            f"Defaut: {DEFAULT_FLAT_GRAD_THRESH}"
        ),
    )
    parser.add_argument(
        "--flat-tile-size",
        type=int,
        default=DEFAULT_FLAT_TILE_SIZE,
        help=f"Taille des tuiles locales pour la detection d'aplat/degrade. Defaut: {DEFAULT_FLAT_TILE_SIZE}",
    )
    parser.add_argument(
        "--min-flat-region-ratio",
        type=float,
        default=DEFAULT_MIN_FLAT_REGION_RATIO,
        help=(
            "Part minimale de l'image couverte par des zones plates pour autoriser k-means. "
            f"Defaut: {DEFAULT_MIN_FLAT_REGION_RATIO}"
        ),
    )
    parser.add_argument(
        "--min-flat-component-ratio",
        type=float,
        default=DEFAULT_MIN_FLAT_COMPONENT_RATIO,
        help=(
            "Part minimale du plus grand composant plat pour autoriser k-means. "
            f"Defaut: {DEFAULT_MIN_FLAT_COMPONENT_RATIO}"
        ),
    )
    parser.add_argument(
        "--min-flat-images",
        type=int,
        default=DEFAULT_MIN_FLAT_IMAGES,
        help=(
            "Nombre minimum d'images consecutives avec aplat/degrade pour lancer k-means. "
            f"Defaut: {DEFAULT_MIN_FLAT_IMAGES}"
        ),
    )
    parser.add_argument(
        "--boundary-seconds",
        type=float,
        default=DEFAULT_BOUNDARY_SECONDS,
        help=(
            "Fenetre maximale au debut et a la fin pour detecter intro/outro par faible complexite. "
            f"Defaut: {DEFAULT_BOUNDARY_SECONDS}"
        ),
    )
    parser.add_argument(
        "--boundary-max-ratio",
        type=float,
        default=DEFAULT_BOUNDARY_MAX_RATIO,
        help=(
            "Part maximale de la video analysee a chaque bord pour intro/outro. "
            f"Defaut: {DEFAULT_BOUNDARY_MAX_RATIO}"
        ),
    )
    parser.add_argument(
        "--boundary-max-raw-edge-ratio",
        type=float,
        default=DEFAULT_BOUNDARY_MAX_RAW_EDGE_RATIO,
        help=(
            "Densite maximum de contours non floutes pour accepter une frame intro/outro. "
            f"Defaut: {DEFAULT_BOUNDARY_MAX_RAW_EDGE_RATIO}"
        ),
    )
    parser.add_argument(
        "--boundary-max-gray-entropy",
        type=float,
        default=DEFAULT_BOUNDARY_MAX_GRAY_ENTROPY,
        help=(
            "Entropie grayscale maximum pour accepter une frame intro/outro. "
            f"Defaut: {DEFAULT_BOUNDARY_MAX_GRAY_ENTROPY}"
        ),
    )
    parser.add_argument(
        "--boundary-min-sequence-images",
        type=int,
        default=DEFAULT_BOUNDARY_MIN_SEQUENCE_IMAGES,
        help=(
            "Nombre minimum d'images consecutives pour valider une sequence intro/outro. "
            f"Defaut: {DEFAULT_BOUNDARY_MIN_SEQUENCE_IMAGES}"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Nombre maximum d'iterations k-means. Defaut: 50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed k-means. Defaut: 0",
    )
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--image-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", help=argparse.SUPPRESS)
    parser.add_argument("--answer-min-similarity", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--answer-merge-similarity", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--intertitle-dominant-color-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--intertitle-green-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere les clusters meme si le manifeste existe deja.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clusters != 2:
        raise ValueError("--clusters doit etre egal a 2")
    if args.blur_kernel < 1:
        raise ValueError("--blur-kernel doit etre superieur ou egal a 1")
    if args.feature_size <= 0:
        raise ValueError("--feature-size doit etre superieur a 0")
    if args.min_cluster_images < 1:
        raise ValueError("--min-cluster-images doit etre superieur ou egal a 1")
    if not 0.0 <= args.min_majority_ratio <= 1.0:
        raise ValueError("--min-majority-ratio doit etre entre 0 et 1")
    if not -1.0 <= args.min_silhouette <= 1.0:
        raise ValueError("--min-silhouette doit etre entre -1 et 1")
    if not 0.0 <= args.graphic_dominant_hue_ratio <= 1.0:
        raise ValueError("--graphic-dominant-hue-ratio doit etre entre 0 et 1")
    if not 0.0 <= args.graphic_max_edge_ratio <= 1.0:
        raise ValueError("--graphic-max-edge-ratio doit etre entre 0 et 1")
    if args.flat_region_min_area < 1:
        raise ValueError("--flat-region-min-area doit etre superieur ou egal a 1")
    if args.flat_delta_e_thresh <= 0:
        raise ValueError("--flat-delta-e-thresh doit etre superieur a 0")
    if args.flat_grad_thresh < 0:
        raise ValueError("--flat-grad-thresh doit etre superieur ou egal a 0")
    if args.flat_tile_size < 8:
        raise ValueError("--flat-tile-size doit etre superieur ou egal a 8")
    if not 0.0 <= args.min_flat_region_ratio <= 1.0:
        raise ValueError("--min-flat-region-ratio doit etre entre 0 et 1")
    if not 0.0 <= args.min_flat_component_ratio <= 1.0:
        raise ValueError("--min-flat-component-ratio doit etre entre 0 et 1")
    if args.min_flat_images < 1:
        raise ValueError("--min-flat-images doit etre superieur ou egal a 1")
    if args.boundary_seconds <= 0:
        raise ValueError("--boundary-seconds doit etre superieur a 0")
    if not 0.0 <= args.boundary_max_ratio <= 1.0:
        raise ValueError("--boundary-max-ratio doit etre entre 0 et 1")
    if not 0.0 <= args.boundary_max_raw_edge_ratio <= 1.0:
        raise ValueError("--boundary-max-raw-edge-ratio doit etre entre 0 et 1")
    if not 0.0 <= args.boundary_max_gray_entropy <= 1.0:
        raise ValueError("--boundary-max-gray-entropy doit etre entre 0 et 1")
    if args.boundary_min_sequence_images < 1:
        raise ValueError("--boundary-min-sequence-images doit etre superieur ou egal a 1")

    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(
        "Classification OpenCV: gray_std + color_std + edge_ratio + flat_region, k=2",
        flush=True,
    )

    done = 0
    for video_path in videos:
        if classify_video_images(video_path, args):
            done += 1
    print(f"{done} classification(s) image creee(s).")


if __name__ == "__main__":
    main()
