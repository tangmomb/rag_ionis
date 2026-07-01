import argparse
import json
import re
import shutil
import sys
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
DEFAULT_MODEL = "dinov2_vitl14"
DEFAULT_CLUSTERS = 2
DEFAULT_BATCH_SIZE = 2
DEFAULT_IMAGE_SIZE = 518
DEFAULT_INTERTITLE_DOMINANT_COLOR_RATIO = 0.45
ANSWERS_DIR_NAME = "answers"
GRAPHIC_DIR_NAME = "graphic"
MANIFEST_NAME = "manifest.json"
EMBEDDINGS_NAME = "dinov2_embeddings.json"
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
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
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


def load_dinov2(model_name, device):
    import torch

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def image_transform(image_size):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def embed_images(model, paths, device, batch_size, image_size):
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    transform = image_transform(image_size)
    embeddings = []
    autocast_enabled = device.startswith("cuda")

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = []
        for path in batch_paths:
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
        tensor = torch.stack(batch).to(device, non_blocking=True)
        autocast_device = "cuda" if device.startswith("cuda") else "cpu"
        with torch.inference_mode(), torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
            output = model(tensor)
            if isinstance(output, dict):
                output = output.get("x_norm_clstoken")
                if output is None:
                    output = output.get("x_prenorm")
                if output is None:
                    output = next(iter(output.values()))
            output = functional.normalize(output.float(), dim=1)
        embeddings.append(output.cpu())
        print(f"[embed] {start + len(batch_paths)}/{len(paths)} images", flush=True)

    return torch.cat(embeddings, dim=0)


def image_visual_features(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((224, 126)), dtype=np.float32) / 255.0

    red = array[..., 0]
    green = array[..., 1]
    blue = array[..., 2]
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    chroma = maximum - minimum
    saturation = np.divide(chroma, maximum, out=np.zeros_like(chroma), where=maximum > 0)
    hue = np.zeros_like(maximum)

    red_is_max = maximum == red
    green_is_max = maximum == green
    blue_is_max = maximum == blue
    chromatic = chroma > 0
    hue[red_is_max & chromatic] = ((green - blue)[red_is_max & chromatic] / chroma[red_is_max & chromatic]) % 6
    hue[green_is_max & chromatic] = ((blue - red)[green_is_max & chromatic] / chroma[green_is_max & chromatic]) + 2
    hue[blue_is_max & chromatic] = ((red - green)[blue_is_max & chromatic] / chroma[blue_is_max & chromatic]) + 4
    hue = hue / 6.0

    saturated_mask = (saturation >= 0.28) & (maximum >= 0.20)
    hue_bins = 24
    if saturated_mask.any():
        bins = np.floor(hue[saturated_mask] * hue_bins).astype(np.int32) % hue_bins
        histogram = np.bincount(bins, minlength=hue_bins)
        smoothed = histogram + np.roll(histogram, 1) + np.roll(histogram, -1)
        dominant_color_ratio = float(smoothed.max() / hue.size)
        dominant_chromatic_ratio = float(histogram.max() / saturated_mask.sum())
        dominant_hue = float(histogram.argmax() / hue_bins)
    else:
        dominant_color_ratio = 0.0
        dominant_chromatic_ratio = 0.0
        dominant_hue = 0.0

    brightness = array.mean(axis=2)
    return {
        "dominant_color_ratio": dominant_color_ratio,
        "dominant_chromatic_ratio": dominant_chromatic_ratio,
        "dominant_hue": dominant_hue,
        "saturation_mean": float(saturation.mean()),
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
    }


def visual_features_for_images(paths):
    return [image_visual_features(path) for path in paths]


def kmeans_plusplus(points, cluster_count, seed):
    import torch

    generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    first = torch.randint(points.size(0), (1,), generator=generator, device=points.device)
    centroids = [points[first.item()]]
    closest_distances = torch.cdist(points, centroids[0].unsqueeze(0)).squeeze(1).pow(2)

    for _ in range(1, cluster_count):
        total = closest_distances.sum()
        if float(total) == 0.0:
            candidate = torch.randint(points.size(0), (1,), generator=generator, device=points.device).item()
        else:
            candidate = torch.multinomial(closest_distances / total, 1, generator=generator).item()
        centroids.append(points[candidate])
        distances = torch.cdist(points, centroids[-1].unsqueeze(0)).squeeze(1).pow(2)
        closest_distances = torch.minimum(closest_distances, distances)

    return torch.stack(centroids)


def kmeans(points, cluster_count, iterations, seed, device):
    import torch

    points = points.to(device)
    centroids = kmeans_plusplus(points, cluster_count, seed)
    labels = torch.full((points.size(0),), -1, device=device, dtype=torch.long)

    for _ in range(iterations):
        distances = torch.cdist(points, centroids)
        next_labels = distances.argmin(dim=1)
        if torch.equal(labels, next_labels):
            break
        labels = next_labels

        updated = []
        for cluster in range(cluster_count):
            mask = labels == cluster
            if mask.any():
                updated.append(points[mask].mean(dim=0))
            else:
                updated.append(centroids[cluster])
        centroids = torch.stack(updated)

    return labels.cpu(), centroids.cpu()


def largest_cluster(labels, eligible_indices=None):
    counts = {}
    if eligible_indices is None:
        values = labels.tolist()
    else:
        values = labels[eligible_indices].tolist()
    for label in values:
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return largest_cluster(labels)
    return max(sorted(counts), key=lambda cluster: counts[cluster])


def recompute_centroids(points, labels):
    import torch
    import torch.nn.functional as functional

    centroids = {}
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        centroid = points[mask].mean(dim=0)
        centroids[cluster] = functional.normalize(centroid.unsqueeze(0), dim=1).squeeze(0)
    return centroids


def refine_labels(embeddings, labels, visual_features, intertitle_dominant_color_ratio):
    import torch

    dominant_color_indices = [
        index
        for index, features in enumerate(visual_features)
        if features["dominant_color_ratio"] >= intertitle_dominant_color_ratio
    ]
    dominant_color_index_set = set(dominant_color_indices)
    answer_eligible_indices = [
        index
        for index in range(len(labels))
        if index not in dominant_color_index_set
    ]
    answer_cluster = largest_cluster(labels, answer_eligible_indices)
    candidate_indices = torch.tensor(
        [
            index
            for index in answer_eligible_indices
            if int(labels[index].item()) == answer_cluster
        ],
        dtype=torch.long,
    )
    if len(candidate_indices) == 0:
        candidate_indices = torch.nonzero(labels == answer_cluster, as_tuple=False).flatten()
    candidate_vectors = embeddings[candidate_indices]
    pairwise_similarity = candidate_vectors @ candidate_vectors.T
    medoid_local_index = pairwise_similarity.mean(dim=1).argmax().item()
    medoid_index = candidate_indices[medoid_local_index].item()
    medoid = embeddings[medoid_index]
    answer_similarities = embeddings @ medoid

    roles = [
        "graphic" if index in dominant_color_index_set else "answer"
        for index in range(len(labels))
    ]

    return {
        "answer_cluster": int(answer_cluster),
        "answer_medoid_index": medoid_index,
        "answer_similarities": answer_similarities,
        "roles": roles,
        "dominant_color_intertitle_count": len(dominant_color_indices),
        "answer_core_count": len(answer_eligible_indices),
    }


def role_target(images_dir, role):
    if role == "answer":
        return images_dir / ANSWERS_DIR_NAME
    return images_dir / GRAPHIC_DIR_NAME


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
        images_dir / "question_intertitles",
    ):
        if path.exists():
            shutil.rmtree(path)


def write_outputs(video_path, image_paths, embeddings, labels, model_name, images_dir, force, refinement, visual_features):
    manifest_path = images_dir / MANIFEST_NAME
    if manifest_path.exists() and not force:
        print(f"[skip] {video_path.name}: {manifest_path} existe deja")
        return manifest_path

    role_counts = {
        "answer": refinement["roles"].count("answer"),
        "graphic": refinement["roles"].count("graphic"),
    }
    centroids = recompute_centroids(embeddings, labels)
    staged_paths = stage_images(images_dir, image_paths)
    clear_cluster_dirs(images_dir)

    items = []
    role_images = {"answer": [], "graphic": []}
    for index, path in enumerate(staged_paths):
        role = refinement["roles"][index]
        target_dir = role_target(images_dir, role)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.move(path, target)
        relative_target = target.relative_to(video_path.parent).as_posix()
        role_images[role].append(path.name)
        items.append(
            {
                "image": path.name,
                "role": role,
                "initial_cluster": int(refinement["initial_labels"][index].item()),
                "answer_similarity": round(float(refinement["answer_similarities"][index].item()), 6),
                "dominant_color_ratio": round(visual_features[index]["dominant_color_ratio"], 6),
                "dominant_chromatic_ratio": round(visual_features[index]["dominant_chromatic_ratio"], 6),
                "dominant_hue": round(visual_features[index]["dominant_hue"], 6),
                "saturation_mean": round(visual_features[index]["saturation_mean"], 6),
                "brightness_mean": round(visual_features[index]["brightness_mean"], 6),
                "brightness_std": round(visual_features[index]["brightness_std"], 6),
                "target": relative_target,
            }
        )
    clusters = [
        {"role": "answer", "count": role_counts["answer"], "images": role_images["answer"]},
        {"role": "graphic", "count": role_counts["graphic"], "images": role_images["graphic"]},
    ]

    staging_dir = images_dir / STAGING_DIR_NAME
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    embeddings_path = images_dir / EMBEDDINGS_NAME
    embeddings_payload = {
        "model": f"facebookresearch/dinov2:{model_name}",
        "source": "images",
        "embedding_count": len(image_paths),
        "embedding_dim": int(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0,
        "items": [
            {
                "image": path.name,
                "embedding": [round(float(value), 6) for value in vector],
            }
            for path, vector in zip(image_paths, embeddings.tolist())
        ],
    }
    embeddings_path.write_text(json.dumps(embeddings_payload, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "model": f"facebookresearch/dinov2:{model_name}",
        "source": "images",
        "output_dir": "images",
        "clusters_count": 2,
        "answer_cluster": "answer",
        "graphic_cluster": "graphic",
        "answer_candidate_cluster": refinement["answer_candidate_cluster"],
        "intertitle_dominant_color_ratio": refinement["intertitle_dominant_color_ratio"],
        "graphic_count": role_counts["graphic"],
        "answer_medoid_image": image_paths[refinement["answer_medoid_index"]].name,
        "answer_count": role_counts["answer"],
        "embeddings": embeddings_path.relative_to(video_path.parent).as_posix(),
        "clusters": clusters,
        "items": sorted(items, key=lambda item: item["image"]),
        "kmeans_centroids": [
            {
                "cluster": cluster,
                "embedding": [round(float(value), 6) for value in vector],
            }
            for cluster, vector in centroids.items()
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[ok] {video_path.name}: answers={role_counts['answer']}, "
        f"graphic={role_counts['graphic']} -> {manifest_path}",
        flush=True,
    )
    return manifest_path


def classify_video_images(model, video_path, args, device):
    images_dir = video_path.parent / "images"
    paths = image_files(images_dir)
    if not paths:
        print(f"[skip] {video_path.name}: aucune image dans {images_dir}")
        return None

    manifest_path = images_dir / MANIFEST_NAME
    if manifest_path.exists() and not args.force:
        print(f"[skip] {video_path.name}: {manifest_path} existe deja")
        return manifest_path

    cluster_count = min(args.clusters, len(paths))
    if cluster_count < 2:
        print(f"[skip] {video_path.name}: au moins 2 images sont necessaires pour k-means")
        return None

    print(f"[analyse] {video_path.name}: {len(paths)} images, k={cluster_count}", flush=True)
    visual_features = visual_features_for_images(paths)
    embeddings = embed_images(model, paths, device, args.batch_size, args.image_size)
    labels, centroids = kmeans(embeddings, cluster_count, args.iterations, args.seed, device)
    refinement = refine_labels(
        embeddings,
        labels,
        visual_features,
        args.intertitle_dominant_color_ratio,
    )
    refinement["initial_labels"] = labels
    refinement["answer_candidate_cluster"] = refinement["answer_cluster"]
    refinement["intertitle_dominant_color_ratio"] = args.intertitle_dominant_color_ratio
    return write_outputs(
        video_path,
        paths,
        embeddings,
        labels,
        args.model,
        images_dir,
        args.force,
        refinement,
        visual_features,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classe les images extraites avec DINOv2 ViT-L/14 et k-means local."
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
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modele DINOv2 torch.hub. Defaut: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_CLUSTERS,
        help=f"Nombre de clusters k-means. Defaut: {DEFAULT_CLUSTERS}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Taille de batch DINOv2. Defaut: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Taille carree envoyee a DINOv2. Defaut: {DEFAULT_IMAGE_SIZE}",
    )
    parser.add_argument("--answer-min-similarity", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--answer-merge-similarity", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--intertitle-dominant-color-ratio",
        type=float,
        default=DEFAULT_INTERTITLE_DOMINANT_COLOR_RATIO,
        help=(
            "Part minimale de pixels domines par une meme famille de couleur pour classer une image en graphic. "
            f"Defaut: {DEFAULT_INTERTITLE_DOMINANT_COLOR_RATIO}"
        ),
    )
    parser.add_argument(
        "--intertitle-green-ratio",
        dest="intertitle_dominant_color_ratio",
        type=float,
        help=argparse.SUPPRESS,
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
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device torch, par exemple cuda ou cpu. Defaut: cuda",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere les clusters meme si le manifeste existe deja.",
    )
    return parser.parse_args()


def main():
    import torch

    args = parse_args()
    if args.clusters < 2:
        raise ValueError("--clusters doit etre superieur ou egal a 2")
    if args.batch_size <= 0:
        raise ValueError("--batch-size doit etre superieur a 0")
    if not 0.0 <= args.intertitle_dominant_color_ratio <= 1.0:
        raise ValueError("--intertitle-dominant-color-ratio doit etre entre 0 et 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible. Utilise --device cpu ou installe PyTorch CUDA.")

    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    print(f"DINOv2: {args.model} sur {args.device}", flush=True)
    model = load_dinov2(args.model, args.device)

    done = 0
    for video_path in videos:
        if classify_video_images(model, video_path, args, args.device):
            done += 1
    print(f"{done} classification(s) image creee(s).")


if __name__ == "__main__":
    main()
