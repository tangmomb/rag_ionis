import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from analysed_infos import update_analysed_infos

DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DEFAULT_MODEL_PATH = Path("models/frame_filter_2026-07-02_21-30-31.joblib")
DEFAULT_EMBEDDING_CACHE_DIRNAME = ".embedding_cache"
DEFAULT_BATCH_SIZE = 16
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
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


def ensure_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def choose_device(requested=None):
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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


def load_image_rgb(path, crop_bottom=0.0):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if crop_bottom > 0:
        width, height = image.size
        keep_height = max(1, int(round(height * (1.0 - crop_bottom))))
        image = image.crop((0, 0, width, keep_height))
    return image


def file_signature(path):
    import hashlib

    p = Path(path)
    stat = p.stat()
    payload = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_hash(parts):
    import hashlib

    text = json.dumps(list(parts), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingConfig:
    def __init__(self, dino_model, clip_model, crop_bottom):
        self.dino_model = dino_model
        self.clip_model = clip_model
        self.crop_bottom = crop_bottom

    def to_dict(self):
        return {
            "dino_model": self.dino_model,
            "clip_model": self.clip_model,
            "crop_bottom": self.crop_bottom,
        }


class FrozenBackboneEmbedder:
    def __init__(self, config, device="cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.config = config
        self.torch = torch
        self.device = torch.device(device)
        self.dino_processor = AutoImageProcessor.from_pretrained(self.config.dino_model)
        self.dino_model = AutoModel.from_pretrained(self.config.dino_model).to(self.device)
        self.dino_model.eval()
        for parameter in self.dino_model.parameters():
            parameter.requires_grad_(False)

        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)
        self.clip_model = CLIPModel.from_pretrained(self.config.clip_model).to(self.device)
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)

    def embed_images(self, images):
        import numpy as np

        torch = self.torch
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        with torch.inference_mode():
            dino_inputs = self.dino_processor(images=list(images), return_tensors="pt").to(self.device)
            dino_outputs = self.dino_model(**dino_inputs)
            dino_vec = dino_outputs.last_hidden_state[:, 0, :]
            dino_vec = torch.nn.functional.normalize(dino_vec, dim=1)

            clip_inputs = self.clip_processor(images=list(images), return_tensors="pt").to(self.device)
            clip_outputs = self.clip_model.vision_model(pixel_values=clip_inputs["pixel_values"])
            if hasattr(clip_outputs, "pooler_output") and clip_outputs.pooler_output is not None:
                clip_vec = clip_outputs.pooler_output
            else:
                clip_vec = clip_outputs.last_hidden_state[:, 0, :]
            clip_vec = self.clip_model.visual_projection(clip_vec)
            clip_vec = torch.nn.functional.normalize(clip_vec, dim=1)

            return (
                __import__("numpy").concatenate(
                    [
                        dino_vec.detach().cpu().numpy().astype("float32"),
                        clip_vec.detach().cpu().numpy().astype("float32"),
                    ],
                    axis=1,
                )
            )


def cache_path_for_image(image_path, cache_dir, config):
    signature = stable_hash(
        [
            "frame-filter-embedding-v1",
            file_signature(image_path),
            config.to_dict(),
        ]
    )
    return cache_dir / f"{signature}.joblib"


def embed_image_paths(image_paths, embedder, batch_size=16, cache_dir=None):
    import joblib
    import numpy as np

    paths = [Path(p) for p in image_paths]
    if not paths:
        return np.empty((0, 0), dtype=np.float32)

    cache_root = ensure_dir(cache_dir) if cache_dir is not None else None
    output = [None] * len(paths)
    missing_indices = []

    for idx, path in enumerate(paths):
        cache_file = cache_path_for_image(path, cache_root, embedder.config) if cache_root else None
        if cache_file and cache_file.exists():
            output[idx] = joblib.load(cache_file)
        else:
            missing_indices.append(idx)

    for start in range(0, len(missing_indices), batch_size):
        batch_indices = missing_indices[start : start + batch_size]
        images = [load_image_rgb(paths[i], crop_bottom=embedder.config.crop_bottom) for i in batch_indices]
        embeddings = embedder.embed_images(images)
        for local_idx, original_idx in enumerate(batch_indices):
            vector = embeddings[local_idx].astype(np.float32)
            output[original_idx] = vector
            if cache_root:
                joblib.dump(vector, cache_path_for_image(paths[original_idx], cache_root, embedder.config))
        print(f"[embeddings] {min(start + batch_size, len(missing_indices))}/{len(missing_indices)} images", flush=True)

    ready = [vector for vector in output if vector is not None]
    if len(ready) != len(paths):
        raise RuntimeError("Failed to compute all embeddings")
    return np.vstack(ready).astype(np.float32)


def load_model_payload(path):
    import joblib

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "classifier" not in payload:
        raise ValueError(f"Unsupported model format: {path}")
    return payload


def class_names_from_payload(payload):
    raw_class_names = payload.get("class_names")
    if isinstance(raw_class_names, list) and all(isinstance(item, str) for item in raw_class_names):
        return [str(item) for item in raw_class_names]

    raw_mapping = payload.get("label_mapping", {"footage": 0, "graphic": 1, "mixture": 2})
    if isinstance(raw_mapping, dict):
        pairs = sorted((int(value), str(key)) for key, value in raw_mapping.items())
        return [name for _, name in pairs]
    return ["footage", "graphic", "mixture"]


def align_probabilities(classifier, proba, class_names):
    import numpy as np

    classes = getattr(classifier, "classes_", None)
    if classes is None:
        return proba
    class_ids = [int(value) for value in np.asarray(classes).tolist()]
    if class_ids == list(range(len(class_names))):
        return proba

    aligned = np.zeros((proba.shape[0], len(class_names)), dtype=np.float32)
    for source_index, class_id in enumerate(class_ids):
        if 0 <= class_id < len(class_names):
            aligned[:, class_id] = proba[:, source_index]
    return aligned


def classify_paths(image_paths, args):
    model_payload = load_model_payload(args.model)
    class_names = class_names_from_payload(model_payload)
    filter_class_names = model_payload.get("filter_class_names", ["graphic", "mixture"])
    if not isinstance(filter_class_names, list):
        filter_class_names = ["graphic", "mixture"]
    filter_indices = [class_names.index(str(name)) for name in filter_class_names if str(name) in class_names]
    if not filter_indices and "graphic" in class_names:
        filter_indices = [class_names.index("graphic")]

    crop_bottom = float(model_payload.get("crop_bottom", 0.20))
    threshold = float(args.threshold if args.threshold is not None else model_payload.get("threshold_recommended", 0.5))
    backbones = model_payload.get("backbones", {})
    if not isinstance(backbones, dict):
        backbones = {}
    dino_model = str(backbones.get("dino", "facebook/dinov2-base"))
    clip_model = str(backbones.get("clip", "openai/clip-vit-base-patch32"))

    device = choose_device(args.device)
    print(f"[model] {args.model}", flush=True)
    print(f"[device] {device}", flush=True)
    config = EmbeddingConfig(dino_model=dino_model, clip_model=clip_model, crop_bottom=crop_bottom)
    embedder = FrozenBackboneEmbedder(config=config, device=device)
    embeddings = embed_image_paths(
        image_paths,
        embedder,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )
    classifier = model_payload["classifier"]
    proba = align_probabilities(classifier, classifier.predict_proba(embeddings), class_names)
    filter_score = proba[:, filter_indices].sum(axis=1) if filter_indices else proba[:, 0] * 0.0

    roles = ["graphic" if float(score) >= threshold else "answer" for score in filter_score]
    items = []
    for index, path in enumerate(image_paths):
        probs = {f"prob_{class_name}": round(float(proba[index, class_idx]), 6) for class_idx, class_name in enumerate(class_names)}
        predicted_index = int(max(range(len(class_names)), key=lambda class_idx: float(proba[index, class_idx])))
        items.append(
            {
                "source_image": path.name,
                "second": image_second(path),
                "pred_label": class_names[predicted_index],
                "role": roles[index],
                "filter_score": round(float(filter_score[index]), 6),
                **probs,
            }
        )

    return {
        "roles": roles,
        "class_names": class_names,
        "filter_class_names": [class_names[index] for index in filter_indices],
        "threshold": threshold,
        "crop_bottom": crop_bottom,
        "backbones": {"dino": dino_model, "clip": clip_model},
        "device": device,
        "items": items,
    }


def infer_video_type(prediction_payload):
    labels = {
        str(item.get("pred_label", "")).strip().lower()
        for item in prediction_payload.get("items", [])
        if str(item.get("pred_label", "")).strip()
    }
    if labels and labels.issubset({"graphic", "mixture"}):
        return "motion_design"
    return None


def write_outputs(video_path, image_paths, images_dir, force, prediction_payload, model_path):
    manifest_path = images_dir / MANIFEST_NAME
    if manifest_path.exists() and not force:
        print(f"[skip] {video_path.name}: {manifest_path} existe deja")
        return manifest_path

    roles = prediction_payload["roles"]
    role_counts = {
        "answer": roles.count("answer"),
        "graphic": roles.count("graphic"),
        "no_cluster": roles.count("no_cluster"),
    }
    graphic_sequences_by_index, graphic_sequences = graphic_sequence_map(image_paths, roles)

    staged_paths = stage_images(images_dir, image_paths)
    clear_cluster_dirs(images_dir)
    (images_dir / ANSWERS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (images_dir / GRAPHIC_DIR_NAME).mkdir(parents=True, exist_ok=True)

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
        prediction_item = dict(prediction_payload["items"][index])
        prediction_item["image"] = relative_image
        prediction_item["target"] = relative_target
        items.append(prediction_item)

    staging_dir = images_dir / STAGING_DIR_NAME
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    features_path = images_dir / FEATURES_NAME
    features_payload = {
        "method": "frame_filter_model",
        "source": "images",
        "model": str(Path(model_path).as_posix()),
        "class_names": prediction_payload["class_names"],
        "filter_class_names": prediction_payload["filter_class_names"],
        "threshold": prediction_payload["threshold"],
        "crop_bottom": prediction_payload["crop_bottom"],
        "backbones": prediction_payload["backbones"],
        "items": sorted(items, key=lambda item: item["image"]),
    }
    features_path.write_text(json.dumps(features_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "method": "frame_filter_model",
        "source": "images",
        "output_dir": "images",
        "model": str(Path(model_path).as_posix()),
        "class_names": prediction_payload["class_names"],
        "filter_class_names": prediction_payload["filter_class_names"],
        "threshold": prediction_payload["threshold"],
        "crop_bottom": prediction_payload["crop_bottom"],
        "backbones": prediction_payload["backbones"],
        "cluster_identifiable": True,
        "identification_strategy": "frame_filter_model",
        "no_graphic_reasons": [],
        "kmeans_rejection_reasons": [],
        "answer_count": role_counts["answer"],
        "graphic_count": role_counts["graphic"],
        "no_cluster_count": role_counts["no_cluster"],
        "graphic_sequences_count": len(graphic_sequences),
        "features": features_path.relative_to(video_path.parent).as_posix(),
        "clusters": [
            {
                "role": "answer",
                "cluster": 0,
                "count": role_counts["answer"],
                "images": role_images["answer"],
            },
            {
                "role": "graphic",
                "cluster": 1,
                "count": role_counts["graphic"],
                "sequences": graphic_sequences,
                "images": role_images["graphic"],
            },
        ],
        "items": sorted(items, key=lambda item: item["image"]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    video_type = infer_video_type(prediction_payload)
    update_analysed_infos(
        video_path,
        "classify_images",
        {
            "status": "done",
            "model": str(Path(model_path).as_posix()),
            "threshold": prediction_payload["threshold"],
            "answer_count": role_counts["answer"],
            "graphic_count": role_counts["graphic"],
            "manifest": "images/manifest.json",
            "features": "images/cv_features.json",
            "predicted_labels": sorted(
                {
                    str(item.get("pred_label", "")).strip().lower()
                    for item in prediction_payload["items"]
                    if str(item.get("pred_label", "")).strip()
                }
            ),
            "video_type": video_type,
        },
    )
    print(
        f"[ok] {video_path.name}: answers={role_counts['answer']}, graphic={role_counts['graphic']} -> {manifest_path}",
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

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable: {model_path}")

    print(f"[analyse] {video_path.name}: {len(paths)} images, model={model_path.name}", flush=True)
    prediction_payload = classify_paths(paths, args)
    return write_outputs(video_path, paths, images_dir, args.force, prediction_payload, model_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classe les images extraites en answers/graphic avec un modele joblib DINO+CLIP."
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
        "--model-path",
        "--model",
        dest="model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Chemin du modele joblib. Defaut: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Seuil de filter_score pour classer en graphic. Defaut: threshold_recommended du modele.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Taille de batch pour les embeddings. Defaut: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--device",
        help="Device pour les embeddings, ex: cuda ou cpu. Defaut: auto.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Dossier de cache des embeddings. Defaut: images/.embedding_cache",
    )
    parser.add_argument("--clusters", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--blur-kernel", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--feature-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--min-cluster-images", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--min-majority-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--min-silhouette", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--graphic-dominant-hue-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--graphic-max-edge-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--flat-region-min-area", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--flat-delta-e-thresh", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--flat-grad-thresh", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--flat-tile-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--min-flat-region-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--min-flat-component-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--min-flat-images", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--boundary-seconds", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--boundary-max-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--boundary-max-raw-edge-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--boundary-max-gray-entropy", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--boundary-min-sequence-images", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere les clusters meme si le manifeste existe deja.",
    )
    return parser.parse_args()


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
    print(f"Classification modele: {args.model}", flush=True)

    done = 0
    for video_path in videos:
        if args.cache_dir:
            cache_dir = Path(args.cache_dir)
        else:
            cache_dir = video_path.parent / "images" / DEFAULT_EMBEDDING_CACHE_DIRNAME
        args.cache_dir = str(cache_dir)
        if classify_video_images(video_path, args):
            done += 1
    print(f"{done} classification(s) image creee(s).")


if __name__ == "__main__":
    main()
