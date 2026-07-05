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
FOOTAGE_DIR_NAME = "footage"
GRAPHIC_DIR_NAME = "graphic"
MIXTURE_DIR_NAME = "mixture"
MANIFEST_NAME = "manifest.json"
FEATURES_NAME = "cv_features.json"
OLD_EMBEDDINGS_NAME = "dinov2_embeddings.json"
STAGING_DIR_NAME = ".classify_tmp"
LEGACY_STAGING_DIR_NAME = ".cluster_tmp"
LEGACY_OUTPUT_DIR_NAMES = ("answer", "answers", "answer_variants", "no_cluster", "question_intertitles")
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
        and LEGACY_STAGING_DIR_NAME not in path.parts
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
    if role == "footage":
        return images_dir / FOOTAGE_DIR_NAME
    if role == "mixture":
        return images_dir / MIXTURE_DIR_NAME
    if role == "graphic":
        return images_dir / GRAPHIC_DIR_NAME
    raise ValueError(f"Classe modele non supportee: {role}")


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


def clear_classification_dirs(images_dir, class_names):
    dir_names = set(LEGACY_OUTPUT_DIR_NAMES)
    dir_names.update(str(class_name) for class_name in class_names)
    dir_names.update((FOOTAGE_DIR_NAME, GRAPHIC_DIR_NAME, MIXTURE_DIR_NAME))
    for path in (images_dir / name for name in sorted(dir_names)):
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


def make_classifier_compatible(classifier):
    estimator = classifier.steps[-1][1] if hasattr(classifier, "steps") else classifier
    if estimator.__class__.__name__ == "LogisticRegression" and not hasattr(estimator, "multi_class"):
        estimator.multi_class = "auto"
    return classifier


def classify_paths(image_paths, args):
    model_payload = load_model_payload(args.model)
    class_names = class_names_from_payload(model_payload)
    crop_bottom = float(model_payload.get("crop_bottom", 0.20))
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
    classifier = make_classifier_compatible(model_payload["classifier"])
    proba = align_probabilities(classifier, classifier.predict_proba(embeddings), class_names)

    roles = []
    items = []
    for index, path in enumerate(image_paths):
        probs = {f"prob_{class_name}": round(float(proba[index, class_idx]), 6) for class_idx, class_name in enumerate(class_names)}
        predicted_index = int(max(range(len(class_names)), key=lambda class_idx: float(proba[index, class_idx])))
        predicted_label = class_names[predicted_index]
        roles.append(predicted_label)
        items.append(
            {
                "source_image": path.name,
                "second": image_second(path),
                "pred_label": predicted_label,
                "role": predicted_label,
                **probs,
            }
        )

    return {
        "roles": roles,
        "class_names": class_names,
        "crop_bottom": crop_bottom,
        "backbones": {"dino": dino_model, "clip": clip_model},
        "device": device,
        "items": items,
    }


def write_outputs(video_path, image_paths, images_dir, force, prediction_payload, model_path):
    manifest_path = images_dir / MANIFEST_NAME
    if manifest_path.exists() and not force:
        print(f"[skip] {video_path.name}: {manifest_path} existe deja")
        return manifest_path

    roles = prediction_payload["roles"]
    class_names = prediction_payload["class_names"]
    role_counts = {class_name: roles.count(class_name) for class_name in class_names}

    staged_paths = stage_images(images_dir, image_paths)
    clear_classification_dirs(images_dir, class_names)
    for class_name in class_names:
        role_target(images_dir, class_name).mkdir(parents=True, exist_ok=True)

    items = []
    role_images = {class_name: [] for class_name in class_names}
    for index, path in enumerate(staged_paths):
        role = roles[index]
        target_dir = role_target(images_dir, role)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.move(path, target)
        relative_image = target.relative_to(images_dir).as_posix()
        role_images[role].append(path.name)
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
        "class_names": class_names,
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
        "class_names": class_names,
        "crop_bottom": prediction_payload["crop_bottom"],
        "backbones": prediction_payload["backbones"],
        "identification_strategy": "frame_filter_model",
        "class_counts": role_counts,
        "footage_count": role_counts.get("footage", 0),
        "graphic_count": role_counts.get("graphic", 0),
        "mixture_count": role_counts.get("mixture", 0),
        "features": features_path.relative_to(video_path.parent).as_posix(),
        "classes": [
            {"label": class_name, "count": role_counts[class_name], "images": role_images[class_name]}
            for class_name in class_names
        ],
        "items": sorted(items, key=lambda item: item["image"]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "classify_images",
        {
            "status": "done",
            "model": str(Path(model_path).as_posix()),
            "footage_count": role_counts.get("footage", 0),
            "graphic_count": role_counts.get("graphic", 0),
            "mixture_count": role_counts.get("mixture", 0),
            "manifest": "images/manifest.json",
            "features": "images/cv_features.json",
        },
    )
    print(
        f"[ok] {video_path.name}: footage={role_counts.get('footage', 0)}, graphic={role_counts.get('graphic', 0)}, mixture={role_counts.get('mixture', 0)} -> {manifest_path}",
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
        description="Classe les images extraites en footage/graphic/mixture avec un modele joblib DINO+CLIP."
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere la classification meme si le manifeste existe deja.",
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
