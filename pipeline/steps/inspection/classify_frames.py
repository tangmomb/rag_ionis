import json
import os
import re
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import existing_images_dir, images_dir, relative_to_video_dir

DEFAULT_MODEL_PATH = Path("models/frame_filter_2026-07-02_21-30-31.joblib")
DEFAULT_EMBEDDING_CACHE_DIRNAME = ".embedding_cache"
DINO_EMBEDDING_DIMENSIONS = {
    "facebook/dinov2-base": 768,
}
DEFAULT_BATCH_SIZE = 16
_EMBEDDER_CACHE = {}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
FOOTAGE_DIR_NAME = "footage"
GRAPHIC_DIR_NAME = "graphic"
MIXTURE_DIR_NAME = "mixture"
MANIFEST_NAME = "frame_classification_manifest.json"
FEATURES_NAME = "frame_classification_features.json"
LEGACY_MANIFEST_NAME = "manifest.json"
LEGACY_FEATURES_NAME = "cv_features.json"
OLD_EMBEDDINGS_NAME = "dinov2_embeddings.json"
STAGING_DIR_NAME = ".classify_tmp"
LEGACY_STAGING_DIR_NAME = ".cluster_tmp"
LEGACY_OUTPUT_DIR_NAMES = ("answer", "answers", "answer_variants", "no_cluster", "question_intertitles")
SECOND_PATTERN = re.compile(r"^seconde_(\d+(?:_\d+)?)$")
TIMECODE_PATTERN = re.compile(r"^(?:(\d{2})_)?(\d{2})_(\d{2})$")


@dataclass(frozen=True)
class ClassificationOptions:
    model: str | Path = DEFAULT_MODEL_PATH
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str | None = None
    cache_dir: str | Path | None = None
    force: bool = False


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


def existing_manifest_path(images_dir):
    preferred = images_dir / MANIFEST_NAME
    legacy = images_dir / LEGACY_MANIFEST_NAME
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_image_rgb(path, crop_bottom=0.0):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    if crop_bottom > 0:
        width, height = image.size
        keep_height = max(1, int(round(height * (1.0 - crop_bottom))))
        image = image.crop((0, 0, width, keep_height))
    return image


def file_signature(path, stat_path=None):
    import hashlib

    p = Path(path)
    stat = Path(stat_path).stat() if stat_path is not None else p.stat()
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
        self.inference_dtype = os.getenv(
            "FRAME_CLASSIFICATION_DTYPE",
            "float32",
        ).strip().lower()
        if self.inference_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "FRAME_CLASSIFICATION_DTYPE doit valoir float32, float16 ou bfloat16."
            )
        self.dino_processor = AutoImageProcessor.from_pretrained(self.config.dino_model, use_fast=False)
        self.dino_model = AutoModel.from_pretrained(self.config.dino_model).to(self.device)
        self.dino_model.eval()
        for parameter in self.dino_model.parameters():
            parameter.requires_grad_(False)

        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model, use_fast=False)
        self.clip_model = CLIPModel.from_pretrained(self.config.clip_model).to(self.device)
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)

    def prepare_images(self, images):
        """Prepare tensors on CPU so this work can overlap GPU inference."""

        return (
            self.dino_processor(images=list(images), return_tensors="pt"),
            self.clip_processor(images=list(images), return_tensors="pt"),
        )

    def _to_device(self, inputs):
        non_blocking = self.device.type == "cuda"
        moved = {}
        for name, value in inputs.items():
            if non_blocking and hasattr(value, "pin_memory"):
                value = value.pin_memory()
            moved[name] = value.to(self.device, non_blocking=non_blocking)
        return moved

    def embed_prepared(self, prepared):
        import numpy as np

        torch = self.torch
        dino_inputs, clip_inputs = prepared

        autocast_enabled = (
            self.device.type == "cuda" and self.inference_dtype != "float32"
        )
        autocast_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.inference_dtype, torch.float32)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            dino_inputs = self._to_device(dino_inputs)
            dino_outputs = self.dino_model(**dino_inputs)
            dino_vec = dino_outputs.last_hidden_state[:, 0, :]
            dino_vec = torch.nn.functional.normalize(dino_vec, dim=1)

            clip_inputs = self._to_device(clip_inputs)
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

    def embed_images(self, images):
        import numpy as np

        if not images:
            return np.empty((0, 0), dtype=np.float32)
        return self.embed_prepared(self.prepare_images(images))


def get_frozen_backbone_embedder(config, device):
    keep_model = os.getenv(
        "FRAME_CLASSIFICATION_KEEP_MODEL",
        "0",
    ).strip().lower() in {"1", "true", "yes"}
    if not keep_model:
        return FrozenBackboneEmbedder(config=config, device=device)
    key = (
        config.dino_model,
        config.clip_model,
        config.crop_bottom,
        str(device),
        os.getenv("FRAME_CLASSIFICATION_DTYPE", "float32").strip().lower(),
    )
    embedder = _EMBEDDER_CACHE.get(key)
    if embedder is None:
        embedder = FrozenBackboneEmbedder(config=config, device=device)
        _EMBEDDER_CACHE[key] = embedder
    return embedder


def cache_path_for_image(
    image_path,
    cache_dir,
    config,
    *,
    stat_path=None,
):
    signature = stable_hash(
        [
            "frame-filter-embedding-v1",
            file_signature(image_path, stat_path=stat_path),
            config.to_dict(),
        ]
    )
    return cache_dir / f"{signature}.joblib"


def load_cached_dino_embeddings(
    image_paths,
    images_directory,
    cache_dir=None,
):
    import joblib
    import numpy as np

    paths = [Path(path) for path in image_paths]
    images_root = Path(images_directory)
    features_path = images_root / FEATURES_NAME
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features de classification introuvables: {features_path}"
        )

    payload = read_json(features_path)
    backbones = payload.get("backbones", {})
    if not isinstance(backbones, dict):
        backbones = {}
    dino_model = str(
        backbones.get("dino", "facebook/dinov2-base")
    )
    clip_model = str(
        backbones.get("clip", "openai/clip-vit-base-patch32")
    )
    config = EmbeddingConfig(
        dino_model=dino_model,
        clip_model=clip_model,
        crop_bottom=float(payload.get("crop_bottom", 0.20)),
    )
    dimensions = payload.get("embedding_dimensions", {})
    if not isinstance(dimensions, dict):
        dimensions = {}
    dino_dimensions = dimensions.get("dino")
    if not isinstance(dino_dimensions, int) or dino_dimensions <= 0:
        dino_dimensions = DINO_EMBEDDING_DIMENSIONS.get(dino_model)
    if dino_dimensions is None:
        raise ValueError(
            "Dimension DINO inconnue pour "
            f"{dino_model!r}; relancer la classification des frames."
        )

    items_by_image = {
        Path(str(item.get("image", ""))).as_posix(): item
        for item in payload.get("items", [])
        if isinstance(item, dict) and item.get("image")
    }
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else images_root / DEFAULT_EMBEDDING_CACHE_DIRNAME
    )
    vectors = []
    for path in paths:
        try:
            relative_image = path.relative_to(images_root).as_posix()
        except ValueError:
            relative_image = path.name
        item = items_by_image.get(relative_image)
        if item is None:
            raise KeyError(
                f"Frame absente de {features_path}: {relative_image}"
            )

        cache_key = str(item.get("embedding_cache_key", "")).strip()
        if cache_key:
            cache_path = cache_root / cache_key
        else:
            # Les anciens artefacts ont ete mis en cache avant le
            # deplacement de la frame dans footage/graphic/mixture.
            original_path = images_root / str(
                item.get("source_image") or path.name
            )
            cache_path = cache_path_for_image(
                original_path,
                cache_root,
                config,
                stat_path=path,
            )
        if not cache_path.exists():
            raise FileNotFoundError(
                "Embedding de frame introuvable: "
                f"{cache_path}. Relancer la classification des frames."
            )
        vector = np.asarray(joblib.load(cache_path), dtype=np.float32)
        if vector.ndim != 1 or vector.size < dino_dimensions:
            raise ValueError(
                f"Embedding invalide dans {cache_path}: {vector.shape}"
            )
        vectors.append(vector[:dino_dimensions])

    if not vectors:
        return np.empty((0, dino_dimensions), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def prepare_image_batch(paths, batch_indices, embedder):
    images = [
        load_image_rgb(paths[index], crop_bottom=embedder.config.crop_bottom)
        for index in batch_indices
    ]
    return embedder.prepare_images(images)


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

    batches = [
        missing_indices[start : start + batch_size]
        for start in range(0, len(missing_indices), batch_size)
    ]
    supports_prefetch = all(
        callable(getattr(embedder, method, None))
        for method in ("prepare_images", "embed_prepared")
    )

    if supports_prefetch and batches:
        # Double buffering: while the GPU consumes one batch, the CPU decodes and
        # transforms the following batch. Only one preparation thread touches the
        # Hugging Face processors, which keeps their use deterministic.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame-prefetch") as pool:
            prepared = pool.submit(prepare_image_batch, paths, batches[0], embedder)
            for batch_number, batch_indices in enumerate(batches):
                batch_inputs = prepared.result()
                if batch_number + 1 < len(batches):
                    prepared = pool.submit(
                        prepare_image_batch,
                        paths,
                        batches[batch_number + 1],
                        embedder,
                    )
                embeddings = embedder.embed_prepared(batch_inputs)
                for local_idx, original_idx in enumerate(batch_indices):
                    vector = embeddings[local_idx].astype(np.float32)
                    output[original_idx] = vector
                    if cache_root:
                        joblib.dump(
                            vector,
                            cache_path_for_image(
                                paths[original_idx],
                                cache_root,
                                embedder.config,
                            ),
                        )
                completed = min((batch_number + 1) * batch_size, len(missing_indices))
                print(
                    f"[embeddings] {completed}/{len(missing_indices)} images",
                    flush=True,
                )
    else:
        for batch_number, batch_indices in enumerate(batches):
            images = [
                load_image_rgb(paths[index], crop_bottom=embedder.config.crop_bottom)
                for index in batch_indices
            ]
            embeddings = embedder.embed_images(images)
            for local_idx, original_idx in enumerate(batch_indices):
                vector = embeddings[local_idx].astype(np.float32)
                output[original_idx] = vector
                if cache_root:
                    joblib.dump(
                        vector,
                        cache_path_for_image(
                            paths[original_idx],
                            cache_root,
                            embedder.config,
                        ),
                    )
            completed = min((batch_number + 1) * batch_size, len(missing_indices))
            print(
                f"[embeddings] {completed}/{len(missing_indices)} images",
                flush=True,
            )

    ready = [vector for vector in output if vector is not None]
    if len(ready) != len(paths):
        raise RuntimeError("Failed to compute all embeddings")
    return np.vstack(ready).astype(np.float32)


def load_model_payload(path):
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        payload = joblib.load(path)

    version_warnings = [
        warning.message
        for warning in caught
        if isinstance(warning.message, InconsistentVersionWarning)
    ]
    if version_warnings:
        trained_version = getattr(version_warnings[0], "original_sklearn_version", "unknown")
        current_version = getattr(version_warnings[0], "current_sklearn_version", "unknown")
        estimators = ", ".join(
            sorted(
                {
                    str(getattr(warning, "estimator_name", "")).strip()
                    for warning in version_warnings
                    if str(getattr(warning, "estimator_name", "")).strip()
                }
            )
        )
        details = f" ({estimators})" if estimators else ""
        print(
            "[warning] Modele scikit-learn charge avec une version differente: "
            f"entraine en {trained_version}, environnement courant {current_version}{details}. "
            "Le pipeline continue, mais le plus sur est d'aligner scikit-learn sur la version du modele.",
            flush=True,
        )

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
    embedder = get_frozen_backbone_embedder(config=config, device=device)
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
        item = {
            "source_image": path.name,
            "second": image_second(path),
            "pred_label": predicted_label,
            "role": predicted_label,
            **probs,
        }
        if args.cache_dir is not None:
            item["embedding_cache_key"] = cache_path_for_image(
                path,
                Path(args.cache_dir),
                config,
            ).name
        items.append(item)

    return {
        "roles": roles,
        "class_names": class_names,
        "crop_bottom": crop_bottom,
        "backbones": {"dino": dino_model, "clip": clip_model},
        "embedding_dimensions": {
            "dino": int(
                getattr(embedder.dino_model.config, "hidden_size")
            ),
            "clip": int(
                getattr(embedder.clip_model.config, "projection_dim")
            ),
        },
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
        relative_target = relative_to_video_dir(target, video_path)
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
        "source": relative_to_video_dir(images_dir, video_path),
        "model": str(Path(model_path).as_posix()),
        "class_names": class_names,
        "crop_bottom": prediction_payload["crop_bottom"],
        "backbones": prediction_payload["backbones"],
        "embedding_dimensions": prediction_payload[
            "embedding_dimensions"
        ],
        "items": sorted(items, key=lambda item: item["image"]),
    }
    write_json(features_path, features_payload)

    manifest = {
        "method": "frame_filter_model",
        "source": relative_to_video_dir(images_dir, video_path),
        "output_dir": relative_to_video_dir(images_dir, video_path),
        "model": str(Path(model_path).as_posix()),
        "class_names": class_names,
        "crop_bottom": prediction_payload["crop_bottom"],
        "backbones": prediction_payload["backbones"],
        "embedding_dimensions": prediction_payload[
            "embedding_dimensions"
        ],
        "identification_strategy": "frame_filter_model",
        "class_counts": role_counts,
        "footage_count": role_counts.get("footage", 0),
        "graphic_count": role_counts.get("graphic", 0),
        "mixture_count": role_counts.get("mixture", 0),
        "features": relative_to_video_dir(features_path, video_path),
        "classes": [
            {"label": class_name, "count": role_counts[class_name], "images": role_images[class_name]}
            for class_name in class_names
        ],
        "items": sorted(items, key=lambda item: item["image"]),
    }
    write_json(manifest_path, manifest)
    print(
        f"[ok] {video_path.name}: footage={role_counts.get('footage', 0)}, graphic={role_counts.get('graphic', 0)}, mixture={role_counts.get('mixture', 0)} -> {manifest_path}",
        flush=True,
    )
    return manifest_path


def classify_video_images(video_path, args):
    video_images_dir = existing_images_dir(video_path)
    paths = image_files(video_images_dir)
    if not paths:
        print(f"[skip] {video_path.name}: aucune image dans {video_images_dir}")
        return None

    manifest_path = existing_manifest_path(video_images_dir)
    if manifest_path.exists() and not args.force:
        print(f"[skip] {video_path.name}: {manifest_path} existe deja")
        return manifest_path

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable: {model_path}")

    print(f"[analyse] {video_path.name}: {len(paths)} images, model={model_path.name}", flush=True)
    prediction_payload = classify_paths(paths, args)
    return write_outputs(video_path, paths, video_images_dir, args.force, prediction_payload, model_path)


def classify_video(
    video_path,
    *,
    model: str | Path = DEFAULT_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
):
    """API Python nommee pour classifier les frames d'une video."""

    image_directory = existing_images_dir(video_path)
    selected_cache_dir = (
        cache_dir
        if cache_dir is not None
        else image_directory / DEFAULT_EMBEDDING_CACHE_DIRNAME
    )
    return classify_video_images(
        video_path,
        ClassificationOptions(
            model=model,
            batch_size=batch_size,
            device=device,
            cache_dir=selected_cache_dir,
            force=force,
        ),
    )
