import inspect
import importlib.machinery
import re
import sys
import types
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SECOND_PATTERN = re.compile(r"^seconde_(\d+(?:_\d+)?)$")
TIMECODE_PATTERN = re.compile(r"^(?:(\d{2})_)?(\d{2})_(\d{2})$")
IGNORED_TEXT_KEYS = {"ionis", "kionis", "<ionis", "stm"}


def configure_stdio():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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


def image_files(video_images_dir):
    return sorted(
        (
            path
            for path in video_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=image_second,
    )


def image_video_dirs(video_dir):
    candidates = []

    if (video_dir / "images").is_dir() and any(image_files(video_dir / "images")):
        candidates.append(video_dir)

    for child in sorted(video_dir.iterdir()):
        if child.is_dir() and (child / "images").is_dir() and any(image_files(child / "images")):
            candidates.append(child)

    return candidates


def latest_video_dir(parent_dir):
    candidates = sorted(path for path in parent_dir.iterdir() if path.is_dir() and any(image_video_dirs(path)))
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier avec images trouve dans {parent_dir}")
    return candidates[-1]


def format_timecode(seconds):
    seconds = int(seconds or 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def normalize_detected_text(text):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    text = " ".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def text_key(text):
    text = re.sub(r"\s+", " ", normalize_detected_text(text).casefold())
    return text.strip(" \t\r\n,.;:!?()[]{}\"'")


def confidence_label(score):
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def point_list(poly):
    if poly is None:
        return []
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    return [[float(point[0]), float(point[1])] for point in poly if len(point) >= 2]


def box_bounds(poly):
    points = point_list(poly)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def classify_text(text, box, image_size):
    cleaned = normalize_detected_text(text)
    if not cleaned:
        return "other"

    width, height = image_size or (0, 0)
    lower_band = False
    relative_width = 0.0
    if box and width and height:
        x1, y1, x2, y2 = box
        y_center = (y1 + y2) / 2
        lower_band = y_center >= height * 0.66
        relative_width = (x2 - x1) / width

    word_count = len(re.findall(r"\w+", cleaned, flags=re.UNICODE))
    has_sentence_punctuation = any(mark in cleaned for mark in ".?!")
    mostly_upper = cleaned.isupper() and len(cleaned) >= 3
    title_like = word_count <= 8 and (mostly_upper or cleaned[:1].isupper())

    if word_count <= 4 and title_like and not has_sentence_punctuation and "'" not in cleaned and "," not in cleaned and not any(char.isdigit() for char in cleaned):
        return "name"
    if lower_band and (word_count >= 5 or relative_width >= 0.25 or has_sentence_punctuation):
        return "subtitle"
    if word_count <= 10 and title_like:
        return "title"
    return "other"


def image_size(path):
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def install_torch_import_stub():
    if "torch" in sys.modules:
        return

    class TorchPlaceholder:
        def __init__(self, *_, **__):
            pass

        def __call__(self, *_, **__):
            return self

        def __iter__(self):
            return iter(())

        def __getattr__(self, _):
            return self

    class TorchStubModule(types.ModuleType):
        def __getattr__(self, _):
            return TorchPlaceholder()

    torch_stub = TorchStubModule("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch_stub.__version__ = "0.0.0"
    torch_stub.compile = lambda model, **_: model
    torch_stub.Tensor = TorchPlaceholder

    multiprocessing_stub = types.ModuleType("torch.multiprocessing")
    multiprocessing_stub.__spec__ = importlib.machinery.ModuleSpec("torch.multiprocessing", loader=None)
    multiprocessing_stub.get_start_method = lambda allow_none=True: "spawn"
    multiprocessing_stub.set_start_method = lambda *_, **__: None

    distributed_stub = types.ModuleType("torch.distributed")
    distributed_stub.__spec__ = importlib.machinery.ModuleSpec("torch.distributed", loader=None)
    distributed_stub.is_available = lambda: False
    distributed_stub.is_initialized = lambda: False
    distributed_stub.get_rank = lambda: 0
    distributed_stub.get_world_size = lambda: 1

    nn_stub = TorchStubModule("torch.nn")
    nn_stub.__spec__ = importlib.machinery.ModuleSpec("torch.nn", loader=None)
    nn_stub.Module = TorchPlaceholder
    functional_stub = TorchStubModule("torch.nn.functional")
    functional_stub.__spec__ = importlib.machinery.ModuleSpec("torch.nn.functional", loader=None)

    torch_stub.multiprocessing = multiprocessing_stub
    torch_stub.distributed = distributed_stub
    torch_stub.nn = nn_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.multiprocessing"] = multiprocessing_stub
    sys.modules["torch.distributed"] = distributed_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = functional_stub


class LocalPaddleOCR:
    def __init__(self, device="gpu:0", lang="fr", min_confidence=0.45):
        self.device = device
        self.lang = lang
        self.min_confidence = min_confidence
        self.backend = None
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        install_torch_import_stub()
        try:
            from paddleocr import PaddleOCR

            signature = inspect.signature(PaddleOCR)
            accepted = set(signature.parameters)
            requested = {
                "lang": self.lang,
                "use_textline_orientation": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_angle_cls": False,
                "use_gpu": self.device.startswith("gpu"),
                "show_log": False,
            }
            kwargs = {key: value for key, value in requested.items() if key in accepted}
            self.engine = PaddleOCR(**kwargs)
            self.backend = "paddleocr"
            return
        except Exception as error:
            print(f"[info] PaddleOCR direct indisponible ou non initialise ({error}); fallback PaddleX.", flush=True)

        try:
            from paddlex import create_pipeline
        except ImportError as error:
            raise RuntimeError(
                "PaddleOCR n'est pas installe. Installe d'abord paddlepaddle-gpu puis paddleocr."
            ) from error

        self.engine = create_pipeline(pipeline="OCR", device=self.device)
        self.backend = "paddlex"

    def recognize(self, image_path):
        if self.backend == "paddlex":
            return self._recognize_paddlex(image_path)
        return self._recognize_paddleocr(image_path)

    def recognize_raw(self, image_path):
        if self.backend == "paddlex":
            return self._recognize_paddlex_raw(image_path)
        return self._recognize_paddleocr_raw(image_path)

    def _recognize_paddlex(self, image_path):
        records = []
        output = self.engine.predict(
            input=str(image_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=self.min_confidence,
        )
        for result in output:
            payload = getattr(result, "json", result)
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            records.extend(self._records_from_payload(payload))
        return records

    def _recognize_paddlex_raw(self, image_path):
        raw_items = []
        output = self.engine.predict(
            input=str(image_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=self.min_confidence,
        )
        for result in output:
            payload = getattr(result, "json", result)
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            raw_items.append(payload)
        return raw_items

    def _records_from_payload(self, payload):
        texts = list(payload.get("rec_texts") or [])
        scores = list(payload.get("rec_scores") or [])
        polys = list(payload.get("rec_polys") or payload.get("dt_polys") or [])
        records = []
        for index, text in enumerate(texts):
            cleaned = normalize_detected_text(text)
            score = float(scores[index]) if index < len(scores) else 0.0
            if not cleaned or score < self.min_confidence:
                continue
            poly = polys[index] if index < len(polys) else None
            records.append({"text": cleaned, "score": score, "poly": point_list(poly), "box": box_bounds(poly)})
        return records

    def _recognize_paddleocr(self, image_path):
        if hasattr(self.engine, "predict"):
            try:
                records = []
                for result in self.engine.predict(str(image_path)):
                    payload = getattr(result, "json", result)
                    if isinstance(payload, dict) and "res" in payload:
                        payload = payload["res"]
                    if isinstance(payload, dict):
                        records.extend(self._records_from_payload(payload))
                return records
            except (AttributeError, NotImplementedError):
                pass

        raw = self.engine.ocr(str(image_path), cls=False)
        records = []
        for page in raw or []:
            lines = page or []
            for line in lines:
                if not line or len(line) < 2:
                    continue
                poly, value = line[0], line[1]
                if not isinstance(value, (list, tuple)) or len(value) < 2:
                    continue
                text, score = value[0], float(value[1] or 0.0)
                cleaned = normalize_detected_text(text)
                if not cleaned or score < self.min_confidence:
                    continue
                records.append({"text": cleaned, "score": score, "poly": point_list(poly), "box": box_bounds(poly)})
        return records

    def _recognize_paddleocr_raw(self, image_path):
        if hasattr(self.engine, "predict"):
            try:
                raw_items = []
                for result in self.engine.predict(str(image_path)):
                    payload = getattr(result, "json", result)
                    if isinstance(payload, dict) and "res" in payload:
                        payload = payload["res"]
                    raw_items.append(payload)
                return raw_items
            except (AttributeError, NotImplementedError):
                pass
        return self.engine.ocr(str(image_path), cls=False)


def ocr_items_for_image(ocr, image_path):
    size = image_size(image_path)
    second = seconds_from_image_name(image_path.name)
    items = []
    seen = set()
    for record in ocr.recognize(image_path):
        key = text_key(record["text"])
        if not key or key in seen or key in IGNORED_TEXT_KEYS:
            continue
        seen.add(key)
        item = {
            "image": image_path.name,
            "text": record["text"],
            "kind": classify_text(record["text"], record.get("box"), size),
            "confidence": confidence_label(float(record.get("score") or 0.0)),
            "score": round(float(record.get("score") or 0.0), 4),
        }
        if second is not None:
            item["second"] = second
            item["timecode"] = format_timecode(second)
        if record.get("poly"):
            item["box"] = record["poly"]
        items.append(item)
    return items


def deduplicate_items(items):
    seen = set()
    deduplicated = []
    for item in sorted(items, key=lambda value: (value.get("second", 0), value.get("image", ""), value.get("text", ""))):
        key = (text_key(item.get("text", "")), item.get("second"))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated
