import inspect
import importlib.machinery
import os
import re
import statistics
import sys
import types
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.support.paths import existing_images_dir


_NVIDIA_DLL_HANDLES = []


def configure_nvidia_dll_paths():
    """Expose les DLL CUDA installées par PyTorch au runtime Paddle sous Windows."""
    if os.name != "nt":
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    bin_dirs = [path / "bin" for path in nvidia_root.glob("*") if (path / "bin").is_dir()]
    if not bin_dirs:
        return
    os.environ["PATH"] = os.pathsep.join(map(str, bin_dirs)) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        for bin_dir in bin_dirs:
            try:
                _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(str(bin_dir)))
            except OSError:
                pass


configure_nvidia_dll_paths()


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SECOND_PATTERN = re.compile(r"^seconde_(\d+(?:_\d+)?)$")
TIMECODE_PATTERN = re.compile(r"^(?:(\d{2})_)?(\d{2})_(\d{2})(?:_(\d{3}))?$")
IGNORED_TEXT_KEYS = {"ionis", "kionis", "<ionis", "stm"}
DECOR_TEXT_KEYS = IGNORED_TEXT_KEYS | {"x", "in"}
MIN_OVERLAY_RELATIVE_HEIGHT = 0.03
MIN_OVERLAY_RELATIVE_WIDTH = 0.24
MIN_SUBTITLE_CLUSTER_SECONDS = 10
MIN_SUBTITLE_CLUSTER_DURATION = 5
STATIC_DECOR_MIN_SECONDS = 5
STATIC_DECOR_MIN_DURATION = 5
STATIC_DECOR_POSITION_TOLERANCE = 0.045
STATIC_DECOR_MIN_TEXT_LENGTH = 8
STATIC_DECOR_VARIANT_RATIO = 0.82
STATIC_DECOR_CONTAINED_RATIO = 0.65
PROGRESSIVE_TEXT_WINDOW_SECONDS = 4
PROGRESSIVE_TEXT_POSITION_TOLERANCE = 0.10
PROGRESSIVE_QUESTION_POSITION_TOLERANCE = 0.24
PROGRESSIVE_TEXT_MIN_EXTRA_CHARS = 2
PROGRESSIVE_TEXT_MIN_OVERLAP_RATIO = 0.75
SUBTITLE_CLUSTER_X_TOLERANCE = 0.16
SUBTITLE_CLUSTER_Y_TOLERANCE = 0.055


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
        milliseconds = int(timecode_match.group(4) or 0)
        return hours * 3600 + minutes * 60 + seconds + (milliseconds / 1000.0)

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
            for path in video_images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: (image_second(path), path.as_posix()),
    )


def image_video_dirs(video_dir):
    direct_images_dir = existing_images_dir(video_dir)
    if direct_images_dir.is_dir() and any(image_files(direct_images_dir)):
        # This is already a video directory. Do not inspect its children:
        # ``outputs`` also contains ``outputs/images`` and would otherwise be
        # misidentified as a second video, producing outputs/outputs/...
        return [video_dir]

    candidates = []
    for child in sorted(video_dir.iterdir()):
        child_images_dir = existing_images_dir(child)
        if child.is_dir() and child_images_dir.is_dir() and any(image_files(child_images_dir)):
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


def compact_text_key(text):
    return re.sub(r"\W+", "", normalize_detected_text(text).casefold())


def confidence_label(score):
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


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


def box_geometry(box, image_size):
    width, height = image_size or (0, 0)
    if not box or not width or not height:
        return {
            "cx": 0.0,
            "cy": 0.0,
            "relative_width": 0.0,
            "relative_height": 0.0,
        }

    x1, y1, x2, y2 = box
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    return {
        "cx": clamp(((x1 + x2) / 2) / width),
        "cy": clamp(((y1 + y2) / 2) / height),
        "relative_width": clamp(box_width / width),
        "relative_height": clamp(box_height / height),
    }


def is_primary_subtitle_box(cx, cy, relative_width, relative_height, word_count, has_sentence_punctuation):
    return (
        0.62 <= cy <= 0.88
        and 0.42 <= cx <= 0.58
        and 0.025 <= relative_height <= 0.075
        and (
            relative_width >= 0.24
            or has_sentence_punctuation
            or word_count >= 3
        )
    )


def is_subtitle_anchor_candidate(entry):
    geometry = entry["geometry"]
    return (
        0.35 <= geometry["cx"] <= 0.65
        and 0.60 <= geometry["cy"] <= 0.94
        and 0.018 <= geometry["relative_height"] <= 0.09
        and geometry["relative_width"] >= 0.06
    )


def subtitle_cluster_score(cluster):
    seconds = sorted({entry["second"] for entry in cluster if entry["second"] is not None})
    if len(seconds) < MIN_SUBTITLE_CLUSTER_SECONDS:
        return 0.0
    if seconds[-1] - seconds[0] < MIN_SUBTITLE_CLUSTER_DURATION:
        return 0.0

    median_cx = statistics.median(entry["geometry"]["cx"] for entry in cluster)
    median_cy = statistics.median(entry["geometry"]["cy"] for entry in cluster)
    x_spread = max(entry["geometry"]["cx"] for entry in cluster) - min(entry["geometry"]["cx"] for entry in cluster)
    y_spread = max(entry["geometry"]["cy"] for entry in cluster) - min(entry["geometry"]["cy"] for entry in cluster)
    centrality = max(0.0, 1.0 - abs(median_cx - 0.5) * 2.0)
    temporal_density = min(len(seconds), len(cluster))
    stability = max(0.0, 1.0 - (x_spread + y_spread))

    return (
        temporal_density * 2.0
        + len(cluster) * 0.5
        + centrality * 4.0
        + stability * 4.0
        - abs(median_cy - 0.78) * 2.0
    )


def infer_subtitle_anchors(entries):
    candidates = [entry for entry in entries if is_subtitle_anchor_candidate(entry)]
    if len(candidates) < 2:
        return []

    best_cluster = []
    best_score = 0.0
    for candidate in candidates:
        cluster = [
            other
            for other in candidates
            if abs(other["geometry"]["cx"] - candidate["geometry"]["cx"]) <= SUBTITLE_CLUSTER_X_TOLERANCE
            and abs(other["geometry"]["cy"] - candidate["geometry"]["cy"]) <= SUBTITLE_CLUSTER_Y_TOLERANCE
        ]
        score = subtitle_cluster_score(cluster)
        if score > best_score:
            best_cluster = cluster
            best_score = score

    if len(best_cluster) < 2:
        return []
    return [entry["geometry"] for entry in best_cluster]


def anchored_subtitle_match(geometry, anchor_cx, anchor_cy, x_tolerance, y_tolerance):
    return (
        abs(geometry["cx"] - anchor_cx) <= x_tolerance
        and abs(geometry["cy"] - anchor_cy) <= y_tolerance
        and geometry["relative_width"] >= 0.06
        and 0.02 <= geometry["relative_height"] <= 0.09
    )


def subtitle_text_signal(text, relative_width):
    cleaned = normalize_detected_text(text)
    word_count = len(re.findall(r"\w+", cleaned, flags=re.UNICODE))
    has_sentence_punctuation = any(mark in cleaned for mark in ".?!")
    return word_count, has_sentence_punctuation, (
        relative_width >= 0.24
        or has_sentence_punctuation
        or word_count >= 3
    )


def classify_text(text, box, image_size):
    cleaned = normalize_detected_text(text)
    if not cleaned:
        return "others"

    geometry = box_geometry(box, image_size)
    cx = geometry["cx"]
    cy = geometry["cy"]
    relative_width = geometry["relative_width"]
    relative_height = geometry["relative_height"]

    word_count, has_sentence_punctuation, _ = subtitle_text_signal(cleaned, relative_width)

    if is_primary_subtitle_box(cx, cy, relative_width, relative_height, word_count, has_sentence_punctuation):
        return "subtitle"
    return "others"


def is_graphic_image_name(image_name):
    return str(image_name or "").replace("\\", "/").split("/", 1)[0] == "graphic"


def is_answer_image_name(image_name):
    prefix = str(image_name or "").replace("\\", "/").split("/", 1)[0]
    return prefix in {"answers", "footage", "mixture"}


def graphic_sequence_key(image_name):
    return None


def graphic_kind_for_image(image_name):
    parts = str(image_name or "").replace("\\", "/").split("/")
    if parts and parts[0] == "graphic":
        return "graphic"
    return None


def is_graphic_kind(kind):
    return kind == "graphic"


def last_graphic_sequence_key(images_dir):
    sequence = None
    for path in image_files(images_dir):
        relative_name = path.relative_to(images_dir).as_posix()
        current_sequence = graphic_sequence_key(relative_name)
        if current_sequence:
            sequence = current_sequence
    return sequence


def refine_subtitle_kinds(items, images_dir):
    geometries = []
    sizes = {}

    for item in items:
        graphic_kind = graphic_kind_for_image(item.get("image"))
        image_name = item.get("image")
        size = None
        if image_name and image_name not in sizes:
            sizes[image_name] = image_size(images_dir / image_name)
        if image_name:
            size = sizes[image_name]
        box = box_bounds(item.get("box"))
        geometry = box_geometry(box, size)
        word_count, has_sentence_punctuation, has_subtitle_signal = subtitle_text_signal(
            item.get("text", ""),
            geometry["relative_width"],
        )
        geometries.append(
            {
                "item": item,
                "geometry": geometry,
                "word_count": word_count,
                "has_sentence_punctuation": has_sentence_punctuation,
                "has_subtitle_signal": has_subtitle_signal,
                "key": text_key(item.get("text", "")),
                "second": item.get("second"),
                "fallback_kind": graphic_kind or item.get("kind"),
            }
        )

    anchors = infer_subtitle_anchors(geometries)

    if len(anchors) < 2:
        refined = []
        for entry in geometries:
            item = dict(entry["item"])
            fallback_kind = entry["fallback_kind"]
            if fallback_kind == "subtitle":
                item["kind"] = "others"
            elif fallback_kind is not None:
                item["kind"] = fallback_kind
            refined.append(item)
        return refined

    anchor_cx = statistics.median(anchor["cx"] for anchor in anchors)
    anchor_cy = statistics.median(anchor["cy"] for anchor in anchors)
    anchor_height = statistics.median(anchor["relative_height"] for anchor in anchors)
    anchor_widths = [anchor["relative_width"] for anchor in anchors]
    x_tolerance = max(0.06, min(0.16, statistics.median(anchor_widths) * 0.25))
    y_tolerance = max(0.04, min(0.075, anchor_height * 1.6))

    refined = []
    for entry in geometries:
        item = dict(entry["item"])
        geometry = entry["geometry"]
        fallback_kind = entry["fallback_kind"]
        is_near_anchor = anchored_subtitle_match(
            geometry,
            anchor_cx,
            anchor_cy,
            x_tolerance,
            y_tolerance,
        )
        if is_near_anchor:
            item["kind"] = "subtitle"
        elif fallback_kind == "subtitle":
            item["kind"] = "others"
        elif fallback_kind is not None:
            item["kind"] = fallback_kind
        refined.append(item)

    return refined


def boxes_are_grouped(left, right):
    if left is None or right is None:
        return False

    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    left_width = max(0.0, left_x2 - left_x1)
    right_width = max(0.0, right_x2 - right_x1)
    if not left_width or not right_width:
        return False

    horizontal_overlap = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    horizontal_gap = max(0.0, max(left_x1, right_x1) - min(left_x2, right_x2))
    vertical_gap = max(0.0, max(left_y1, right_y1) - min(left_y2, right_y2))
    min_width = min(left_width, right_width)
    max_height = max(max(0.0, left_y2 - left_y1), max(0.0, right_y2 - right_y1))

    return (
        horizontal_overlap >= min_width * 0.25
        and vertical_gap <= max(8.0, max_height * 1.2)
    ) or (
        horizontal_gap <= min_width * 0.8
        and vertical_gap <= max(6.0, max_height * 0.7)
    )


def similar_static_text(left, right):
    left_text = left["compact_key"]
    right_text = right["compact_key"]
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True

    shorter, longer = sorted((left_text, right_text), key=len)
    if len(shorter) < STATIC_DECOR_MIN_TEXT_LENGTH:
        return False
    if shorter in longer and len(shorter) / len(longer) >= STATIC_DECOR_CONTAINED_RATIO:
        return True

    return SequenceMatcher(None, left_text, right_text).ratio() >= STATIC_DECOR_VARIANT_RATIO


def same_static_text_position(left, right):
    return (
        similar_static_text(left, right)
        and abs(left["geometry"]["cx"] - right["geometry"]["cx"]) <= STATIC_DECOR_POSITION_TOLERANCE
        and abs(left["geometry"]["cy"] - right["geometry"]["cy"]) <= STATIC_DECOR_POSITION_TOLERANCE
    )


def static_decor_keys(entries):
    groups = []
    for entry in entries:
        if entry["item"].get("kind") == "subtitle" or not entry["key"]:
            continue

        target_group = None
        for group in groups:
            if same_static_text_position(entry, group[0]):
                target_group = group
                break
        if target_group is None:
            groups.append([entry])
        else:
            target_group.append(entry)

    static_keys = set()
    for group in groups:
        seconds = sorted(
            {
                entry["item"].get("second")
                for entry in group
                if entry["item"].get("second") is not None
            }
        )
        if len(seconds) < STATIC_DECOR_MIN_SECONDS:
            continue
        if seconds[-1] - seconds[0] < STATIC_DECOR_MIN_DURATION:
            continue
        for entry in group:
            static_keys.add((entry["key"], entry["item"].get("image"), entry["item"].get("text")))

    return static_keys


def same_progressive_text_area(left, right):
    tolerance = PROGRESSIVE_TEXT_POSITION_TOLERANCE
    left_text = left["compact_key"]
    right_text = right["compact_key"]
    if (
        normalize_detected_text(left["item"].get("text", "")).endswith("?")
        or normalize_detected_text(right["item"].get("text", "")).endswith("?")
        or (left_text and right_text and (left_text in right_text or right_text in left_text))
    ):
        tolerance = PROGRESSIVE_QUESTION_POSITION_TOLERANCE
    return (
        abs(left["geometry"]["cx"] - right["geometry"]["cx"]) <= tolerance
        and abs(left["geometry"]["cy"] - right["geometry"]["cy"]) <= tolerance
    )


def edge_overlap_length(left, right):
    max_overlap = min(len(left), len(right))
    for length in range(max_overlap, 0, -1):
        if left[-length:] == right[:length] or right[-length:] == left[:length]:
            return length
    return 0


def is_progressive_fragment(shorter, longer):
    short_text = shorter["compact_key"]
    long_text = longer["compact_key"]
    if not short_text or not long_text:
        return False
    if len(long_text) - len(short_text) < PROGRESSIVE_TEXT_MIN_EXTRA_CHARS:
        return False
    if short_text in long_text:
        return True
    overlap = edge_overlap_length(short_text, long_text)
    return overlap / len(short_text) >= PROGRESSIVE_TEXT_MIN_OVERLAP_RATIO


def progressive_fragment_keys(entries):
    fragments = set()
    candidates = [
        entry
        for entry in entries
        if entry["item"].get("kind") != "subtitle"
        and entry["compact_key"]
        and entry["item"].get("second") is not None
    ]

    for entry in candidates:
        for other in candidates:
            if entry is other:
                continue
            if abs(entry["item"].get("second") - other["item"].get("second")) > PROGRESSIVE_TEXT_WINDOW_SECONDS:
                continue
            if not same_progressive_text_area(entry, other):
                continue
            if is_progressive_fragment(entry, other):
                fragments.add((entry["key"], entry["item"].get("image"), entry["item"].get("text")))
                break

    return fragments


def filter_decor_items(items, images_dir):
    entries = []
    by_image = {}
    sizes = {}
    passthrough_items = []

    for item in items:
        graphic_kind = graphic_kind_for_image(item.get("image"))
        if item.get("kind") == "subtitle":
            passthrough_items.append(item)
            continue
        if is_graphic_kind(item.get("kind")) or graphic_kind:
            item = dict(item)
            item["kind"] = graphic_kind or item.get("kind")
            passthrough_items.append(item)
            continue

        image_name = item.get("image")
        size = None
        if image_name and image_name not in sizes:
            sizes[image_name] = image_size(images_dir / image_name)
        if image_name:
            size = sizes[image_name]
        box = box_bounds(item.get("box"))
        geometry = box_geometry(box, size)
        word_count, _, _ = subtitle_text_signal(item.get("text", ""), geometry["relative_width"])
        entry = {
            "item": item,
            "box": box,
            "geometry": geometry,
            "key": text_key(item.get("text", "")),
            "compact_key": compact_text_key(item.get("text", "")),
            "word_count": word_count,
        }
        entries.append(entry)
        by_image.setdefault(image_name, []).append(entry)

    static_keys = static_decor_keys(entries)
    progressive_keys = progressive_fragment_keys(entries)
    filtered = list(passthrough_items)
    for entry in entries:
        item = entry["item"]
        if item.get("kind") == "subtitle":
            filtered.append(item)
            continue

        key = entry["key"]
        if not key or key in DECOR_TEXT_KEYS:
            continue
        if (key, item.get("image"), item.get("text")) in static_keys:
            continue
        if (key, item.get("image"), item.get("text")) in progressive_keys:
            continue

        geometry = entry["geometry"]
        is_readable_overlay = (
            geometry["relative_height"] >= MIN_OVERLAY_RELATIVE_HEIGHT
            or geometry["relative_width"] >= MIN_OVERLAY_RELATIVE_WIDTH
        )
        if not is_readable_overlay:
            continue

        grouped = any(
            other is not entry
            and other["item"].get("kind") != "subtitle"
            and boxes_are_grouped(entry["box"], other["box"])
            for other in by_image.get(item.get("image"), [])
        )
        compact_isolated_text = entry["word_count"] <= 1 and len(key) <= 4 and not grouped
        if compact_isolated_text and geometry["relative_height"] < 0.07 and geometry["relative_width"] < 0.35:
            continue

        filtered.append(item)

    return filtered


def collapse_graphic_sequence_items(items):
    groups = {}
    for item in items:
        if not is_graphic_kind(item.get("kind")):
            continue
        sequence = graphic_sequence_key(item.get("image"))
        if not sequence:
            continue
        groups.setdefault(sequence, {}).setdefault(item.get("image"), []).append(item)

    selected_images = {}
    for sequence, images in groups.items():
        best_image = max(
            sorted(images),
            key=lambda image_name: (
                sum(len(normalize_detected_text(item.get("text", ""))) for item in images[image_name]),
                sum(len(re.findall(r"\w+", normalize_detected_text(item.get("text", "")), flags=re.UNICODE)) for item in images[image_name]),
                sum(float(item.get("score") or 0.0) for item in images[image_name]) / max(1, len(images[image_name])),
                -float(seconds_from_image_name(Path(image_name).name) or 0.0),
            ),
        )
        selected_images[sequence] = best_image

    collapsed = []
    for item in items:
        sequence = graphic_sequence_key(item.get("image"))
        if sequence and item.get("image") != selected_images.get(sequence):
            continue
        collapsed.append(item)
    return collapsed


def mark_last_graphic_sequence_as_outro(items, images_dir):
    del images_dir
    return list(items)


def image_order_map(images_dir):
    if not images_dir:
        return {}
    return {
        path.relative_to(images_dir).as_posix(): index
        for index, path in enumerate(image_files(images_dir))
    }


def item_frame_index(item, order_map):
    image_name = item.get("image")
    if image_name in order_map:
        return order_map[image_name]
    second = item.get("second")
    if second is not None:
        return float(second)
    return image_second(Path(image_name or ""))


def item_time_index(item):
    second = item.get("second")
    if second is not None:
        return float(second)
    image_name = item.get("image")
    if image_name:
        parsed = seconds_from_image_name(Path(image_name).name)
        if parsed is not None:
            return float(parsed)
    return float("inf")


def collapse_answer_overlay_items(items, images_dir=None, time_window_seconds=10):
    grouped = {}
    passthrough = []

    for position, item in enumerate(items):
        if (
            is_answer_image_name(item.get("image"))
            and item.get("kind") != "subtitle"
            and not is_graphic_kind(item.get("kind"))
        ):
            key = compact_text_key(item.get("text", ""))
            if key:
                grouped.setdefault(key, []).append((position, item_time_index(item), item))
                continue
        passthrough.append((position, item))

    keep_positions = {position for position, _ in passthrough}
    for occurrences in grouped.values():
        sorted_occurrences = sorted(occurrences, key=lambda value: (value[1], value[0]))
        window_start = None
        last_position = None

        for position, time_index, _ in sorted_occurrences:
            if window_start is None:
                window_start = time_index
                last_position = position
                continue
            if time_index - window_start <= time_window_seconds:
                last_position = position
                continue

            keep_positions.add(last_position)
            window_start = time_index
            last_position = position

        if last_position is not None:
            keep_positions.add(last_position)

    return [item for position, item in enumerate(items) if position in keep_positions]


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
    def __init__(self, device="gpu:0", lang="fr", min_confidence=0.9):
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


def records_from_raw_result(raw_result, min_confidence=0.9):
    records = []

    if isinstance(raw_result, dict):
        texts = list(raw_result.get("rec_texts") or [])
        scores = list(raw_result.get("rec_scores") or [])
        polys = list(raw_result.get("rec_polys") or raw_result.get("dt_polys") or [])
        for index, text in enumerate(texts):
            cleaned = normalize_detected_text(text)
            score = float(scores[index]) if index < len(scores) else 0.0
            if not cleaned or score < min_confidence:
                continue
            poly = polys[index] if index < len(polys) else None
            records.append({"text": cleaned, "score": score, "poly": point_list(poly), "box": box_bounds(poly)})
        return records

    if isinstance(raw_result, list):
        for page in raw_result:
            if isinstance(page, dict):
                records.extend(records_from_raw_result(page, min_confidence=min_confidence))
                continue
            lines = page or []
            for line in lines:
                if not line or len(line) < 2:
                    continue
                poly, value = line[0], line[1]
                if not isinstance(value, (list, tuple)) or len(value) < 2:
                    continue
                text, score = value[0], float(value[1] or 0.0)
                cleaned = normalize_detected_text(text)
                if not cleaned or score < min_confidence:
                    continue
                records.append({"text": cleaned, "score": score, "poly": point_list(poly), "box": box_bounds(poly)})
        return records

    return records


def boxes_from_raw_result(raw_result):
    boxes = []

    if isinstance(raw_result, dict):
        polys = list(raw_result.get("rec_polys") or raw_result.get("dt_polys") or [])
        for poly in polys:
            points = point_list(poly)
            if points:
                boxes.append(points)
        return boxes

    if isinstance(raw_result, list):
        for page in raw_result:
            if isinstance(page, dict):
                boxes.extend(boxes_from_raw_result(page))
                continue
            lines = page or []
            for line in lines:
                if not line:
                    continue
                poly = line[0] if isinstance(line, (list, tuple)) and line else None
                points = point_list(poly)
                if points:
                    boxes.append(points)
        return boxes

    return boxes


def ocr_items_from_raw_result(raw_result, image_name, image_path=None, min_confidence=0.9):
    size = image_size(image_path) if image_path is not None else None
    second = seconds_from_image_name(Path(image_name).name)
    image_kind = graphic_kind_for_image(image_name)
    items = []
    seen = set()
    for record in records_from_raw_result(raw_result, min_confidence=min_confidence):
        key = text_key(record["text"])
        if not key or key in seen or key in IGNORED_TEXT_KEYS:
            continue
        seen.add(key)
        item = {
            "image": image_name,
            "text": record["text"],
            "kind": image_kind or classify_text(record["text"], record.get("box"), size),
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


def ocr_items_for_image(ocr, image_path, images_dir=None):
    size = image_size(image_path)
    second = seconds_from_image_name(image_path.name)
    image_name = image_path.name
    if images_dir is not None:
        image_name = image_path.relative_to(images_dir).as_posix()
    image_kind = graphic_kind_for_image(image_name)
    items = []
    seen = set()
    for record in ocr.recognize(image_path):
        key = text_key(record["text"])
        if not key or key in seen or key in IGNORED_TEXT_KEYS:
            continue
        seen.add(key)
        item = {
            "image": image_name,
            "text": record["text"],
            "kind": image_kind or classify_text(record["text"], record.get("box"), size),
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


def item_box_order(item):
    box = box_bounds(item.get("box"))
    if not box:
        return (float("inf"), float("inf"))
    x1, y1, _, _ = box
    return (y1, x1)


def deduplicate_items(items, images_dir=None):
    order_map = image_order_map(images_dir)
    seen = set()
    deduplicated = []
    indexed_items = enumerate(items)
    sorted_items = sorted(
        indexed_items,
        key=lambda value: (
            item_frame_index(value[1], order_map),
            value[1].get("image", ""),
            *item_box_order(value[1]),
            value[1].get("text", ""),
            value[0],
        ),
    )
    for _, item in sorted_items:
        key = (text_key(item.get("text", "")), item.get("second"))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated
