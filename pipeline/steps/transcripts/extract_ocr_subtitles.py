from pipeline.support.analysis import routing_fact
from pipeline.support.json_io import read_json
from pipeline.support.paths import OCR_DIR_NAME, existing_ocr_dir, transcripts_dir

OCR_PROCESSED_NAME = "01_processed_ocr_items.json"
OCR_PROCESSED_CORRECTED_NAME = "corrected_ocr_items.json"
LEGACY_OCR_PROCESSED_CORRECTED_NAME = "ocr_processed_corrected.json"
OCR_SUBTITLE_NAME = "ocr_subtitles.txt"
OCR_SUBTITLE_TIMECODES_NAME = "ocr_subtitles_timecoded.txt"
LEGACY_OCR_SUBTITLE_SUFFIX = "_ocr_subtitle.txt"
LEGACY_OCR_SUBTITLE_TIMECODES_SUFFIX = "_ocr_subtitle_timecodes.txt"
def processed_ocr_path(video_path):
    ocr_dir = existing_ocr_dir(video_path)
    corrected = ocr_dir / OCR_PROCESSED_CORRECTED_NAME
    legacy_corrected = ocr_dir / LEGACY_OCR_PROCESSED_CORRECTED_NAME
    if corrected.exists():
        return corrected
    if legacy_corrected.exists():
        return legacy_corrected
    return ocr_dir / OCR_PROCESSED_NAME


def subtitle_path(video_path, *, transcripts_dir_name=OCR_DIR_NAME):
    transcript_dir = transcripts_dir(video_path, name=transcripts_dir_name)
    preferred = transcript_dir / OCR_SUBTITLE_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_OCR_SUBTITLE_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def subtitle_timecodes_path(video_path, *, transcripts_dir_name=OCR_DIR_NAME):
    transcript_dir = transcripts_dir(video_path, name=transcripts_dir_name)
    preferred = transcript_dir / OCR_SUBTITLE_TIMECODES_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_OCR_SUBTITLE_TIMECODES_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def load_processed_items(path):
    payload = read_json(path)
    items = payload.get("items", [])
    return sorted(items, key=lambda item: (item.get("second", 0), item.get("image", ""), item.get("text", "")))


def analysed_has_subtitles(video_path):
    value = routing_fact(video_path, "has_subtitles")
    return value if isinstance(value, bool) else None


def has_ocr_subtitles(items):
    return any(str(item.get("kind", "")).strip().lower() == "subtitle" for item in items)


def normalize_text(text):
    return "".join(str(text).casefold().split())


def one_edit_apart(left, right):
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False

    if len(left) == len(right):
        differences = sum(1 for left_char, right_char in zip(left, right) if left_char != right_char)
        return differences <= 1

    shorter, longer = sorted((left, right), key=len)
    short_index = 0
    long_index = 0
    differences = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        differences += 1
        if differences > 1:
            return False
        long_index += 1
    return True


def is_duplicate_subtitle(normalized, seen_normalized):
    if normalized in seen_normalized:
        return True
    if len(normalized) < 12:
        return False
    return any(one_edit_apart(normalized, previous) for previous in seen_normalized if len(previous) >= 12)


def format_timecode(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def collect_subtitles(items):
    subtitles = []
    seen_normalized = set()
    for item in items:
        if str(item.get("kind", "")).strip().lower() != "subtitle":
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        cleaned = " ".join(text.split())
        normalized = normalize_text(cleaned)
        if is_duplicate_subtitle(normalized, seen_normalized):
            continue
        seen_normalized.add(normalized)
        subtitles.append(
            {
                "text": cleaned,
                "second": int(item.get("second", 0)),
            }
        )
    return subtitles


def render_subtitles(items):
    return " ".join(item["text"] for item in items)


def render_subtitles_timecodes(items):
    lines = []
    for item in items:
        lines.append(f"[{format_timecode(item['second'])}] {item['text']}")
    return "\n".join(lines)


def extract_for_video(
    video_path,
    *,
    force=False,
    transcripts_dir_name=OCR_DIR_NAME,
):
    if analysed_has_subtitles(video_path) is not True:
        print(
            f"[skip] {video_path.name}: manifest.routing_facts.has_subtitles n'est pas true"
        )
        return None

    source = processed_ocr_path(video_path)
    target_timecodes = subtitle_timecodes_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    if target_timecodes.exists() and not force:
        print(f"[skip] {target_timecodes.name} existe deja")
        return target_timecodes
    if not source.exists():
        print(f"[skip] OCR traite introuvable: {source}")
        return None

    items = load_processed_items(source)
    if not has_ocr_subtitles(items):
        print(f"[skip] aucun kind=subtitle dans {source.name}")
        return None
    subtitles = collect_subtitles(items)
    timecoded_text = render_subtitles_timecodes(subtitles)
    target_timecodes.parent.mkdir(parents=True, exist_ok=True)
    target_timecodes.write_text(
        timecoded_text + ("\n" if timecoded_text else ""),
        encoding="utf-8",
    )
    print(f"[ok] {target_timecodes}")
    return target_timecodes
