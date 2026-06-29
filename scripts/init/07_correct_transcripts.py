import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
TIMECODED_SUFFIX = "_transcript_timecodes.txt"
CORRECTED_SUFFIX = "_transcript_timecodes_corrected.txt"
OCR_SUFFIX = "_ocr.json"
TOKEN_RE = re.compile(r"[0-9A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]+(?:-[0-9A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]+)*")
TIMECODE_RE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})-((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
RELIABLE_CONFIDENCES = {"high", "medium"}
PRIMARY_KINDS = {"name"}
SECONDARY_KINDS = {"other", "subtitle", "title"}
DEFAULT_TIME_PADDING_SECONDS = 3
STOPWORDS = {
    "alors",
    "avec",
    "chez",
    "dans",
    "des",
    "de",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "il",
    "j",
    "je",
    "l",
    "la",
    "le",
    "les",
    "on",
    "pour",
    "qu",
    "que",
    "qui",
    "sur",
    "un",
    "une",
}
PROPER_WORD_EXCLUSIONS = {
    "affaires",
    "analyse",
    "analyst",
    "analyste",
    "business",
    "consultante",
    "fondatrice",
    "responsable",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_timecode(value):
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def format_timecode(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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


def transcript_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{TIMECODED_SUFFIX}"


def corrected_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{CORRECTED_SUFFIX}"


def ocr_path(video_path):
    return video_path.parent / "transcript" / f"{video_path.stem}{OCR_SUFFIX}"


def parse_transcript(path):
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TIMECODE_RE.match(line)
        if not match:
            continue
        start, end, text = match.groups()
        segments.append(
            {
                "start": parse_timecode(start),
                "end": parse_timecode(end),
                "text": text,
            }
        )
    return segments


def load_ocr(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for item in payload.get("items", []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        second = item.get("second")
        if second is None:
            timecode = str(item.get("timecode", "")).strip()
            if not timecode:
                continue
            second = parse_timecode(timecode)
        items.append(
            {
                "second": int(second),
                "text": text,
                "kind": str(item.get("kind", "")).strip().lower(),
                "confidence": str(item.get("confidence", "")).strip().lower(),
            }
        )
    return sorted(items, key=lambda item: item["second"])


def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^0-9A-Za-z]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return TOKEN_RE.findall(text)


def token_spans(text):
    return [
        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "norm": normalize_text(match.group(0)),
        }
        for match in TOKEN_RE.finditer(text)
    ]


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def is_named_candidate(token):
    clean = normalize_text(token)
    if len(clean) < 3 or clean in STOPWORDS:
        return False
    return token[:1].isupper() or token.isupper() or any(char.isdigit() for char in token) or "-" in token


def is_secondary_token_candidate(token):
    clean = normalize_text(token)
    if clean in PROPER_WORD_EXCLUSIONS:
        return False
    return is_named_candidate(token)


def clean_candidate_text(text):
    text = re.sub(r"\s+", " ", str(text).strip())
    return text.strip(" \t\r\n,.;:!?()[]{}\"")


def candidate_key(text):
    return normalize_text(" ".join(tokenize(text)))


def candidate_from_text(text, item, source, priority):
    cleaned = clean_candidate_text(text)
    tokens = tokenize(cleaned)
    if not tokens:
        return None
    return {
        "text": cleaned,
        "tokens": tokens,
        "norm": normalize_text(" ".join(tokens)),
        "second": item["second"],
        "kind": item["kind"],
        "confidence": item["confidence"],
        "source": source,
        "priority": priority,
    }


def named_sequences(tokens):
    sequence = []
    for token in tokens:
        if is_named_candidate(token):
            sequence.append(token)
            continue
        if sequence:
            yield " ".join(sequence)
        sequence = []
    if sequence:
        yield " ".join(sequence)


def ocr_candidates(items):
    candidates = []
    seen = set()
    for item in items:
        if item["confidence"] not in RELIABLE_CONFIDENCES:
            continue
        kind = item["kind"]
        if kind not in PRIMARY_KINDS and kind not in SECONDARY_KINDS:
            continue

        extracted = []
        if kind in PRIMARY_KINDS:
            extracted.append((item["text"], "explicit", 3))
            for sequence in named_sequences(tokenize(item["text"])):
                extracted.append((sequence, "extracted", 2))
        else:
            for token in tokenize(item["text"]):
                if is_secondary_token_candidate(token):
                    extracted.append((token, "extracted", 1))

        for text, source, priority in extracted:
            candidate = candidate_from_text(text, item, source, priority)
            if not candidate:
                continue
            key = (candidate_key(candidate["text"]), candidate["second"])
            if not key[0] or key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def nearby_candidates(segment, candidates, time_padding):
    start = segment["start"] - time_padding
    end = segment["end"] + time_padding
    return [candidate for candidate in candidates if start <= candidate["second"] <= end]


def window_sizes(candidate):
    size = len(candidate["tokens"])
    sizes = {size}
    if size >= 2 and candidate["kind"] == "name":
        sizes.add(size - 1)
    return sorted(size for size in sizes if size > 0)


def single_token_allowed(candidate, window_tokens):
    if len(candidate["tokens"]) != 1:
        return True
    return is_named_candidate(window_tokens[0]["text"])


def required_score(candidate):
    token_count = len(candidate["tokens"])
    if candidate["source"] == "explicit" and candidate["kind"] == "name":
        return 0.76 if token_count >= 2 else 0.78
    if token_count >= 2:
        return 0.80
    return 0.78


def find_replacements(text, candidates):
    tokens = token_spans(text)
    replacements = []
    if not tokens:
        return replacements

    for candidate in candidates:
        if not candidate["norm"]:
            continue
        for size in window_sizes(candidate):
            if size > len(tokens):
                continue
            for index in range(len(tokens) - size + 1):
                window_tokens = tokens[index : index + size]
                if not single_token_allowed(candidate, window_tokens):
                    continue
                start = window_tokens[0]["start"]
                end = window_tokens[-1]["end"]
                original = text[start:end]
                window_norm = normalize_text(" ".join(token["text"] for token in window_tokens))
                if not window_norm or window_norm == candidate["norm"]:
                    continue
                score = similarity(window_norm, candidate["norm"])
                if score < required_score(candidate):
                    continue
                replacements.append(
                    {
                        "start": start,
                        "end": end,
                        "original": original,
                        "replacement": candidate["text"],
                        "score": score,
                        "priority": candidate["priority"],
                        "second": candidate["second"],
                    }
                )
    return replacements


def resolve_replacements(replacements):
    chosen = []
    occupied = []
    ranked = sorted(
        replacements,
        key=lambda item: (
            item["score"],
            item["priority"],
            item["end"] - item["start"],
        ),
        reverse=True,
    )
    for replacement in ranked:
        if any(replacement["start"] < end and replacement["end"] > start for start, end in occupied):
            continue
        chosen.append(replacement)
        occupied.append((replacement["start"], replacement["end"]))
    return sorted(chosen, key=lambda item: item["start"])


def apply_replacements(text, replacements):
    corrected = text
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        corrected = corrected[: replacement["start"]] + replacement["replacement"] + corrected[replacement["end"] :]
    return corrected


def correct_segment(segment, candidates, time_padding=DEFAULT_TIME_PADDING_SECONDS):
    segment_candidates = nearby_candidates(segment, candidates, time_padding)
    replacements = resolve_replacements(find_replacements(segment["text"], segment_candidates))
    return apply_replacements(segment["text"], replacements), replacements


def render_corrected_line(segment, corrected_text):
    return f"[{format_timecode(segment['start'])}-{format_timecode(segment['end'])}] {corrected_text}"


def correct_transcript(video_path, force=False, time_padding=DEFAULT_TIME_PADDING_SECONDS, show_corrections=False):
    source = transcript_path(video_path)
    ocr = ocr_path(video_path)
    target = corrected_path(video_path)

    if target.exists() and not force:
        print(f"[skip] {target.name} existe deja")
        return target
    if not source.exists():
        print(f"[skip] transcript introuvable: {source}")
        return None
    if not ocr.exists():
        print(f"[skip] OCR introuvable: {ocr}")
        return None

    segments = parse_transcript(source)
    candidates = ocr_candidates(load_ocr(ocr))

    corrected_lines = []
    correction_count = 0
    for segment in segments:
        corrected_text, replacements = correct_segment(segment, candidates, time_padding=time_padding)
        correction_count += len(replacements)
        if show_corrections:
            for replacement in replacements:
                print(
                    "[fix] "
                    f"{format_timecode(segment['start'])}-{format_timecode(segment['end'])}: "
                    f"{replacement['original']} -> {replacement['replacement']} "
                    f"(score={replacement['score']:.2f}, ocr={format_timecode(replacement['second'])})"
                )
        corrected_lines.append(render_corrected_line(segment, corrected_text))

    target.write_text("\n".join(corrected_lines).strip() + "\n", encoding="utf-8")
    print(f"[ok] {target} ({correction_count} corrections)")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produit des copies _corrected des transcripts a partir de l'OCR JSON."
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
        "--force",
        action="store_true",
        help="Regenere les transcripts corriges meme s'ils existent deja.",
    )
    parser.add_argument(
        "--time-padding",
        type=float,
        default=DEFAULT_TIME_PADDING_SECONDS,
        help="Marge temporelle autour de chaque segment pour chercher des candidats OCR. Defaut: 3 secondes.",
    )
    parser.add_argument(
        "--show-corrections",
        action="store_true",
        help="Affiche chaque remplacement applique.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    done = 0
    for video_path in videos:
        if correct_transcript(
            video_path,
            force=args.force,
            time_padding=args.time_padding,
            show_corrections=args.show_corrections,
        ):
            done += 1

    print(f"{done} transcripts corriges.")


if __name__ == "__main__":
    main()
