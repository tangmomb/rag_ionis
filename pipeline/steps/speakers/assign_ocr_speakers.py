import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.support.json_io import read_json, write_json
from pipeline.support.paths import (
    existing_ocr_dir,
    existing_speakers_dir,
    existing_transcripts_dir,
    output_is_current,
    relative_to_video_dir,
)
from pipeline.steps.transcripts.transcribe_with_whisper import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_TRANSCRIBE_DEVICE,
    extract_audio,
    load_diarization_pipeline,
    resolved_device,
)
from pipeline.steps.speakers.correct_speaker_transcripts import (
    clean_name,
    validated_speakers,
)


OCR_TIMECODED_NAME = "ocr_subtitles_timecoded.txt"
SPEAKERS_VALIDATED_NAME = "speakers_validated.json"
DIARIZATION_NAME = "speaker_diarization.json"
OCR_PROCESSED_NAMES = ("corrected_ocr_items.json", "01_processed_ocr_items.json")
TIMECODED_LINE = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2})\]\s*(.*)$")
DIARIZATION_LABEL = re.compile(r"^SPEAKER[_ -]?\d+\s*:\s*", re.IGNORECASE)

def load_json(path):
    return read_json(path)


def parse_timecode(value):
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def normalize_text(value):
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^0-9a-z]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def name_similarity(left, right):
    left_key = normalize_text(left)
    right_key = normalize_text(right)
    if not left_key or not right_key:
        return 0.0
    direct = SequenceMatcher(None, left_key, right_key).ratio()
    token_sorted = SequenceMatcher(
        None,
        " ".join(sorted(left_key.split())),
        " ".join(sorted(right_key.split())),
    ).ratio()
    return max(direct, token_sorted)


def contains_name(text, name):
    text_key = f" {normalize_text(text)} "
    name_key = normalize_text(name)
    return bool(name_key) and f" {name_key} " in text_key


def is_spoken_introduction(text, name):
    text_key = f" {normalize_text(text)} "
    name_key = normalize_text(name)
    if not name_key:
        return False
    return any(
        f" {prefix} {name_key} " in text_key
        for prefix in ("je m appelle", "moi c est", "mon nom est", "je suis")
    )


def transcript_path(video_path, *, transcripts_dir_name=None):
    return (
        existing_transcripts_dir(video_path, name=transcripts_dir_name)
        / OCR_TIMECODED_NAME
    )


def validated_path(video_path):
    return existing_speakers_dir(video_path) / SPEAKERS_VALIDATED_NAME


def diarization_path(video_path):
    return existing_speakers_dir(video_path) / DIARIZATION_NAME


def processed_ocr_path(video_path):
    ocr_dir = existing_ocr_dir(video_path)
    for name in OCR_PROCESSED_NAMES:
        candidate = ocr_dir / name
        if candidate.exists():
            return candidate
    return ocr_dir / OCR_PROCESSED_NAMES[-1]


def assignment_is_current(video_path, *, transcripts_dir_name=None):
    source = transcript_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    validated_source = validated_path(video_path)
    ocr_source = processed_ocr_path(video_path)
    return (
        source.exists()
        and validated_source.exists()
        and output_is_current(
            diarization_path(video_path),
            [source, validated_source, ocr_source],
        )
    )


def load_previous_labels(target):
    if not target.exists():
        return []
    try:
        payload = load_json(target)
    except (OSError, json.JSONDecodeError):
        return []
    labels = []
    for mapping in payload.get("speaker_mapping", []) or []:
        name = clean_name(mapping.get("validated_name", ""))
        if name:
            labels.append(name)
    return labels


def strip_owned_speaker_prefix(text, known_names):
    cleaned = DIARIZATION_LABEL.sub("", str(text), count=1)
    for name in sorted({clean_name(name) for name in known_names if clean_name(name)}, key=len, reverse=True):
        pattern = re.compile(rf"^{re.escape(name)}\s*:\s*", re.IGNORECASE)
        cleaned, count = pattern.subn("", cleaned, count=1)
        if count:
            break
    return cleaned.strip()


def parse_transcript_lines(text, known_names):
    parsed = []
    for raw_line in text.splitlines():
        match = TIMECODED_LINE.match(raw_line)
        if not match:
            parsed.append({"raw": raw_line, "timecode": None, "second": None, "text": raw_line})
            continue
        timecode, body = match.groups()
        parsed.append(
            {
                "raw": raw_line,
                "timecode": timecode,
                "second": parse_timecode(timecode),
                "text": strip_owned_speaker_prefix(body, known_names),
            }
        )
    return parsed


def normalize_diarization_segments(result):
    segments = []
    if hasattr(result, "iterrows"):
        rows = (row for _index, row in result.iterrows())
    elif isinstance(result, (list, tuple)):
        rows = iter(result)
    elif hasattr(result, "itertracks"):
        rows = (
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _track, speaker in result.itertracks(yield_label=True)
        )
    else:
        rows = iter(())

    for row in rows:
        getter = row.get if hasattr(row, "get") else lambda key, default=None: getattr(row, key, default)
        try:
            start = float(getter("start", 0))
            end = float(getter("end", start))
        except (TypeError, ValueError):
            continue
        speaker = clean_name(getter("speaker", getter("label", "")))
        if speaker and end >= start:
            segments.append({"start": start, "end": end, "speaker": speaker})
    return sorted(segments, key=lambda item: (item["start"], item["end"], item["speaker"]))


def speaker_at_time(segments, second):
    if second is None or not segments:
        return None
    window_start = max(0.0, float(second) - 0.75)
    window_end = float(second) + 1.5
    ranked = []
    for segment in segments:
        overlap = max(0.0, min(window_end, segment["end"]) - max(window_start, segment["start"]))
        contains = segment["start"] <= second <= segment["end"]
        distance = (
            0.0
            if contains
            else min(abs(float(second) - segment["start"]), abs(float(second) - segment["end"]))
        )
        ranked.append((overlap, int(contains), -distance, segment["speaker"]))
    overlap, contains, negative_distance, speaker = max(ranked)
    if overlap > 0 or contains or -negative_distance <= 5.0:
        return speaker
    return None


def first_appearance_order(segments):
    first_seen = {}
    for segment in segments:
        first_seen.setdefault(segment["speaker"], segment["start"])
    return sorted(first_seen, key=lambda speaker: (first_seen[speaker], speaker))


def load_ocr_name_evidence(video_path, speakers, segments):
    source = processed_ocr_path(video_path)
    if not source.exists():
        return [], source
    try:
        items = load_json(source).get("items", []) or []
    except (OSError, json.JSONDecodeError):
        return [], source

    evidence = []
    for item in items:
        if str(item.get("kind", "")).strip().lower() != "name":
            continue
        text = clean_name(item.get("text", ""))
        try:
            second = float(item.get("second", 0))
        except (TypeError, ValueError):
            continue
        diarization_speaker = speaker_at_time(segments, second)
        if not text or not diarization_speaker:
            continue
        best_name = max(speakers, key=lambda name: name_similarity(text, name))
        similarity = name_similarity(text, best_name)
        if similarity < 0.78:
            continue
        evidence.append(
            {
                "source": "ocr_name",
                "second": second,
                "text": text,
                "diarization_speaker": diarization_speaker,
                "validated_name": best_name,
                "similarity": round(similarity, 4),
                "weight": 5.0 * similarity,
            }
        )
    return evidence, source


def load_intro_evidence(lines, speakers, segments):
    evidence = []
    for line in lines:
        if line["second"] is None:
            continue
        diarization_speaker = speaker_at_time(segments, line["second"])
        if not diarization_speaker:
            continue
        for name in speakers:
            if not contains_name(line["text"], name) or not is_spoken_introduction(line["text"], name):
                continue
            evidence.append(
                {
                    "source": "spoken_introduction",
                    "second": line["second"],
                    "text": line["text"],
                    "diarization_speaker": diarization_speaker,
                    "validated_name": name,
                    "similarity": 1.0,
                    "weight": 6.0,
                }
            )
    return evidence


def build_speaker_mapping(segments, speakers, evidence):
    diarization_speakers = first_appearance_order(segments)
    scores = defaultdict(float)
    sources = defaultdict(set)
    for item in evidence:
        key = (item["diarization_speaker"], item["validated_name"])
        scores[key] += float(item["weight"])
        sources[key].add(item["source"])

    mapping = {}
    mapping_details = []
    used_names = set()
    ranked = sorted(
        scores,
        key=lambda key: (
            scores[key],
            len(sources[key]),
            -diarization_speakers.index(key[0]) if key[0] in diarization_speakers else 0,
        ),
        reverse=True,
    )
    for diarization_speaker, name in ranked:
        if diarization_speaker in mapping or name in used_names:
            continue
        mapping[diarization_speaker] = name
        used_names.add(name)
        mapping_details.append(
            {
                "diarization_speaker": diarization_speaker,
                "validated_name": name,
                "method": "+".join(sorted(sources[(diarization_speaker, name)])),
                "evidence_score": round(scores[(diarization_speaker, name)], 4),
            }
        )

    remaining_names = [name for name in speakers if name not in used_names]
    remaining_labels = [speaker for speaker in diarization_speakers if speaker not in mapping]
    for diarization_speaker, name in zip(remaining_labels, remaining_names):
        mapping[diarization_speaker] = name
        mapping_details.append(
            {
                "diarization_speaker": diarization_speaker,
                "validated_name": name,
                "method": "first_appearance_fallback",
                "evidence_score": 0.0,
            }
        )
    mapping_details.sort(
        key=lambda item: diarization_speakers.index(item["diarization_speaker"])
        if item["diarization_speaker"] in diarization_speakers
        else 10**9
    )
    return mapping, mapping_details, diarization_speakers


def render_assigned_transcript(lines, segments, mapping, single_speaker=None):
    rendered = []
    assignments = []
    for line in lines:
        if line["timecode"] is None:
            rendered.append(line["raw"])
            continue
        diarization_speaker = None if single_speaker else speaker_at_time(segments, line["second"])
        speaker_name = single_speaker or mapping.get(diarization_speaker) or diarization_speaker
        body = line["text"]
        rendered.append(
            f"[{line['timecode']}] {speaker_name}: {body}" if speaker_name else f"[{line['timecode']}] {body}"
        )
        assignments.append(
            {
                "timecode": line["timecode"],
                "second": line["second"],
                "diarization_speaker": diarization_speaker,
                "speaker_name": speaker_name,
                "text": body,
            }
        )
    return "\n".join(rendered).rstrip() + "\n", assignments


def assign_ocr_speakers(
    video_path,
    force=False,
    diarization_pipeline=None,
    diarization_device=None,
    keep_audio=False,
    min_speakers=None,
    max_speakers=None,
    transcripts_dir_name=None,
):
    source = transcript_path(
        video_path,
        transcripts_dir_name=transcripts_dir_name,
    )
    validated_source = validated_path(video_path)
    target = diarization_path(video_path)
    if not source.exists():
        print(f"[skip] transcript OCR corrige introuvable: {source}")
        return None
    if not validated_source.exists():
        print(f"[skip] speakers valides introuvables: {validated_source}")
        return None

    validated_payload = load_json(validated_source)
    speakers = validated_speakers(validated_payload)
    ocr_source = processed_ocr_path(video_path)
    if target.exists() and not force and output_is_current(target, [source, validated_source, ocr_source]):
        print(f"[skip] {target.name} existe deja")
        return target

    known_names = [*speakers, *load_previous_labels(target)]
    lines = parse_transcript_lines(source.read_text(encoding="utf-8"), known_names)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not speakers:
        rendered, assignments = render_assigned_transcript(
            lines,
            segments=[],
            mapping={},
        )
        source.write_text(rendered, encoding="utf-8")
        payload = {
            "status": "no_validated_speaker",
            "source": relative_to_video_dir(source, video_path),
            "validated_speakers_source": relative_to_video_dir(validated_source, video_path),
            "speakers": [],
            "diarization": {"enabled": False, "reason": "no_validated_speaker", "segments": []},
            "speaker_mapping": [],
            "assignments": assignments,
        }
        write_json(target, payload)
        print(f"[ok] {target} (aucun speaker valide)")
        return target

    if len(speakers) == 1:
        rendered, assignments = render_assigned_transcript(
            lines,
            segments=[],
            mapping={},
            single_speaker=speakers[0],
        )
        source.write_text(rendered, encoding="utf-8")
        payload = {
            "status": "done",
            "source": relative_to_video_dir(source, video_path),
            "validated_speakers_source": relative_to_video_dir(validated_source, video_path),
            "speakers": speakers,
            "diarization": {
                "enabled": False,
                "reason": "single_validated_speaker",
                "segments": [],
            },
            "speaker_mapping": [
                {
                    "diarization_speaker": None,
                    "validated_name": speakers[0],
                    "method": "single_validated_speaker",
                    "evidence_score": None,
                }
            ],
            "assignments": assignments,
        }
        write_json(target, payload)
        print(f"[ok] {target} ({speakers[0]} applique a toutes les lignes)")
        return target

    device = diarization_device or resolved_device(DEFAULT_TRANSCRIBE_DEVICE)
    if diarization_pipeline is None:
        diarization_pipeline, device = load_diarization_pipeline(device)
    requested_min = min_speakers if min_speakers is not None else len(speakers)
    requested_max = max_speakers if max_speakers is not None else len(speakers)

    audio_dir = (
        existing_transcripts_dir(video_path, name=transcripts_dir_name)
        / "audio"
    )
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = extract_audio(video_path, audio_dir)
    try:
        result = diarization_pipeline(
            str(audio_path),
            min_speakers=requested_min,
            max_speakers=requested_max,
        )
        segments = normalize_diarization_segments(result)
        if not segments:
            raise RuntimeError("Pyannote n'a retourne aucun segment de voix.")
    finally:
        if not keep_audio:
            audio_path.unlink(missing_ok=True)
            try:
                audio_dir.rmdir()
            except OSError:
                pass

    intro_evidence = load_intro_evidence(lines, speakers, segments)
    ocr_evidence, ocr_source = load_ocr_name_evidence(video_path, speakers, segments)
    evidence = [*intro_evidence, *ocr_evidence]
    mapping, mapping_details, diarization_speakers = build_speaker_mapping(
        segments,
        speakers,
        evidence,
    )
    rendered, assignments = render_assigned_transcript(lines, segments, mapping)
    source.write_text(rendered, encoding="utf-8")
    payload = {
        "status": "done",
        "source": relative_to_video_dir(source, video_path),
        "validated_speakers_source": relative_to_video_dir(validated_source, video_path),
        "ocr_source": relative_to_video_dir(ocr_source, video_path) if ocr_source.exists() else None,
        "speakers": speakers,
        "diarization": {
            "enabled": True,
            "model": DEFAULT_DIARIZATION_MODEL,
            "device": str(device),
            "min_speakers": requested_min,
            "max_speakers": requested_max,
            "speaker_ids": diarization_speakers,
            "segments": segments,
        },
        "speaker_mapping": mapping_details,
        "evidence": evidence,
        "assignments": assignments,
        "unmatched_validated_speakers": [name for name in speakers if name not in mapping.values()],
        "unmatched_diarization_speakers": [
            speaker for speaker in diarization_speakers if speaker not in mapping
        ],
    }
    write_json(target, payload)
    print(f"[ok] {target} ({len(diarization_speakers)} voix, {len(mapping)} nom(s) associe(s))")
    return target
