import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from imageio_ffmpeg import get_ffmpeg_exe
from openai import OpenAI


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
BIN_DIR = Path("downloads/bin")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
OPENAI_UPLOAD_LIMIT_BYTES = 25 * 1024 * 1024
DEFAULT_TRANSCRIBE_PROMPT = (
    "Transcrire strictement l'audio en francais. Ne pas traduire en anglais. "
    "Conserver les noms propres et termes techniques lies a IONIS-STM, job dating, "
    "management, biotechnologies, informatique, energie et double competence."
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def ffmpeg_exe():
    source = Path(get_ffmpeg_exe())
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "ffmpeg.exe"
    if not target.exists():
        shutil.copy2(source, target)
    os.environ["PATH"] = f"{target.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return target


def latest_video_dir(parent_dir):
    candidates = sorted(
        path
        for path in parent_dir.iterdir()
        if path.is_dir() and any(video_files(path))
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


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


def has_timecodes(model):
    return model == "whisper-1" or "diarize" in model


def transcript_path(transcript_dir, video_path, model):
    suffix = "_transcript_timecodes" if has_timecodes(model) else "_transcript"
    return transcript_dir / f"{video_path.stem}{suffix}.txt"


def extract_audio(video_path, audio_dir):
    audio_path = audio_dir / f"{video_path.stem}.mp3"
    if audio_path.exists():
        return audio_path

    command = [
        str(ffmpeg_exe()),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        os.getenv("OPENAI_TRANSCRIBE_AUDIO_BITRATE", "48k"),
        str(audio_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def transcribe_with_openai(client, audio_path):
    if audio_path.stat().st_size > OPENAI_UPLOAD_LIMIT_BYTES:
        size_mb = audio_path.stat().st_size / 1024 / 1024
        raise ValueError(f"{audio_path.name} fait {size_mb:.1f} MiB apres extraction audio")

    model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
    language = os.getenv("OPENAI_TRANSCRIBE_LANGUAGE") or os.getenv("WHISPER_LANGUAGE", "fr") or None
    print(f"[openai] {audio_path.name} ({model})")

    with audio_path.open("rb") as audio_file:
        if "diarize" in model:
            response_format = "diarized_json"
        elif model == "whisper-1":
            response_format = "verbose_json"
        else:
            response_format = "text"
        request = {
            "model": model,
            "file": audio_file,
            "language": language,
            "response_format": response_format,
        }
        prompt = os.getenv("OPENAI_TRANSCRIBE_PROMPT", DEFAULT_TRANSCRIBE_PROMPT)
        if prompt and "diarize" not in model:
            request["prompt"] = prompt
        if "diarize" in model:
            request["chunking_strategy"] = "auto"
        if model == "whisper-1":
            request["timestamp_granularities"] = ["segment"]
        transcript = client.audio.transcriptions.create(**request)

    if isinstance(transcript, str):
        return transcript
    if hasattr(transcript, "segments") and transcript.segments:
        if "diarize" in model:
            return format_diarized_transcript(transcript.segments)
        return format_timestamped_transcript(transcript.segments)
    return transcript.text


def format_timestamp(seconds):
    seconds = int(seconds or 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def segment_value(segment, key, default=None):
    if isinstance(segment, dict):
        return segment.get(key, default)
    return getattr(segment, key, default)


def format_diarized_transcript(segments):
    lines = []
    for segment in segments:
        speaker = segment_value(segment, "speaker", "speaker_unknown")
        start = format_timestamp(segment_value(segment, "start", 0))
        end = format_timestamp(segment_value(segment, "end", 0))
        text = str(segment_value(segment, "text", "")).strip()
        if text:
            lines.append(f"[{start}-{end}] {speaker}: {text}")
    return "\n".join(lines)


def format_timestamped_transcript(segments):
    lines = []
    for segment in segments:
        start = format_timestamp(segment_value(segment, "start", 0))
        end = format_timestamp(segment_value(segment, "end", 0))
        text = str(segment_value(segment, "text", "")).strip()
        if text:
            lines.append(f"[{start}-{end}] {text}")
    return "\n".join(lines)


def transcribe_video(client, video_path, transcript_dir, audio_dir, force=False):
    model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
    output_path = transcript_path(transcript_dir, video_path, model)
    if output_path.exists() and not force:
        print(f"[skip] {output_path.name} existe deja")
        return output_path

    print(f"[audio] {video_path.name}")
    audio_path = extract_audio(video_path, audio_dir)
    text = transcribe_with_openai(client, audio_path).strip()
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"[ok] {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcrit avec l'API OpenAI les videos d'un dossier vers un sous-dossier transcript."
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
        "--limit",
        type=int,
        help="Nombre maximum de videos a transcrire.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere les transcriptions meme si les fichiers txt existent deja.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Conserve les fichiers audio extraits dans transcript/audio.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    client = OpenAI()

    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit is not None:
        videos = videos[: args.limit]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")

    done = 0
    failed = []
    for video_path in videos:
        try:
            transcript_dir = video_path.parent / "transcript"
            audio_dir = transcript_dir / "audio"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)
            print(f"Dossier transcriptions: {transcript_dir}")
            transcribe_video(client, video_path, transcript_dir, audio_dir, force=args.force)
            done += 1
        except Exception as error:
            print(f"[error] {video_path.name}: {error}")
            failed.append(video_path.name)
        finally:
            if not args.keep_audio and "audio_dir" in locals():
                shutil.rmtree(audio_dir, ignore_errors=True)

    print(f"{done} videos transcrites.")
    if failed:
        print(f"{len(failed)} videos en erreur: {', '.join(failed)}")


if __name__ == "__main__":
    main()
