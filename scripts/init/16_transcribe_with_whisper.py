import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline_analysis import analysed_infos_path, update_analysed_infos
from dotenv import load_dotenv
from imageio_ffmpeg import get_ffmpeg_exe
from pipeline_paths import relative_to_video_dir, transcripts_dir
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
BIN_DIR = Path("downloads/bin")
TRANSCRIPT_WHISPER_DIR_NAME = "transcripts"
WHISPER_TRANSCRIPT_TIMECODED_NAME = "whisper_transcript_timecoded.txt"
LEGACY_TRANSCRIPT_TIMECODED_SUFFIX = "_transcript_timecodes.txt"
OCR_SUBTITLES_TIMECODED_NAME = "ocr_subtitles_timecoded.txt"
LEGACY_OCR_SUBTITLE_TIMECODED_SUFFIX = "_ocr_subtitle_timecodes.txt"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
DEFAULT_TRANSCRIBE_MODEL = os.getenv("WHISPERX_MODEL", "large-v3")
DEFAULT_TRANSCRIBE_LANGUAGE = os.getenv("WHISPERX_LANGUAGE", "fr")
REQUESTED_TRANSCRIBE_DEVICE = os.getenv("WHISPERX_DEVICE", "cuda")
DEFAULT_TRANSCRIBE_DEVICE = REQUESTED_TRANSCRIBE_DEVICE
DEFAULT_TRANSCRIBE_COMPUTE_TYPE = os.getenv(
    "WHISPERX_COMPUTE_TYPE",
    "float16" if DEFAULT_TRANSCRIBE_DEVICE == "cuda" else "int8",
)
DEFAULT_TRANSCRIBE_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
STRICT_CUDA = os.getenv("WHISPERX_STRICT_CUDA", "1").lower() not in {"0", "false", "no"}

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
        path for path in parent_dir.iterdir() if path.is_dir() and any(video_files(path))
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


def transcript_path(transcript_dir, video_path):
    preferred = transcript_dir / WHISPER_TRANSCRIPT_TIMECODED_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_TRANSCRIPT_TIMECODED_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def subtitle_timecodes_path(transcript_dir, video_path):
    preferred = transcript_dir / OCR_SUBTITLES_TIMECODED_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_OCR_SUBTITLE_TIMECODED_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def analysed_has_subtitles(video_path):
    path = analysed_infos_path(video_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("has_subtitles")
    return value if isinstance(value, bool) else None


def extract_audio(video_path, audio_dir):
    audio_path = audio_dir / f"{video_path.stem}.wav"
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
        str(audio_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def load_whisperx_model():
    try:
        import whisperx
    except ImportError as error:
        raise RuntimeError(
            "Le package whisperx est absent. Installe-le avec `pip install whisperx`."
        ) from error

    if DEFAULT_TRANSCRIBE_DEVICE == "cuda" and (torch is None or not torch.cuda.is_available()):
        message = (
            "WHISPERX_DEVICE=cuda est demande, mais le GPU n'est pas accessible dans cette "
            "environnement Python. Verifie l'installation PyTorch CUDA, les pilotes NVIDIA "
            "et la visibilite du GPU dans la .venv."
        )
        if STRICT_CUDA:
            raise RuntimeError(message)
        print(f"[warn] {message} Fallback sur cpu.")
    device = DEFAULT_TRANSCRIBE_DEVICE if DEFAULT_TRANSCRIBE_DEVICE != "cuda" or (torch is not None and torch.cuda.is_available()) else "cpu"
    compute_type = DEFAULT_TRANSCRIBE_COMPUTE_TYPE
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    print(
        f"[whisperx] model={DEFAULT_TRANSCRIBE_MODEL} device={device} "
        f"compute_type={compute_type}"
    )
    model = whisperx.load_model(
        DEFAULT_TRANSCRIBE_MODEL,
        device,
        compute_type=compute_type,
    )
    return whisperx, model


def format_timestamp(seconds):
    seconds = int(seconds or 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_timestamped_transcript(segments):
    lines = []
    for segment in segments:
        start = format_timestamp(segment.get("start", 0))
        end = format_timestamp(segment.get("end", 0))
        text = str(segment.get("text", "")).strip()
        if text:
            lines.append(f"[{start}-{end}] {text}")
    return "\n".join(lines)


def transcribe_with_whisperx(whisperx, model, audio_path):
    result = model.transcribe(
        str(audio_path),
        batch_size=DEFAULT_TRANSCRIBE_BATCH_SIZE,
        language=DEFAULT_TRANSCRIBE_LANGUAGE,
    )
    segments = result.get("segments", [])
    if not segments:
        return ""

    language_code = result.get("language") or DEFAULT_TRANSCRIBE_LANGUAGE
    align_model, metadata = whisperx.load_align_model(language_code=language_code, device=DEFAULT_TRANSCRIBE_DEVICE)
    aligned = whisperx.align(
        segments,
        align_model,
        metadata,
        str(audio_path),
        DEFAULT_TRANSCRIBE_DEVICE,
        return_char_alignments=False,
    )
    return format_timestamped_transcript(aligned.get("segments", segments))


def transcribe_video(whisperx, model, video_path, transcript_dir, audio_dir, force=False):
    output_path = transcript_path(transcript_dir, video_path)
    if output_path.exists() and not force:
        print(f"[skip] {output_path.name} existe deja")
        return output_path

    print(f"[audio] {video_path.name}")
    audio_path = extract_audio(video_path, audio_dir)
    text = transcribe_with_whisperx(whisperx, model, audio_path).strip()
    output_path.write_text(text + "\n", encoding="utf-8")
    update_analysed_infos(
        video_path,
        "whisper_transcription",
        {
            "status": "done",
            "model": DEFAULT_TRANSCRIBE_MODEL,
            "device": DEFAULT_TRANSCRIBE_DEVICE,
            "transcript_timecodes_file": relative_to_video_dir(output_path, video_path),
            "char_count": len(text),
        },
    )
    print(f"[ok] {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcrit localement avec whisperx les videos d'un dossier vers outputs/transcripts."
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
        help="Regenerer les transcriptions meme si les fichiers txt existent deja.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Conserve les fichiers audio extraits dans outputs/transcripts/audio.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()

    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    videos = list(video_files(video_dir))
    if args.limit is not None:
        videos = videos[: args.limit]

    if not videos:
        print(f"Aucune video trouvee dans {video_dir}")
        return

    print(f"Dossier videos: {video_dir}")
    runnable_videos = []
    for video_path in videos:
        has_subtitles = analysed_has_subtitles(video_path)
        if has_subtitles is not False:
            print(f"[skip] {video_path.name}: pipeline_analysis.has_subtitles n'est pas false")
            continue
        runnable_videos.append(video_path)

    if not runnable_videos:
        print("Aucune video a transcrire avec WhisperX.")
        return

    whisperx, model = load_whisperx_model()

    done = 0
    failed = []
    for video_path in runnable_videos:
        try:
            transcript_dir = transcripts_dir(video_path)
            audio_dir = transcript_dir / "audio"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)
            print(f"Dossier transcriptions: {transcript_dir}")
            if transcribe_video(whisperx, model, video_path, transcript_dir, audio_dir, force=args.force):
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
