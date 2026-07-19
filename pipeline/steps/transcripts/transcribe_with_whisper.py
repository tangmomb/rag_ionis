import gc
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.steps.transcripts.artifacts import TRANSCRIPT_1_BRUT_NAME

from imageio_ffmpeg import get_ffmpeg_exe
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ROOT_DIR = Path(__file__).resolve().parents[3]
BIN_DIR = Path("downloads/bin")
WHISPER_TRANSCRIPT_TIMECODED_NAME = TRANSCRIPT_1_BRUT_NAME
LEGACY_TRANSCRIPT_TIMECODED_SUFFIX = "_transcript_timecodes.txt"
OCR_SUBTITLES_TIMECODED_NAME = "ocr_subtitles_timecoded.txt"
LEGACY_OCR_SUBTITLE_TIMECODED_SUFFIX = "_ocr_subtitle_timecodes.txt"


def optional_positive_int_env(name):
    value = str(os.getenv(name, "")).strip()
    if not value:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} doit etre un entier positif.")
    return parsed


def refresh_environment_defaults():
    global DEFAULT_TRANSCRIBE_MODEL
    global DEFAULT_TRANSCRIBE_LANGUAGE
    global REQUESTED_TRANSCRIBE_DEVICE
    global DEFAULT_TRANSCRIBE_DEVICE
    global DEFAULT_TRANSCRIBE_COMPUTE_TYPE
    global DEFAULT_TRANSCRIBE_BATCH_SIZE
    global STRICT_CUDA
    global DEFAULT_DIARIZATION_MODEL
    global REQUESTED_DIARIZATION_DEVICE
    global DEFAULT_DIARIZATION_CACHE_DIR
    global DEFAULT_MIN_SPEAKERS
    global DEFAULT_MAX_SPEAKERS

    DEFAULT_TRANSCRIBE_MODEL = os.getenv("WHISPERX_MODEL", "large-v3")
    DEFAULT_TRANSCRIBE_LANGUAGE = os.getenv("WHISPERX_LANGUAGE", "fr")
    REQUESTED_TRANSCRIBE_DEVICE = os.getenv("WHISPERX_DEVICE", "cuda")
    DEFAULT_TRANSCRIBE_DEVICE = REQUESTED_TRANSCRIBE_DEVICE
    DEFAULT_TRANSCRIBE_COMPUTE_TYPE = os.getenv(
        "WHISPERX_COMPUTE_TYPE",
        "float16" if DEFAULT_TRANSCRIBE_DEVICE == "cuda" else "int8",
    )
    DEFAULT_TRANSCRIBE_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
    STRICT_CUDA = os.getenv("WHISPERX_STRICT_CUDA", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    DEFAULT_DIARIZATION_MODEL = os.getenv(
        "WHISPERX_DIARIZATION_MODEL",
        "pyannote/speaker-diarization-community-1",
    )
    REQUESTED_DIARIZATION_DEVICE = os.getenv(
        "WHISPERX_DIARIZATION_DEVICE",
        "",
    ).strip()
    cache_dir = Path(
        os.getenv("WHISPERX_DIARIZATION_CACHE_DIR", "models/huggingface")
    )
    DEFAULT_DIARIZATION_CACHE_DIR = (
        cache_dir if cache_dir.is_absolute() else ROOT_DIR / cache_dir
    )
    DEFAULT_MIN_SPEAKERS = optional_positive_int_env("WHISPERX_MIN_SPEAKERS")
    DEFAULT_MAX_SPEAKERS = optional_positive_int_env("WHISPERX_MAX_SPEAKERS")


refresh_environment_defaults()


def ffmpeg_exe():
    source = Path(get_ffmpeg_exe())
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "ffmpeg.exe"
    if not target.exists():
        shutil.copy2(source, target)
    os.environ["PATH"] = f"{target.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return target


def transcript_path(transcript_dir, video_path):
    return transcript_dir / WHISPER_TRANSCRIPT_TIMECODED_NAME


def subtitle_timecodes_path(transcript_dir, video_path):
    preferred = transcript_dir / OCR_SUBTITLES_TIMECODED_NAME
    legacy = transcript_dir / f"{video_path.stem}{LEGACY_OCR_SUBTITLE_TIMECODED_SUFFIX}"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


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


def resolved_device(requested_device):
    if requested_device == "cuda" and (torch is None or not torch.cuda.is_available()):
        message = (
            "Le device CUDA est demande, mais le GPU n'est pas accessible dans cet "
            "environnement Python. Verifie l'installation PyTorch CUDA, les pilotes NVIDIA "
            "et la visibilite du GPU dans la .venv."
        )
        if STRICT_CUDA:
            raise RuntimeError(message)
        print(f"[warn] {message} Fallback sur cpu.")
        return "cpu"
    return requested_device


def load_whisperx_model():
    try:
        import whisperx
    except ImportError as error:
        raise RuntimeError(
            "Le package whisperx est absent. Installe-le avec `pip install whisperx`."
        ) from error

    device = resolved_device(DEFAULT_TRANSCRIBE_DEVICE)
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
    return whisperx, model, device


def huggingface_token():
    for name in ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = str(os.getenv(name, "")).strip()
        if token:
            return token
    return None


def load_diarization_pipeline(device):
    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError as error:
        raise RuntimeError(
            "La diarisation WhisperX/Pyannote est indisponible. Reinstalle `whisperx`."
        ) from error

    model_path = Path(DEFAULT_DIARIZATION_MODEL)
    token = huggingface_token()
    if not model_path.exists() and not token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN est requis pour le premier telechargement du modele "
            f"{DEFAULT_DIARIZATION_MODEL}. Il n'est plus necessaire avec un chemin local."
        )

    DEFAULT_DIARIZATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diarization_device = resolved_device(REQUESTED_DIARIZATION_DEVICE or device)
    print(
        f"[diarization] model={DEFAULT_DIARIZATION_MODEL} device={diarization_device} "
        f"cache={DEFAULT_DIARIZATION_CACHE_DIR}"
    )
    try:
        pipeline = DiarizationPipeline(
            model_name=DEFAULT_DIARIZATION_MODEL,
            token=token,
            device=diarization_device,
            cache_dir=str(DEFAULT_DIARIZATION_CACHE_DIR),
        )
    except Exception as error:
        try:
            from huggingface_hub.errors import GatedRepoError
        except ImportError:  # pragma: no cover
            GatedRepoError = ()
        if GatedRepoError and isinstance(error, GatedRepoError):
            raise RuntimeError(
                "Acces refuse au modele Pyannote. Le token HUGGINGFACE_TOKEN est present, "
                "mais son compte n'a pas accepte les conditions de "
                "https://huggingface.co/pyannote/speaker-diarization-community-1 "
                "ou le token appartient a un autre compte. Accepte l'acces avec le meme "
                "compte que celui du token, puis relance la Step 16."
            ) from error
        raise
    return pipeline, diarization_device


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
            speaker = str(segment.get("speaker", "")).strip()
            speaker_prefix = f"{speaker}: " if speaker else ""
            lines.append(f"[{start}-{end}] {speaker_prefix}{text}")
    return "\n".join(lines)


def transcribe_with_whisperx(
    whisperx,
    model,
    audio_path,
    device,
    diarization_pipeline=None,
    min_speakers=None,
    max_speakers=None,
):
    result = model.transcribe(
        str(audio_path),
        batch_size=DEFAULT_TRANSCRIBE_BATCH_SIZE,
        language=DEFAULT_TRANSCRIBE_LANGUAGE,
    )
    segments = result.get("segments", [])
    if not segments:
        return "", []

    language_code = result.get("language") or DEFAULT_TRANSCRIBE_LANGUAGE
    align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    aligned = whisperx.align(
        segments,
        align_model,
        metadata,
        str(audio_path),
        device,
        return_char_alignments=False,
    )
    del align_model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if diarization_pipeline is not None:
        diarized_segments = diarization_pipeline(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        aligned = whisperx.assign_word_speakers(diarized_segments, aligned)

    final_segments = aligned.get("segments", segments)
    speakers = sorted(
        {
            str(segment.get("speaker")).strip()
            for segment in final_segments
            if str(segment.get("speaker", "")).strip()
        }
    )
    return format_timestamped_transcript(final_segments), speakers


def transcribe_video(
    whisperx,
    model,
    video_path,
    transcript_dir,
    audio_dir,
    device,
    diarization_pipeline=None,
    diarization_device=None,
    min_speakers=None,
    max_speakers=None,
    force=False,
):
    output_path = transcript_path(transcript_dir, video_path)
    if output_path.exists() and not force:
        print(f"[skip] {output_path.name} existe deja")
        return output_path

    print(f"[audio] {video_path.name}")
    audio_path = extract_audio(video_path, audio_dir)
    text, speakers = transcribe_with_whisperx(
        whisperx,
        model,
        audio_path,
        device,
        diarization_pipeline=diarization_pipeline,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    text = text.strip()
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"[ok] {output_path}")
    return output_path
