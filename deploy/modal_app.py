"""Fonction GPU Modal : execute une tache ``requires_gpu`` du pipeline.

Le VPS fait tourner le pipeline et n'envoie ici que les taches GPU
(classification des frames, OCR Paddle, WhisperX) avec leurs fichiers d'entree
(voir pipeline/gpu_offload.py). La fonction ne lit ni S3, ni YouTube, ni
OpenAI : elle n'a besoin d'aucun secret Modal.

Modal construit l'image lui-meme a partir des versions de requirements-gpu.txt
et y ajoute le code de pipeline/ a chaque deploiement.

Deploiement :
    modal deploy deploy/modal_app.py

Test du GPU (Torch, Paddle, WhisperX) :
    modal run deploy/modal_app.py

Le modele de diarisation pyannote est lu dans le volume ``rag-ionis-models``,
televerse une fois (voir deploy/MODAL.md) : aucun token Hugging Face ici.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = os.environ.get("MODAL_APP_NAME", "rag-ionis-gpu")
MAX_CONTAINERS = int(os.environ.get("MODAL_MAX_CONTAINERS", "1"))
FRAME_MODEL = "models/frame_filter_2026-07-02_21-30-31.joblib"
CODE_DIR = "/root"

# Same pins as Dockerfile.scaleway. Torch 2.8 and Paddle 3.2 declare
# incompatible NVIDIA versions: Torch's are kept, as on the Scaleway image.
NVIDIA_PACKAGES = (
    "nvidia-cudnn-cu12==9.10.2.21",
    "nvidia-cublas-cu12==12.6.4.1",
    "nvidia-cufft-cu12==11.3.0.4",
    "nvidia-cusolver-cu12==11.7.1.2",
    "nvidia-cusparse-cu12==12.5.4.2",
    "nvidia-cusparselt-cu12==0.7.1",
    "nvidia-nccl-cu12==2.27.3",
    "nvidia-cuda-cupti-cu12==12.6.80",
    "nvidia-cuda-nvrtc-cu12==12.6.77",
    "nvidia-cuda-runtime-cu12==12.6.77",
    "nvidia-cufile-cu12==1.11.1.6",
    "nvidia-curand-cu12==10.3.7.77",
    "nvidia-nvjitlink-cu12==12.6.85",
    "nvidia-nvtx-cu12==12.6.77",
    "nvidia-cuda-cccl-cu12==12.6.77",
    "opt-einsum==3.3.0",
)
SEPARATELY_INSTALLED = (
    "--",
    "paddlepaddle-gpu==",
    "paddleocr==",
    "torch==",
    "torchvision==",
    "torchaudio==",
    "whisperx==",
    "transformers",
)


def core_requirements() -> list[str]:
    """requirements-gpu.txt without the packages installed in their own layer."""
    path = PROJECT_ROOT / "requirements-gpu.txt"
    if not path.exists():  # inside the container: the image is already built
        return []
    lines = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    return [
        line for line in lines if line and not line.startswith(SEPARATELY_INSTALLED)
    ]


image = (
    modal.Image.from_registry(
        # Modal's image builder no longer offers 3.10, the Scaleway image's version.
        "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("ffmpeg")
    .pip_install(*NVIDIA_PACKAGES, extra_options="--no-deps")
    .pip_install(
        "torch==2.8.0+cu126",
        "torchvision==0.23.0+cu126",
        "torchaudio==2.8.0+cu126",
        index_url="https://download.pytorch.org/whl/cu126",
        extra_options="--no-deps",
    )
    # Torch and Paddle are installed without their dependencies; these are
    # the ones they declare for Python 3.12, at their uv.lock versions.
    .pip_install(
        "filelock==4.1.1",
        "fsspec==2026.9.0",
        "jinja2==3.1.6",
        "networkx==3.7",
        "setuptools==84.0.0",
        "sympy==1.14.0",
        "triton==3.4.0",
        "httpx==0.28.1",
        "safetensors==0.8.0",
        "typing-extensions==4.16.0",
    )
    .pip_install(*core_requirements())
    .pip_install("pyannote-database==6.1.1", "whisperx==3.8.6", "transformers==4.57.6")
    .pip_install("paddleocr==3.7.0")
    .pip_install(
        "paddlepaddle-gpu==3.2.0",
        index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        extra_options="--no-deps",
    )
    .env(
        {
            "PYTHONPATH": CODE_DIR,
            "PIPELINE_EXECUTION_BACKEND": "local",
            "PIPELINE_GPU_WORKER": "1",
            "CUDA_MODULE_LOADING": "LAZY",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
            "XDG_CACHE_HOME": "/models/cache",
            "TORCH_HOME": "/models/torch",
            "HF_HOME": "/models/huggingface",
            "PADDLE_HOME": "/models/paddle",
            "PADDLE_PDX_CACHE_HOME": "/models/paddlex",
            "WHISPERX_DEVICE": "cuda",
            "WHISPERX_COMPUTE_TYPE": "float16",
            "WHISPERX_STRICT_CUDA": "1",
            "WHISPERX_BATCH_SIZE": "8",
            "WHISPERX_DIARIZATION_DEVICE": "cuda",
            "WHISPERX_DIARIZATION_CACHE_DIR": "/models/huggingface",
            "WHISPERX_DIARIZATION_MODEL": "/models/pyannote-community-1",
            "WHISPERX_KEEP_MODEL": "1",
            "FRAME_CLASSIFICATION_BATCH_SIZE": "32",
            "FRAME_CLASSIFICATION_DTYPE": "float16",
            "FRAME_CLASSIFICATION_KEEP_MODEL": "1",
            "PADDLEOCR_BATCH_SIZE": "16",
        }
    )
    .workdir(CODE_DIR)
    # The frame classifier is resolved relative to the working directory.
    .add_local_file(PROJECT_ROOT / FRAME_MODEL, f"{CODE_DIR}/{FRAME_MODEL}")
    .add_local_dir(
        PROJECT_ROOT / "pipeline",
        f"{CODE_DIR}/pipeline",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)

app = modal.App(APP_NAME, image=image)
# Whisper, Paddle and Hugging Face caches, plus the pyannote model.
models_volume = modal.Volume.from_name("rag-ionis-models", create_if_missing=True)


@app.function(
    gpu=["L4", "A10"],  # both 24 GB; an L4 alone can wait for capacity
    cpu=4,
    memory=16384,
    timeout=3 * 3600,
    scaledown_window=120,  # stay warm between the GPU tasks of one video
    max_containers=MAX_CONTAINERS,
    retries=1,  # a container reclaimed by Modal mid-task
    volumes={"/models": models_volume},
)
def run_gpu_task(
    task_id: str,
    video_dir_name: str,
    video_name: str,
    archive: bytes,
    task_context: dict,
) -> dict:
    import tempfile

    from pipeline.gpu_offload import run_task_in_directory

    try:
        with tempfile.TemporaryDirectory(prefix="rag-ionis-gpu-") as directory:
            return run_task_in_directory(
                task_id,
                Path(directory) / video_dir_name,
                video_name,
                archive,
                task_context,
            )
    finally:
        # Keep models downloaded by this task for the next containers.
        models_volume.commit()


@app.function(gpu=["L4", "A10"], timeout=900, volumes={"/models": models_volume})
def healthcheck() -> dict:
    import paddle
    import torch
    import whisperx  # noqa: F401

    properties = torch.cuda.get_device_properties(0)
    product = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    product = product @ product
    torch.cuda.synchronize()
    paddle.set_device("gpu:0")
    squares = float((paddle.to_tensor([1.0, 2.0, 3.0]) ** 2).sum().numpy())
    return {
        "gpu": str(properties.name),
        "gpu_memory_gib": round(properties.total_memory / 1024**3, 1),
        "torch": str(torch.__version__),
        "torch_finite": bool(torch.isfinite(product).all().item()),
        "paddle": str(paddle.__version__),
        "paddle_result": squares,
        "pyannote_model_present": Path("/models/pyannote-community-1/config.yaml").exists(),
    }


@app.local_entrypoint()
def main() -> None:
    print(healthcheck.remote())
