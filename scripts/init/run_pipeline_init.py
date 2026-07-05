import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOAD_DIR = ROOT_DIR / "downloads" / "youtube"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class VideoSelection:
    mode: str
    value: object = None


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
    candidates = sorted(
        path
        for path in parent_dir.iterdir()
        if path.is_dir() and any(video_files(path))
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun dossier de videos trouve dans {parent_dir}")
    return candidates[-1]


def extract_youtube_video_id(value):
    raw_value = value.strip()
    parsed = urlparse(raw_value if "://" in raw_value else f"https://{raw_value}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = parts[1] if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"} else ""
    else:
        video_id = ""

    return video_id if len(video_id) == 11 else ""


def parse_video_selection(value):
    normalized = value.strip().lower()
    if normalized == "all":
        return VideoSelection("all")

    if extract_youtube_video_id(value):
        return VideoSelection("url", value.strip())

    try:
        count = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Utilise 'all', un nombre entier, ou un lien YouTube."
        ) from error
    if count <= 0:
        raise argparse.ArgumentTypeError("Le nombre de videos doit etre superieur a 0.")
    return VideoSelection("count", count)


def ask_video_selection():
    while True:
        value = input("Videos a traiter (nombre, 'all', ou lien YouTube): ").strip()
        try:
            return parse_video_selection(value)
        except argparse.ArgumentTypeError as error:
            print(error)


def run_step(label, command, env):
    printable = " ".join(str(part) for part in command)
    print(f"\n=== {label} ===")
    print(printable)
    subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)


def run_step_numbered(index, total, label, command, env):
    print(f"\n[step {index:02d}/{total:02d}] {label}", flush=True)
    run_step(label, command, env)


def step_command(script_name, *args):
    return [sys.executable, str(ROOT_DIR / "scripts" / "init" / script_name), *map(str, args)]


def utils_command(script_name, *args):
    return [sys.executable, str(ROOT_DIR / "utils" / script_name), *map(str, args)]


def clean_download_root(download_parent):
    root = download_parent.resolve()
    workspace_root = ROOT_DIR.resolve()
    if workspace_root not in root.parents and root != workspace_root:
        raise RuntimeError(f"Chemin refuse pour suppression: {root}")
    if download_parent.exists():
        print(f"[clean] Suppression complete: {download_parent}")
        shutil.rmtree(download_parent)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute le pipeline complet: data, download, transcription, images, S3 et SQL."
    )
    parser.add_argument(
        "videos",
        nargs="?",
        type=parse_video_selection,
        help="Nombre de videos a tester, 'all' pour toutes les videos, ou lien YouTube precis.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent des telechargements. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Ne lance pas la Step 01 de collecte YouTube.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Ne lance pas la Step 18 d'upload S3.",
    )
    parser.add_argument(
        "--skip-sql",
        action="store_true",
        help="Ne lance pas la Step 19 de mise a jour SQL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force la regeneration quand les steps le supportent.",
    )
    parser.add_argument(
        "--correction-mode",
        choices=("aggressive", "balanced", "conservative"),
        default="balanced",
        help="Sensibilite de correction des noms propres pendant la Step 10. Defaut: balanced.",
    )
    parser.add_argument(
        "--chunk-speaker-validation-model",
        default=os.getenv("CHUNK_SPEAKER_VALIDATION_MODEL", "gpt-5.4-nano"),
        help="Modele OpenAI pour valider les speakers des chunks. Defaut: gpt-5.4-nano.",
    )
    parser.add_argument(
        "--image-clusters",
        type=int,
        default=2,
        help="Nombre de clusters k-means pour la classification OpenCV des images. Defaut: 2.",
    )
    parser.add_argument(
        "--image-blur-kernel",
        type=int,
        default=31,
        help="Taille du flou applique avant classification image. Defaut: 31.",
    )
    parser.add_argument(
        "--image-min-cluster-images",
        type=int,
        default=5,
        help="Minimum d'images dans le petit cluster pour accepter un cluster graphic. Defaut: 5.",
    )
    parser.add_argument(
        "--image-min-majority-ratio",
        type=float,
        default=0.70,
        help="Part minimale du plus gros cluster pour accepter un cluster graphic. Defaut: 0.70.",
    )
    parser.add_argument(
        "--image-min-silhouette",
        type=float,
        default=0.12,
        help="Separation minimale des clusters image pour accepter un cluster graphic. Defaut: 0.12.",
    )
    parser.add_argument(
        "--image-graphic-dominant-hue-ratio",
        type=float,
        default=0.70,
        help="Seuil couleur dominante pour rattacher une image au dossier graphic. Defaut: 0.70.",
    )
    parser.add_argument(
        "--image-graphic-max-edge-ratio",
        type=float,
        default=0.002,
        help="Densite maximum de contours apres flou pour l'override graphic. Defaut: 0.002.",
    )
    parser.add_argument(
        "--image-min-flat-region-ratio",
        type=float,
        default=0.08,
        help="Part minimale d'aplat/degrade dans une image pour autoriser k-means. Defaut: 0.08.",
    )
    parser.add_argument(
        "--image-min-flat-component-ratio",
        type=float,
        default=0.03,
        help="Part minimale du plus grand composant plat pour autoriser k-means. Defaut: 0.03.",
    )
    parser.add_argument(
        "--image-min-flat-images",
        type=int,
        default=2,
        help="Nombre minimum d'images consecutives avec aplat/degrade pour autoriser k-means. Defaut: 2.",
    )
    parser.add_argument("--intertitle-dominant-color-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--intertitle-green-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run-upload",
        action="store_true",
        help="Simule l'upload S3 pendant la Step 18.",
    )
    parser.add_argument(
        "--dry-run-sql",
        action="store_true",
        help="Simule la mise a jour SQL pendant la Step 19.",
    )
    return parser.parse_args()


def main():
    load_dotenv(ROOT_DIR / ".env", override=True)
    args = parse_args()
    if args.videos is None:
        args.videos = ask_video_selection()

    video_limit = args.videos.value if args.videos.mode == "count" else None
    video_url = args.videos.value if args.videos.mode == "url" else None

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    download_parent = Path(args.download_dir)
    if not download_parent.is_absolute():
        download_parent = ROOT_DIR / download_parent

    if args.videos.mode == "all":
        print("Mode pipeline: toutes les videos", flush=True)
    elif args.videos.mode == "url":
        print(f"Mode pipeline: test sur la video {video_url}", flush=True)
    else:
        print(f"Mode pipeline: test sur {video_limit} video(s)", flush=True)

    clean_download_root(download_parent)
    download_parent.mkdir(parents=True, exist_ok=True)
    total_steps = 20
    run_step_numbered(0, total_steps, "Step 00 - Clear SQL Database", utils_command("99_clear_database.py"), env)

    if not args.skip_data:
        step01 = step_command("01_get_data.py", "--skip-transcripts", "--download-dir", download_parent)
        if video_url:
            step01 += ["--video-url", video_url]
        elif video_limit is not None:
            step01 += ["--limit", str(video_limit)]
        run_step_numbered(1, total_steps, "Step 01 - Get Data", step01, env)

    step02 = step_command("02_download_videos.py", "--download-dir", download_parent)
    if video_url:
        step02 += ["--video-url", video_url]
    elif video_limit is not None:
        step02 += ["--limit", str(video_limit)]
    if args.force:
        step02.append("--force")
    run_step_numbered(2, total_steps, "Step 02 - Download Videos", step02, env)

    try:
        video_dir = latest_video_dir(download_parent)
    except FileNotFoundError as error:
        raise RuntimeError(
            "La Step 02 n'a cree aucun dossier de videos. "
            "Verifie que la table videos contient des lignes, ou relance sans --skip-data."
        ) from error
    print(f"\nDossier pipeline: {video_dir}", flush=True)

    step03 = step_command("03_extract_images.py", "--video-dir", video_dir)
    if video_limit is not None:
        step03 += ["--limit", str(video_limit)]
    if args.force:
        step03.append("--force")
    run_step_numbered(3, total_steps, "Step 03 - Extract Images", step03, env)

    step04 = step_command(
        "04_classify_images.py",
        "--video-dir",
        video_dir,
        "--clusters",
        args.image_clusters,
        "--blur-kernel",
        args.image_blur_kernel,
        "--min-cluster-images",
        args.image_min_cluster_images,
        "--min-majority-ratio",
        args.image_min_majority_ratio,
        "--min-silhouette",
        args.image_min_silhouette,
        "--graphic-dominant-hue-ratio",
        args.image_graphic_dominant_hue_ratio,
        "--graphic-max-edge-ratio",
        args.image_graphic_max_edge_ratio,
        "--min-flat-region-ratio",
        args.image_min_flat_region_ratio,
        "--min-flat-component-ratio",
        args.image_min_flat_component_ratio,
        "--min-flat-images",
        args.image_min_flat_images,
    )
    if video_limit is not None:
        step04 += ["--limit-videos", str(video_limit)]
    if args.force:
        step04.append("--force")
    run_step_numbered(4, total_steps, "Step 04 - Classify Images (OpenCV)", step04, env)

    step05 = step_command("05_numero_ocr_brut.py", "--video-dir", video_dir)
    if video_limit is not None:
        step05 += ["--limit-videos", str(video_limit)]
    if args.force:
        step05.append("--force")
    run_step_numbered(5, total_steps, "Step 05 - numero_ocr_brut", step05, env)

    step06 = step_command("05a_images_ocr_boxes.py", "--video-dir", video_dir)
    if video_limit is not None:
        step06 += ["--limit-videos", str(video_limit)]
    if args.force:
        step06.append("--force")
    run_step_numbered(6, total_steps, "Step 06 - OCR Boxes", step06, env)

    step07 = step_command("05b_images_ocr_postprocess.py", "--video-dir", video_dir)
    if video_limit is not None:
        step07 += ["--limit-videos", str(video_limit)]
    if args.force:
        step07.append("--force")
    run_step_numbered(7, total_steps, "Step 07 - Build OCR Processed", step07, env)

    step08 = step_command("06_ocr_subtitles.py", "--video-dir", video_dir)
    if video_limit is not None:
        step08 += ["--limit-videos", str(video_limit)]
    if args.force:
        step08.append("--force")
    run_step_numbered(8, total_steps, "Step 08 - OCR Subtitles", step08, env)

    step09 = step_command("07_whisper_transcription.py", "--video-dir", video_dir)
    if video_limit is not None:
        step09 += ["--limit", str(video_limit)]
    if args.force:
        step09.append("--force")
    run_step_numbered(9, total_steps, "Step 09 - Whisper Transcription", step09, env)

    step10 = step_command("08_correct_timecodes.py", "--video-dir", video_dir, "--mode", args.correction_mode)
    if args.force:
        step10.append("--force")
    run_step_numbered(10, total_steps, "Step 10 - Correct Timecodes", step10, env)

    step11 = step_command("09_enrich_transcripts.py", "--video-dir", video_dir)
    if args.force:
        step11.append("--force")
    run_step_numbered(11, total_steps, "Step 11 - Enrich Timecodes", step11, env)

    step12 = step_command("09_generate_video_summary.py", "--video-dir", video_dir)
    if args.force:
        step12.append("--force")
    run_step_numbered(12, total_steps, "Step 12 - Video Summary", step12, env)

    step13 = step_command("10_strip_timecodes.py", "--video-dir", video_dir)
    if args.force:
        step13.append("--force")
    run_step_numbered(13, total_steps, "Step 13 - Strip Timecodes", step13, env)

    step14 = step_command("11_create_chunks.py", "--video-dir", video_dir)
    if args.force:
        step14.append("--force")
    run_step_numbered(14, total_steps, "Step 14 - Create Transcript Chunks", step14, env)

    step15 = step_command(
        "12_validate_chunk_speakers.py",
        "--video-dir",
        video_dir,
        "--model",
        args.chunk_speaker_validation_model,
    )
    if args.force:
        step15.append("--force")
    run_step_numbered(15, total_steps, "Step 15 - Validate Chunk Speakers (OpenAI)", step15, env)

    step16 = step_command("12_split_alert_chunks.py", "--video-dir", video_dir)
    if args.force:
        step16.append("--force")
    run_step_numbered(16, total_steps, "Step 16 - Split Alert Chunks", step16, env)

    step17 = step_command("13_create_embeddings.py", "--video-dir", video_dir)
    if args.force:
        step17.append("--force")
    run_step_numbered(17, total_steps, "Step 17 - Create Transcript Embeddings", step17, env)

    if not args.skip_upload:
        step18 = step_command("14_upload_s3.py", "--video-dir", video_dir, "--clean-init-prefix")
        if args.force:
            step18.append("--force")
        if args.dry_run_upload:
            step18.append("--dry-run")
        run_step_numbered(18, total_steps, "Step 18 - Upload Videos To S3", step18, env)

    if not args.skip_sql:
        step19 = step_command("15_upload_sql.py", "--video-dir", video_dir, "--clean-init-assets")
        if args.dry_run_sql:
            step19.append("--dry-run")
        run_step_numbered(19, total_steps, "Step 19 - Update SQL Assets", step19, env)

    print("\nPipeline termine.", flush=True)
    print(f"Dossier traite: {video_dir}", flush=True)


if __name__ == "__main__":
    main()
