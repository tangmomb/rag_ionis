import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOAD_DIR = ROOT_DIR / "downloads" / "youtube"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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


def parse_video_count(value):
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    try:
        count = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Utilise 'all' ou un nombre entier, par exemple 3.") from error
    if count <= 0:
        raise argparse.ArgumentTypeError("Le nombre de videos doit etre superieur a 0.")
    return count


def ask_video_count():
    while True:
        value = input("Nombre de videos a traiter ('all' pour toutes les videos): ").strip()
        try:
            return parse_video_count(value)
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
        type=parse_video_count,
        help="Nombre de videos a tester, ou 'all' pour toutes les videos.",
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
        help="Ne lance pas la Step 14 d'upload S3.",
    )
    parser.add_argument(
        "--skip-sql",
        action="store_true",
        help="Ne lance pas la Step 15 de mise a jour SQL.",
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
        help="Sensibilite de correction des noms propres pendant la Step 08. Defaut: balanced.",
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
    parser.add_argument("--intertitle-dominant-color-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--intertitle-green-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run-upload",
        action="store_true",
        help="Simule l'upload S3 pendant la Step 14.",
    )
    parser.add_argument(
        "--dry-run-sql",
        action="store_true",
        help="Simule la mise a jour SQL pendant la Step 15.",
    )
    return parser.parse_args()


def main():
    load_dotenv(ROOT_DIR / ".env", override=True)
    args = parse_args()
    if args.videos is None:
        args.videos = ask_video_count()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    download_parent = Path(args.download_dir)
    if not download_parent.is_absolute():
        download_parent = ROOT_DIR / download_parent

    if args.videos is None:
        print("Mode pipeline: toutes les videos", flush=True)
    else:
        print(f"Mode pipeline: test sur {args.videos} video(s)", flush=True)

    clean_download_root(download_parent)
    download_parent.mkdir(parents=True, exist_ok=True)
    total_steps = 17
    run_step_numbered(0, total_steps, "Step 00 - Clear SQL Database", utils_command("99_clear_database.py"), env)

    if not args.skip_data:
        step01 = step_command("01_get_data.py", "--skip-transcripts", "--download-dir", download_parent)
        if args.videos is not None:
            step01 += ["--limit", str(args.videos)]
        run_step_numbered(1, total_steps, "Step 01 - Get Data", step01, env)

    step02 = step_command("02_download_videos.py", "--download-dir", download_parent)
    if args.videos is not None:
        step02 += ["--limit", str(args.videos)]
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
    if args.videos is not None:
        step03 += ["--limit", str(args.videos)]
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
    )
    if args.videos is not None:
        step04 += ["--limit-videos", str(args.videos)]
    if args.force:
        step04.append("--force")
    run_step_numbered(4, total_steps, "Step 04 - Classify Images (OpenCV)", step04, env)

    step05 = step_command("05_images_ocr.py", "--video-dir", video_dir)
    if args.videos is not None:
        step05 += ["--limit-videos", str(args.videos)]
    if args.force:
        step05.append("--force")
    run_step_numbered(5, total_steps, "Step 05 - Image OCR (PaddleOCR)", step05, env)

    step06 = step_command("06_ocr_subtitles.py", "--video-dir", video_dir)
    if args.videos is not None:
        step06 += ["--limit-videos", str(args.videos)]
    if args.force:
        step06.append("--force")
    run_step_numbered(6, total_steps, "Step 06 - OCR Subtitles", step06, env)

    step07 = step_command("07_whisper_transcription.py", "--video-dir", video_dir)
    if args.videos is not None:
        step07 += ["--limit", str(args.videos)]
    if args.force:
        step07.append("--force")
    run_step_numbered(7, total_steps, "Step 07 - Whisper Transcription", step07, env)

    step08 = step_command("08_correct_timecodes.py", "--video-dir", video_dir, "--mode", args.correction_mode)
    if args.force:
        step08.append("--force")
    run_step_numbered(8, total_steps, "Step 08 - Correct Timecodes", step08, env)

    step09 = step_command("09_enrich_transcripts.py", "--video-dir", video_dir)
    if args.force:
        step09.append("--force")
    run_step_numbered(9, total_steps, "Step 09 - Enrich Timecodes", step09, env)

    step10 = step_command("09_generate_video_summary.py", "--video-dir", video_dir)
    if args.force:
        step10.append("--force")
    run_step_numbered(10, total_steps, "Step 10 - Video Summary", step10, env)

    step11 = step_command("10_strip_timecodes.py", "--video-dir", video_dir)
    if args.force:
        step11.append("--force")
    run_step_numbered(11, total_steps, "Step 11 - Strip Timecodes", step11, env)

    step12 = step_command("11_create_chunks.py", "--video-dir", video_dir)
    if args.force:
        step12.append("--force")
    run_step_numbered(12, total_steps, "Step 12 - Create Transcript Chunks", step12, env)

    step13 = step_command("12_split_alert_chunks.py", "--video-dir", video_dir)
    if args.force:
        step13.append("--force")
    run_step_numbered(13, total_steps, "Step 13 - Split Alert Chunks", step13, env)

    step14 = step_command("13_create_embeddings.py", "--video-dir", video_dir)
    if args.force:
        step14.append("--force")
    run_step_numbered(14, total_steps, "Step 14 - Create Transcript Embeddings", step14, env)

    if not args.skip_upload:
        step15 = step_command("14_upload_videos_to_s3.py", "--video-dir", video_dir, "--clean-init-prefix")
        if args.force:
            step15.append("--force")
        if args.dry_run_upload:
            step15.append("--dry-run")
        run_step_numbered(15, total_steps, "Step 15 - Upload Videos To S3", step15, env)

    if not args.skip_sql:
        step16 = step_command("15_update_sql_assets.py", "--video-dir", video_dir, "--clean-init-assets")
        if args.dry_run_sql:
            step16.append("--dry-run")
        run_step_numbered(16, total_steps, "Step 16 - Update SQL Assets", step16, env)

    print("\nPipeline termine.", flush=True)
    print(f"Dossier traite: {video_dir}", flush=True)


if __name__ == "__main__":
    main()
