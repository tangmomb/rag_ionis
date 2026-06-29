import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def step_command(script_name, *args):
    return [sys.executable, str(ROOT_DIR / "scripts" / "init" / script_name), *map(str, args)]


def utils_command(script_name, *args):
    return [sys.executable, str(ROOT_DIR / "utils" / script_name), *map(str, args)]


def clean_local_init_dirs(download_parent):
    root = download_parent.resolve()
    for path in sorted(download_parent.iterdir()):
        if not path.is_dir() or not path.name.endswith("_init"):
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"Chemin refuse pour suppression: {resolved}")
        print(f"[clean] Suppression ancien init local: {path}")
        shutil.rmtree(path)


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
        help="Ne lance pas la Step 08 d'upload S3.",
    )
    parser.add_argument(
        "--skip-sql",
        action="store_true",
        help="Ne lance pas la Step 09 de mise a jour SQL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force la regeneration quand les steps le supportent.",
    )
    parser.add_argument(
        "--dry-run-upload",
        action="store_true",
        help="Simule l'upload S3 pendant la Step 08.",
    )
    parser.add_argument(
        "--dry-run-sql",
        action="store_true",
        help="Simule la mise a jour SQL pendant la Step 09.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.videos is None:
        args.videos = ask_video_count()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    download_parent = Path(args.download_dir)
    if not download_parent.is_absolute():
        download_parent = ROOT_DIR / download_parent
    download_parent.mkdir(parents=True, exist_ok=True)

    if args.videos is None:
        print("Mode pipeline: toutes les videos")
    else:
        print(f"Mode pipeline: test sur {args.videos} video(s)")

    clean_local_init_dirs(download_parent)
    run_step("Step 00 - Clear SQL Database", utils_command("99_clear_database.py"), env)

    if not args.skip_data:
        step01 = step_command("01_get_data.py", "--skip-transcripts")
        if args.videos is not None:
            step01 += ["--limit", str(args.videos)]
        run_step("Step 01 - Get Data", step01, env)

    step02 = step_command("02_download_videos.py", "--download-dir", download_parent)
    if args.videos is not None:
        step02 += ["--limit", str(args.videos)]
    if args.force:
        step02.append("--force")
    run_step("Step 02 - Download Videos", step02, env)

    try:
        video_dir = latest_video_dir(download_parent)
    except FileNotFoundError as error:
        raise RuntimeError(
            "La Step 02 n'a cree aucun dossier de videos. "
            "Verifie que la table videos contient des lignes, ou relance sans --skip-data."
        ) from error
    print(f"\nDossier pipeline: {video_dir}")

    step03 = step_command("03_transcribe_videos.py", "--video-dir", video_dir)
    if args.videos is not None:
        step03 += ["--limit", str(args.videos)]
    if args.force:
        step03.append("--force")
    run_step("Step 03 - Transcribe Videos", step03, env)

    step05 = step_command("05_extract_images.py", "--video-dir", video_dir)
    if args.videos is not None:
        step05 += ["--limit", str(args.videos)]
    if args.force:
        step05.append("--force")
    run_step("Step 05 - Extract Images", step05, env)

    step06 = step_command("06_images_ocr.py", "--video-dir", video_dir)
    if args.videos is not None:
        step06 += ["--limit-videos", str(args.videos)]
    if args.force:
        step06.append("--force")
    run_step("Step 06 - Analyze Image Text", step06, env)

    step07 = step_command("07_correct_transcripts.py", "--video-dir", video_dir)
    if args.force:
        step07.append("--force")
    run_step("Step 07 - Correct Transcripts", step07, env)

    step08 = step_command("08_enrich_transcripts.py", "--video-dir", video_dir)
    if args.force:
        step08.append("--force")
    run_step("Step 08 - Enrich Transcripts", step08, env)

    step09 = step_command("09_strip_timecodes.py", "--video-dir", video_dir)
    if args.force:
        step09.append("--force")
    run_step("Step 09 - Strip Timecodes", step09, env)

    if not args.skip_upload:
        step10 = step_command("10_upload_videos_to_s3.py", "--video-dir", video_dir, "--clean-init-prefix")
        if args.force:
            step10.append("--force")
        if args.dry_run_upload:
            step10.append("--dry-run")
        run_step("Step 10 - Upload Videos To S3", step10, env)

    if not args.skip_sql:
        step11 = step_command("11_update_sql_assets.py", "--video-dir", video_dir, "--clean-init-assets")
        if args.dry_run_sql:
            step11.append("--dry-run")
        run_step("Step 11 - Update SQL Assets", step11, env)

    print("\nPipeline termine.")
    print(f"Dossier traite: {video_dir}")


if __name__ == "__main__":
    main()
