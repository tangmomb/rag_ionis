import argparse
import json
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


def normalize_openai_mode(value):
    normalized = str(value).strip().lower()
    if normalized in {"normal", "live"}:
        return "normal"
    if normalized == "batch":
        return "batch"
    raise ValueError("Mode OpenAI invalide. Utilise 'batch' ou 'normal'.")


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


def analysed_infos_path(video_path):
    video_path = Path(video_path)
    preferred = video_path.parent / "metadata" / "pipeline_analysis.json"
    legacy_candidates = (
        video_path.parent / "metadata" / "analysed_infos.json",
        video_path.parent / f"{video_path.stem}_analysed_infos.json",
        video_path.parent / "analysed_infos.json",
    )
    for legacy in legacy_candidates:
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


def print_post_step09_routing(video_dir, video_limit=None):
    videos = list(video_files(video_dir))
    if video_limit is not None:
        videos = videos[:video_limit]

    has_sub_videos = []
    no_sub_videos = []
    unknown_videos = []
    for video_path in videos:
        has_subtitles = analysed_has_subtitles(video_path)
        if has_subtitles is True:
            has_sub_videos.append(video_path.name)
        elif has_subtitles is False:
            no_sub_videos.append(video_path.name)
        else:
            unknown_videos.append(video_path.name)

    print("\n" + "=" * 72, flush=True)
    print("BRANCHEMENT APRES STEP 09", flush=True)
    print("=" * 72, flush=True)
    print(f"has_sub -> scripts/init/has_sub ({len(has_sub_videos)} video(s))", flush=True)
    print(f"no_sub  -> scripts/init/no_sub  ({len(no_sub_videos)} video(s))", flush=True)
    if unknown_videos:
        print(f"inconnu -> aucune branche claire ({len(unknown_videos)} video(s))", flush=True)

    if 0 < len(has_sub_videos) <= 10:
        print("videos has_sub: " + ", ".join(has_sub_videos), flush=True)
    if 0 < len(no_sub_videos) <= 10:
        print("videos no_sub: " + ", ".join(no_sub_videos), flush=True)
    if 0 < len(unknown_videos) <= 10:
        print("videos inconnues: " + ", ".join(unknown_videos), flush=True)
    print("=" * 72, flush=True)


def selected_branch_after_step09(video_dir, video_limit=None):
    videos = list(video_files(video_dir))
    if video_limit is not None:
        videos = videos[:video_limit]

    statuses = []
    for video_path in videos:
        statuses.append((video_path.name, analysed_has_subtitles(video_path)))

    if not statuses:
        raise RuntimeError("Aucune video selectionnee apres le step 09.")

    values = {status for _, status in statuses}
    if values == {True}:
        return "has_sub"
    if values == {False}:
        return "no_sub"

    details = ", ".join(f"{name}={status}" for name, status in statuses)
    if None in values:
        raise RuntimeError(
            "Impossible de choisir une branche: has_subtitles manquant ou invalide. "
            f"Details: {details}"
        )
    raise RuntimeError(
        "Impossible de choisir une seule branche: les videos selectionnees melangent "
        f"has_subtitles=true et false. Details: {details}"
    )


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


def ask_openai_mode():
    while True:
        value = input("Mode des appels OpenAI ('batch' ou 'normal'): ").strip()
        try:
            return normalize_openai_mode(value)
        except ValueError as error:
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
        help="Ne lance pas la Step 25 d'upload S3.",
    )
    parser.add_argument(
        "--skip-sql",
        action="store_true",
        help="Ne lance pas la Step 26 de mise a jour SQL.",
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
        help="Sensibilite de correction des noms propres pendant la Step 17. Defaut: balanced.",
    )
    parser.add_argument(
        "--chunk-speaker-validation-model",
        default=os.getenv("CHUNK_SPEAKER_VALIDATION_MODEL", "gpt-5.4-nano"),
        help="Modele OpenAI pour valider les speakers des chunks. Defaut: gpt-5.4-nano.",
    )
    parser.add_argument("--intertitle-dominant-color-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--intertitle-green-ratio", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run-upload",
        action="store_true",
        help="Simule l'upload S3 pendant la Step 25.",
    )
    parser.add_argument(
        "--dry-run-sql",
        action="store_true",
        help="Simule la mise a jour SQL pendant la Step 26.",
    )
    parser.add_argument(
        "--openai-mode",
        choices=("batch", "normal"),
        help="Mode global des appels OpenAI pour tout le pipeline.",
    )
    return parser.parse_args()


def main():
    load_dotenv(ROOT_DIR / ".env", override=True)
    args = parse_args()
    if args.videos is None:
        args.videos = ask_video_selection()
    if args.openai_mode is None:
        args.openai_mode = ask_openai_mode()

    video_limit = args.videos.value if args.videos.mode == "count" else None
    video_url = args.videos.value if args.videos.mode == "url" else None

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PIPELINE_OPENAI_MODE"] = args.openai_mode

    download_parent = Path(args.download_dir)
    if not download_parent.is_absolute():
        download_parent = ROOT_DIR / download_parent

    if args.videos.mode == "all":
        print("Mode pipeline: toutes les videos", flush=True)
    elif args.videos.mode == "url":
        print(f"Mode pipeline: test sur la video {video_url}", flush=True)
    else:
        print(f"Mode pipeline: test sur {video_limit} video(s)", flush=True)
    print(f"Mode OpenAI global: {args.openai_mode}", flush=True)

    clean_download_root(download_parent)
    download_parent.mkdir(parents=True, exist_ok=True)
    total_steps = 25
    run_step("Initial cleanup - Clear SQL Database", utils_command("clear_database.py"), env)

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

    step04 = step_command("04_classify_images.py", "--video-dir", video_dir)
    if video_limit is not None:
        step04 += ["--limit-videos", str(video_limit)]
    if args.force:
        step04.append("--force")
    run_step_numbered(4, total_steps, "Step 04 - Classify Images (model)", step04, env)

    step05 = step_command("05_detect_interviews.py", "--video-dir", video_dir)
    if video_limit is not None:
        step05 += ["--limit-videos", str(video_limit)]
    if args.force:
        step05.append("--force")
    run_step_numbered(5, total_steps, "Step 05 - Detect Interviews", step05, env)

    step06 = step_command("06_infer_video_type.py", "--video-dir", video_dir)
    if video_limit is not None:
        step06 += ["--limit-videos", str(video_limit)]
    if args.force:
        step06.append("--force")
    run_step_numbered(6, total_steps, "Step 06 - Infer Video Type", step06, env)

    step07 = step_command("07_extract_raw_ocr.py", "--video-dir", video_dir)
    if video_limit is not None:
        step07 += ["--limit-videos", str(video_limit)]
    if args.force:
        step07.append("--force")
    run_step_numbered(7, total_steps, "Step 07 - Extract Raw OCR", step07, env)

    step08 = step_command("08_extract_ocr_boxes.py", "--video-dir", video_dir)
    if video_limit is not None:
        step08 += ["--limit-videos", str(video_limit)]
    if args.force:
        step08.append("--force")
    run_step_numbered(8, total_steps, "Step 08 - OCR Boxes", step08, env)

    step09 = step_command("09_detect_ocr_subtitles.py", "--video-dir", video_dir)
    if video_limit is not None:
        step09 += ["--limit-videos", str(video_limit)]
    if args.force:
        step09.append("--force")
    run_step_numbered(9, total_steps, "Step 09 - Detect OCR Subtitles", step09, env)
    print_post_step09_routing(video_dir, video_limit=video_limit)
    selected_branch = selected_branch_after_step09(video_dir, video_limit=video_limit)
    print(f"\nPIPELINE ORIENTE VERS: scripts/init/{selected_branch}\n", flush=True)

    branch_dir = Path(selected_branch)

    step10 = step_command(branch_dir / "10_build_processed_ocr.py", "--video-dir", video_dir)
    if video_limit is not None:
        step10 += ["--limit-videos", str(video_limit)]
    if args.force:
        step10.append("--force")
    run_step_numbered(10, total_steps, f"Step 10 - Build Processed OCR ({selected_branch})", step10, env)

    step11 = step_command(branch_dir / "11_filter_processed_ocr.py", "--video-dir", video_dir)
    if args.force:
        step11.append("--force")
    run_step_numbered(11, total_steps, f"Step 11 - Filter Processed OCR ({selected_branch})", step11, env)

    step12 = step_command(branch_dir / "12_extract_other_text_review_candidates.py", "--video-dir", video_dir)
    if args.force:
        step12.append("--force")
    run_step_numbered(12, total_steps, f"Step 12 - Extract Other Text Review Candidates ({selected_branch})", step12, env)

    step13 = step_command(branch_dir / "13_review_other_text_candidates.py", "--video-dir", video_dir)
    if args.force:
        step13.append("--force")
    run_step_numbered(13, total_steps, f"Step 13 - Review Other Text Candidates ({selected_branch})", step13, env)

    step14 = step_command(branch_dir / "14_apply_other_text_review.py", "--video-dir", video_dir)
    if args.force:
        step14.append("--force")
    run_step_numbered(14, total_steps, f"Step 14 - Apply Other Text Review ({selected_branch})", step14, env)

    if selected_branch == "has_sub":
        step15 = step_command(branch_dir / "15_extract_ocr_subtitles.py", "--video-dir", video_dir)
        if video_limit is not None:
            step15 += ["--limit-videos", str(video_limit)]
        if args.force:
            step15.append("--force")
        run_step_numbered(15, total_steps, "Step 15 - OCR Subtitles", step15, env)

        step16 = step_command(branch_dir / "16_correct_ocr_subtitle_spacing.py", "--video-dir", video_dir)
        if video_limit is not None:
            step16 += ["--limit-videos", str(video_limit)]
        if args.force:
            step16.append("--force")
        run_step_numbered(16, total_steps, "Step 16 - Fix OCR Subtitle Spacing", step16, env)

        step17 = step_command(branch_dir / "17_normalize_ionis_stm.py", "--video-dir", video_dir)
        if args.force:
            step17.append("--force")
        run_step_numbered(17, total_steps, "Step 17 - Normalize Ionis-STM", step17, env)

        step18 = step_command(branch_dir / "18_create_plain_transcript.py", "--video-dir", video_dir)
        if args.force:
            step18.append("--force")
        run_step_numbered(18, total_steps, "Step 18 - Create Plain Transcript", step18, env)

        step19 = step_command(branch_dir / "19_enrich_transcripts.py", "--video-dir", video_dir)
        if args.force:
            step19.append("--force")
        run_step_numbered(19, total_steps, "Step 19 - Enrich Timecodes", step19, env)

        step20 = step_command(branch_dir / "20_generate_video_summary.py", "--video-dir", video_dir)
        if args.force:
            step20.append("--force")
        run_step_numbered(20, total_steps, "Step 20 - Video Summary", step20, env)

        step21 = step_command(branch_dir / "21_create_transcript_chunks.py", "--video-dir", video_dir)
        if args.force:
            step21.append("--force")
        run_step_numbered(21, total_steps, "Step 21 - Create Transcript Chunks", step21, env)

        step22 = step_command(
            branch_dir / "22_validate_chunk_speakers.py",
            "--video-dir",
            video_dir,
            "--model",
            args.chunk_speaker_validation_model,
        )
        if args.force:
            step22.append("--force")
        run_step_numbered(22, total_steps, "Step 22 - Validate Chunk Speakers (OpenAI)", step22, env)

        step23_embeddings = step_command(branch_dir / "23_create_chunk_embeddings.py", "--video-dir", video_dir)
        if args.force:
            step23_embeddings.append("--force")
        run_step_numbered(23, total_steps, "Step 23 - Create Chunk Embeddings", step23_embeddings, env)
    else:
        step16 = step_command(branch_dir / "16_transcribe_with_whisper.py", "--video-dir", video_dir)
        if video_limit is not None:
            step16 += ["--limit", str(video_limit)]
        if args.force:
            step16.append("--force")
        run_step_numbered(16, total_steps, "Step 16 - Transcribe With Whisper", step16, env)

        step17 = step_command(branch_dir / "17_correct_transcript_timecodes.py", "--video-dir", video_dir, "--mode", args.correction_mode)
        if args.force:
            step17.append("--force")
        run_step_numbered(17, total_steps, "Step 17 - Correct Transcript Timecodes", step17, env)

        step18 = step_command(branch_dir / "18_enrich_transcripts.py", "--video-dir", video_dir)
        if args.force:
            step18.append("--force")
        run_step_numbered(18, total_steps, "Step 18 - Enrich Timecodes", step18, env)

        step19 = step_command(branch_dir / "19_generate_video_summary.py", "--video-dir", video_dir)
        if args.force:
            step19.append("--force")
        run_step_numbered(19, total_steps, "Step 19 - Video Summary", step19, env)

        step20 = step_command(branch_dir / "20_create_plain_transcript.py", "--video-dir", video_dir)
        if args.force:
            step20.append("--force")
        run_step_numbered(20, total_steps, "Step 20 - Create Plain Transcript", step20, env)

        step21 = step_command(branch_dir / "21_create_transcript_chunks.py", "--video-dir", video_dir)
        if args.force:
            step21.append("--force")
        run_step_numbered(21, total_steps, "Step 21 - Create Transcript Chunks", step21, env)

        step22 = step_command(
            branch_dir / "22_validate_chunk_speakers.py",
            "--video-dir",
            video_dir,
            "--model",
            args.chunk_speaker_validation_model,
        )
        if args.force:
            step22.append("--force")
        run_step_numbered(22, total_steps, "Step 22 - Validate Chunk Speakers (OpenAI)", step22, env)

        step24 = step_command(branch_dir / "24_create_chunk_embeddings.py", "--video-dir", video_dir)
        if args.force:
            step24.append("--force")
        run_step_numbered(23, total_steps, "Step 23 - Create Chunk Embeddings", step24, env)

    if not args.skip_upload:
        step25 = step_command("final_01_upload_outputs_to_s3.py", "--video-dir", video_dir, "--clean-init-prefix")
        if args.force:
            step25.append("--force")
        if args.dry_run_upload:
            step25.append("--dry-run")
        run_step_numbered(24, total_steps, "Step 24 - Upload Outputs To S3", step25, env)

    if not args.skip_sql:
        step26 = step_command("final_02_update_sql_assets.py", "--video-dir", video_dir, "--clean-init-assets")
        if args.dry_run_sql:
            step26.append("--dry-run")
        run_step_numbered(25, total_steps, "Step 25 - Update SQL Assets", step26, env)

    print("\nPipeline termine.", flush=True)
    print(f"Dossier traite: {video_dir}", flush=True)


if __name__ == "__main__":
    main()
