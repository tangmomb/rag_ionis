import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp
from dotenv import load_dotenv
from imageio_ffmpeg import get_ffmpeg_exe
from yt_dlp.utils import DownloadError

from common.pipeline_paths import consolidate_init_dir, youtube_api_infos_path


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
BIN_DIR = Path("downloads/bin")
# Prefer MP4 video + M4A audio.  Without the extension constraints yt-dlp
# commonly selects WebM/Opus audio, which some Windows players cannot decode
# after the streams are merged into an MP4 container.
DEFAULT_FORMAT = "bestvideo[height=720][ext=mp4]+bestaudio[ext=m4a]/best[height=720][ext=mp4]/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best"
DEFAULT_MERGE_FORMAT = "mp4"
YOUTUBE_API_INFOS_SUFFIX = ".youtube_api_infos.json"
LEGACY_INFO_SUFFIX = ".info.json"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}

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


def info_cache_dir(parent_dir):
    return Path(parent_dir) / "info_videos"


def info_candidates(base_dir, youtube_video_id):
    return [
        Path(base_dir) / f"{youtube_video_id}{YOUTUBE_API_INFOS_SUFFIX}",
        Path(base_dir) / f"{youtube_video_id}{LEGACY_INFO_SUFFIX}",
    ]


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

    if len(video_id) != 11:
        raise argparse.ArgumentTypeError("Lien YouTube invalide ou ID video introuvable.")
    return video_id


def load_video_infos(parent_dir, limit=None):
    cache_dir = info_cache_dir(parent_dir)
    if not cache_dir.is_dir():
        return []

    infos = []
    for path in sorted(cache_dir.glob(f"*{YOUTUBE_API_INFOS_SUFFIX}")) + sorted(cache_dir.glob(f"*{LEGACY_INFO_SUFFIX}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"[skip] info json invalide: {path} ({error})")
            continue
        youtube_video_id = payload.get("youtube_video_id") or path.name.removesuffix(YOUTUBE_API_INFOS_SUFFIX).removesuffix(LEGACY_INFO_SUFFIX)
        title = payload.get("title") or youtube_video_id
        url = payload.get("url") or f"https://www.youtube.com/watch?v={youtube_video_id}"
        infos.append((youtube_video_id, title, url, payload))

    deduped_infos = []
    seen_ids = set()
    for info in infos:
        youtube_video_id = info[0]
        if youtube_video_id in seen_ids:
            continue
        seen_ids.add(youtube_video_id)
        deduped_infos.append(info)

    return deduped_infos[:limit] if limit is not None else deduped_infos


def load_video_info_from_url(parent_dir, video_url):
    youtube_video_id = extract_youtube_video_id(video_url)
    cached = [video for video in load_video_infos(parent_dir) if video[0] == youtube_video_id]
    if cached:
        return cached[0]
    return (
        youtube_video_id,
        youtube_video_id,
        video_url,
        {
            "youtube_video_id": youtube_video_id,
            "title": youtube_video_id,
            "url": video_url,
        },
    )


def existing_download(video_dir, youtube_video_id):
    matches = sorted(
        path
        for path in video_dir.glob(f"{youtube_video_id}.*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return matches[0] if matches else None


def previous_download(parent_dir, youtube_video_id, exclude_video_dir=None):
    excluded = Path(exclude_video_dir).resolve() if exclude_video_dir else None
    run_dirs = sorted(
        (path for path in Path(parent_dir).iterdir() if path.is_dir()),
        reverse=True,
    ) if Path(parent_dir).is_dir() else []
    for run_dir in run_dirs:
        video_dir = run_dir / youtube_video_id
        if excluded is not None and video_dir.resolve() == excluded:
            continue
        existing = existing_download(video_dir, youtube_video_id)
        if existing:
            return existing
    return None


def copy_reused_file(source, target):
    source_path = Path(source)
    target_path = Path(target)
    if source_path.suffix.lower() in VIDEO_EXTENSIONS:
        try:
            os.link(source_path, target_path)
            return str(target_path)
        except OSError:
            pass
    return shutil.copy2(source_path, target_path)


def reuse_video_directory(source_video, target_video_dir):
    source_video_dir = Path(source_video).parent
    target_video_dir = Path(target_video_dir)
    print(f"[reuse] {source_video_dir} -> {target_video_dir}")
    shutil.copytree(
        source_video_dir,
        target_video_dir,
        dirs_exist_ok=True,
        copy_function=copy_reused_file,
    )


def copy_video_info(video_dir, youtube_video_id, parent_dir):
    source = next((path for path in info_candidates(info_cache_dir(parent_dir), youtube_video_id) if path.exists()), None)
    if source is None:
        return None
    target = youtube_api_infos_path(video_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def write_video_info(video_dir, youtube_video_id, payload):
    if not payload:
        return None
    target = youtube_api_infos_path(video_dir)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def download_video(video, download_dir, parent_dir, force=False, cookies_from_browser=None):
    youtube_video_id, title, url, payload = video
    video_dir = download_dir / youtube_video_id

    if not force:
        existing = existing_download(video_dir, youtube_video_id)
        if not existing:
            previous = previous_download(
                parent_dir,
                youtube_video_id,
                exclude_video_dir=video_dir,
            )
            if previous:
                reuse_video_directory(previous, video_dir)
                existing = existing_download(video_dir, youtube_video_id)
        if existing:
            print(f"[skip] {youtube_video_id} deja telecharge: {existing}")
            if copy_video_info(video_dir, youtube_video_id, parent_dir) is None:
                write_video_info(video_dir, youtube_video_id, payload)
            return existing

    video_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(video_dir / "%(id)s.%(ext)s")

    options = {
        "format": DEFAULT_FORMAT,
        "merge_output_format": DEFAULT_MERGE_FORMAT,
        "outtmpl": output_template,
        "ffmpeg_location": str(ffmpeg_exe()),
        "js_runtimes": {"node": {}},
        "quiet": False,
        "noplaylist": True,
    }
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)

    print(f"[download] {youtube_video_id} - {title}")
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.extract_info(url, download=True)

    downloaded = existing_download(video_dir, youtube_video_id)
    if not downloaded:
        raise FileNotFoundError(f"Video telechargee introuvable pour {youtube_video_id}")
    if copy_video_info(video_dir, youtube_video_id, parent_dir) is None:
        write_video_info(video_dir, youtube_video_id, payload)
    return downloaded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Telecharge les videos listees dans le cache de metadonnees local."
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent de sortie des videos. Defaut: downloads/youtube",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit",
        type=int,
        help="Nombre maximum de videos a traiter.",
    )
    selection.add_argument(
        "--video-url",
        help="Lien YouTube d'une video precise a telecharger.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retelecharge meme si un fichier existe deja pour l'ID YouTube.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        choices=("brave", "chrome", "chromium", "edge", "firefox", "opera", "vivaldi"),
        help="Navigateur dont yt-dlp utilise les cookies pour acceder a YouTube.",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=15.0,
        help="Delai minimum en secondes entre deux telechargements. Defaut: 15",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=45.0,
        help="Delai maximum en secondes entre deux telechargements. Defaut: 45",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Liste les videos trouvees sans telecharger.",
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)
    args = parse_args()
    if args.min_delay < 0 or args.max_delay < args.min_delay:
        raise ValueError("Les delais doivent respecter 0 <= min-delay <= max-delay.")
    parent_download_dir = Path(args.download_dir)

    videos = (
        [load_video_info_from_url(parent_download_dir, args.video_url)]
        if args.video_url
        else load_video_infos(parent_download_dir, limit=args.limit)
    )
    if not videos:
        print(f"Aucune video trouvee dans {info_cache_dir(parent_download_dir)}.")
        return

    if args.dry_run:
        for youtube_video_id, title, url, _payload in videos:
            print(f"[dry-run] {youtube_video_id} - {title} - {url}")
        print(f"{len(videos)} videos trouvees.")
        return

    download_dir = consolidate_init_dir(parent_download_dir)
    print(f"Dossier de telechargement: {download_dir}")

    downloaded_count = 0
    failed = []
    network_downloads = 0
    for video in videos:
        youtube_video_id = video[0]
        will_download = args.force or existing_download(
            download_dir / youtube_video_id,
            youtube_video_id,
        ) is None
        if will_download:
            if network_downloads:
                delay = random.uniform(args.min_delay, args.max_delay)
                print(f"[wait] pause de {delay:.1f} secondes avant le prochain telechargement")
                time.sleep(delay)
            network_downloads += 1
        try:
            download_video(
                video,
                download_dir,
                parent_download_dir,
                force=args.force,
                cookies_from_browser=args.cookies_from_browser,
            )
            downloaded_count += 1
        except DownloadError as error:
            youtube_video_id = video[0]
            print(f"[error] {youtube_video_id}: {error}")
            failed.append(youtube_video_id)

    print(f"{downloaded_count} videos traitees dans {download_dir}.")
    if failed:
        print(f"{len(failed)} videos en erreur: {', '.join(failed)}")
    if downloaded_count == 0:
        raise RuntimeError(
            "Aucune video n'a pu etre telechargee. Consulte les erreurs yt-dlp ci-dessus "
            "et utilise --cookies-from-browser si YouTube demande une connexion."
        )


if __name__ == "__main__":
    main()
