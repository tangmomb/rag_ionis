import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import psycopg
import yt_dlp
from dotenv import load_dotenv
from imageio_ffmpeg import get_ffmpeg_exe
from yt_dlp.utils import DownloadError


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
BIN_DIR = Path("downloads/bin")
DEFAULT_FORMAT = "best[height<=360]/bestvideo[height<=360]+bestaudio/best"

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


def existing_download(download_dir, youtube_video_id):
    matches = sorted(
        path
        for path in download_dir.glob(f"{youtube_video_id}.*")
        if path.is_file() and path.suffix not in {".part", ".ytdl", ".temp"}
    )
    return matches[0] if matches else None


def fetch_videos(cursor, limit=None):
    query = """
        SELECT id, youtube_video_id, title, url
        FROM videos
        WHERE url IS NOT NULL
        ORDER BY published_at NULLS LAST, id
    """
    params = ()
    if limit is not None:
        query += " LIMIT %s"
        params = (limit,)

    cursor.execute(query, params)
    return cursor.fetchall()


def timestamped_download_dir(parent_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    download_dir = parent_dir / timestamp
    suffix = 2
    while download_dir.exists():
        download_dir = parent_dir / f"{timestamp}_{suffix}"
        suffix += 1
    download_dir.mkdir(parents=True, exist_ok=False)
    return download_dir


def download_video(video, download_dir, force=False):
    db_id, youtube_video_id, title, url = video
    output_template = str(download_dir / "%(id)s.%(ext)s")

    if not force:
        existing = existing_download(download_dir, youtube_video_id)
        if existing:
            print(f"[skip] {youtube_video_id} deja telecharge: {existing}")
            return existing

    options = {
        "format": os.getenv("YTDLP_FORMAT", DEFAULT_FORMAT),
        "merge_output_format": os.getenv("YTDLP_MERGE_FORMAT", "mp4"),
        "outtmpl": output_template,
        "ffmpeg_location": str(ffmpeg_exe()),
        "js_runtimes": {"node": {}},
        "quiet": False,
        "noplaylist": True,
    }

    print(f"[download] #{db_id} {youtube_video_id} - {title}")
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.extract_info(url, download=True)

    downloaded = existing_download(download_dir, youtube_video_id)
    if not downloaded:
        raise FileNotFoundError(f"Video telechargee introuvable pour {youtube_video_id}")
    return downloaded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Telecharge en 360p les videos listees dans la table SQL videos."
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent de sortie des videos. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Nombre maximum de videos a traiter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retelecharge meme si un fichier existe deja pour l'ID YouTube.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Liste les videos trouvees sans telecharger.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    parent_download_dir = Path(args.download_dir)

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            videos = fetch_videos(cursor, args.limit)

    if not videos:
        print("Aucune video trouvee dans la table videos.")
        return

    if args.dry_run:
        for db_id, youtube_video_id, title, url in videos:
            print(f"[dry-run] #{db_id} {youtube_video_id} - {title} - {url}")
        print(f"{len(videos)} videos trouvees.")
        return

    download_dir = timestamped_download_dir(parent_download_dir)
    print(f"Dossier de telechargement: {download_dir}")

    downloaded_count = 0
    failed = []
    for video in videos:
        try:
            download_video(video, download_dir, force=args.force)
            downloaded_count += 1
        except DownloadError as error:
            youtube_video_id = video[1]
            print(f"[error] {youtube_video_id}: {error}")
            failed.append(youtube_video_id)

    print(f"{downloaded_count} videos traitees dans {download_dir}.")
    if failed:
        print(f"{len(failed)} videos en erreur: {', '.join(failed)}")


if __name__ == "__main__":
    main()
