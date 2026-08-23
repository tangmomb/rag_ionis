import argparse
import os
import re
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from pipeline.support.environment import load_project_env
from pipeline.support.paths import consolidate_init_dir


DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_S3_ROOT_PREFIX = "youtube"
DEFAULT_S3_BUCKET_NAME = ""
DEFAULT_S3_REGION = ""

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def latest_video_dir(parent_dir):
    candidate = consolidate_init_dir(parent_dir)
    if not candidate.is_dir():
        raise FileNotFoundError(f"Dossier init introuvable dans {parent_dir}")
    return candidate


def s3_client(region):
    access_key = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")

    options = {}
    if region:
        options["region_name"] = region
    if access_key and secret_key:
        options["aws_access_key_id"] = access_key
        options["aws_secret_access_key"] = secret_key

    return boto3.client("s3", **options)


def normalize_prefix(prefix):
    return prefix.strip("/") if prefix else ""


def default_prefix(video_dir):
    return normalize_prefix(f"{DEFAULT_S3_ROOT_PREFIX}/{video_dir.name}")


def s3_key_for(path, root_dir, prefix):
    relative_path = path.relative_to(root_dir).as_posix()
    if prefix:
        return f"{prefix}/{relative_path}"
    return relative_path


def object_exists(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        error_code = error.response.get("Error", {}).get("Code")
        if status_code == 404 or error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def list_init_prefixes(client, bucket, root_prefix):
    root = normalize_prefix(root_prefix)
    prefix = f"{root}/" if root else ""
    paginator = client.get_paginator("list_objects_v2")
    prefixes = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            candidate = item["Prefix"].rstrip("/")
            candidate_name = candidate.rsplit("/", 1)[-1]
            if candidate_name == "init" or re.fullmatch(
                r"\d{8}_\d{4}_init(?:_\d+)?",
                candidate_name,
            ):
                prefixes.add(candidate)

    return sorted(prefixes)


def delete_prefix(client, bucket, prefix, dry_run=False):
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        for index in range(0, len(objects), 1000):
            batch = objects[index : index + 1000]
            if not batch:
                continue
            print(f"[clean] s3://{bucket}/{prefix}/ ({len(batch)} objets)")
            if not dry_run:
                client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
            deleted += len(batch)
    return deleted


def clean_init_prefixes(client, bucket, root_prefix, dry_run=False):
    deleted = 0
    for prefix in list_init_prefixes(client, bucket, root_prefix):
        deleted += delete_prefix(client, bucket, prefix, dry_run=dry_run)
    label = "objets a supprimer" if dry_run else "objets supprimes"
    print(f"[clean] {deleted} {label} dans s3://{bucket}/{normalize_prefix(root_prefix)}/")


def print_step_progress(label, current, total):
    total = max(1, int(total or 0))
    current = min(max(0, int(current or 0)), total)
    percent = int((current / total) * 100)
    print(f"\r[{label}] {percent:3d}% ({current}/{total})", end="", flush=True)
    if current >= total:
        print(flush=True)


def upload_directory(client, bucket, root_dir, prefix="", force=False, dry_run=False, video_ids=None):
    selected_ids = set(video_ids or [])
    files = sorted(path for path in root_dir.rglob("*") if path.is_file())
    if selected_ids:
        files = [
            path
            for path in files
            if path.relative_to(root_dir).parts
            and path.relative_to(root_dir).parts[0] in selected_ids
        ]
    if not files:
        print(f"Aucun fichier a uploader dans {root_dir}")
        return {"uploaded": 0, "skipped": 0}

    uploaded = 0
    skipped = 0
    total_files = len(files)
    for path in files:
        key = s3_key_for(path, root_dir, prefix)
        if not dry_run and not force and object_exists(client, bucket, key):
            skipped += 1
            print_step_progress("upload s3", uploaded + skipped, total_files)
            continue

        if not dry_run:
            client.upload_file(str(path), bucket, key)
        uploaded += 1
        print_step_progress("upload s3", uploaded + skipped, total_files)

    return {"uploaded": uploaded, "skipped": skipped}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload le dossier des videos vers S3 en conservant la meme arborescence."
    )
    parser.add_argument(
        "--video-dir",
        help="Dossier a uploader. Defaut: dernier sous-dossier de downloads/youtube",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Dossier parent utilise si --video-dir est absent. Defaut: downloads/youtube",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_S3_BUCKET_NAME,
        help="Nom du bucket S3. Defaut: aucun",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_S3_REGION,
        help="Region AWS. Defaut: aucune",
    )
    parser.add_argument(
        "--prefix",
        help="Prefixe S3 optionnel. Defaut: youtube/nom_du_dossier_uploade",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Upload directement a la racine du bucket.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ecrase les objets S3 deja presents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les uploads sans envoyer les fichiers.",
    )
    parser.add_argument(
        "--clean-init-prefix",
        action="store_true",
        help="Supprime le prefixe S3 youtube/init et les anciens prefixes dates avant l'upload.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        help="Limite l'upload a cet ID video. Option repetable.",
    )
    return parser.parse_args()


def main():
    load_project_env(ROOT_DIR)
    default_bucket = os.environ.get("S3_BUCKET_NAME", "")
    default_region = os.environ.get("S3_REGION", "")
    args = parse_args()

    if not args.bucket:
        args.bucket = default_bucket
    if not args.region:
        args.region = default_region

    if not args.bucket:
        raise RuntimeError("S3_BUCKET_NAME est requis dans .env ou via --bucket")

    video_dir = Path(args.video_dir) if args.video_dir else latest_video_dir(Path(args.download_dir))
    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable: {video_dir}")

    if args.no_prefix:
        prefix = ""
    else:
        prefix = normalize_prefix(args.prefix) if args.prefix is not None else default_prefix(video_dir)

    print(f"Dossier source: {video_dir}")
    print(f"Bucket cible: s3://{args.bucket}/{prefix}".rstrip("/"))

    client = s3_client(args.region)
    if args.clean_init_prefix and not args.no_prefix:
        clean_init_prefixes(client, args.bucket, DEFAULT_S3_ROOT_PREFIX, dry_run=args.dry_run)

    result = upload_directory(
        client,
        args.bucket,
        video_dir,
        prefix=prefix,
        force=args.force,
        dry_run=args.dry_run,
        video_ids=args.video_id,
    )
    uploaded_label = "fichiers a uploader" if args.dry_run else "fichiers uploades"
    print(f"{result['uploaded']} {uploaded_label}, {result['skipped']} deja presents.")


if __name__ == "__main__":
    main()
