"""Remove generated metadata and outputs while preserving downloaded videos."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOT = Path("downloads/youtube/init")
GENERATED_DIRECTORY_NAMES = ("metadata", "outputs")


def clean_generated_directories(root: Path, *, dry_run: bool = False) -> list[Path]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Dossier init introuvable: {resolved_root}")

    targets: list[Path] = []
    for video_dir in sorted(resolved_root.iterdir()):
        if not video_dir.is_dir():
            continue

        resolved_video_dir = video_dir.resolve()
        if resolved_video_dir.parent != resolved_root:
            raise ValueError(f"Dossier vidéo hors de la racine attendue: {resolved_video_dir}")

        for directory_name in GENERATED_DIRECTORY_NAMES:
            target = (resolved_video_dir / directory_name).resolve()
            if target.parent != resolved_video_dir or target.name != directory_name:
                raise ValueError(f"Refus de supprimer un dossier inattendu: {target}")
            if target.is_dir():
                targets.append(target)

    for target in targets:
        action = "à supprimer" if dry_run else "supprimé"
        print(f"[clean-init] {action}: {target}")
        if not dry_run:
            shutil.rmtree(target)

    mode = "trouvé(s)" if dry_run else "supprimé(s)"
    print(f"[clean-init] {len(targets)} dossier(s) {mode}.")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Supprime les dossiers metadata/ et outputs/ de chaque dossier vidéo "
            "dans downloads/youtube/init."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Racine des dossiers vidéo (défaut: {DEFAULT_ROOT.as_posix()}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Liste les dossiers qui seraient supprimés sans les modifier.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    clean_generated_directories(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
