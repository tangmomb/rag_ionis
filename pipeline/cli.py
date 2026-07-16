from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .discovery import select_videos
from .options import PipelineOptions
from .orchestrator import inspect_video, plan_video, run_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "downloads" / "youtube" / "init"


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "selector",
        nargs="?",
        default="all",
        help="'all', nombre, ID video, chemin de video ou dossier video.",
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_VIDEO_ROOT),
        help="Dossier racine contenant un sous-dossier par video.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--openai-mode",
        choices=("normal", "batch"),
        default="normal",
    )
    parser.add_argument(
        "--review-scope",
        choices=("duo", "all"),
        default="duo",
    )
    parser.add_argument("--image-review-model", default=None)
    parser.add_argument("--speaker-validation-model", default=None)
    parser.add_argument(
        "--correction-mode",
        choices=("conservative", "balanced", "aggressive"),
        default="balanced",
    )
    parser.add_argument("--frame-interval", type=float, default=0.5)
    parser.add_argument("--details-per-section", type=int, default=6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Inspecte les videos, produit un manifeste JSON et execute le plan adapte.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect",
        help="Sonde la video puis execute les detecteurs visuels/OCR.",
    )
    add_common_options(inspect_parser)
    inspect_parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Ecrit uniquement les informations techniques du fichier video.",
    )

    plan_parser = commands.add_parser(
        "plan",
        help="Recalcule et affiche le plan sans executer les utilitaires.",
    )
    add_common_options(plan_parser)
    plan_parser.add_argument(
        "--include-inspection",
        action="store_true",
        help="Affiche le plan d'inspection meme si les caracteristiques sont connues.",
    )

    run_parser = commands.add_parser(
        "run",
        help="Inspecte puis execute les utilitaires selectionnes.",
    )
    add_common_options(run_parser)
    run_parser.add_argument(
        "--skip-inspection",
        action="store_true",
        help="Reutilise les caracteristiques deja presentes dans pipeline_analysis.json.",
    )
    return parser.parse_args()


def options_from_args(args: argparse.Namespace) -> PipelineOptions:
    defaults = PipelineOptions()
    return PipelineOptions(
        force=args.force,
        openai_mode=args.openai_mode,
        review_scope=args.review_scope,
        image_review_model=args.image_review_model or defaults.image_review_model,
        speaker_validation_model=(
            args.speaker_validation_model or defaults.speaker_validation_model
        ),
        correction_mode=args.correction_mode,
        frame_interval_seconds=args.frame_interval,
        details_per_section=args.details_per_section,
    )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    args = parse_args()
    options = options_from_args(args)
    videos = select_videos(args.root, args.selector)
    if not videos:
        raise RuntimeError(f"Aucune video trouvee dans {args.root}")

    print(f"{len(videos)} video(s) selectionnee(s)", flush=True)
    for index, video in enumerate(videos, start=1):
        print(f"\n=== VIDEO {index}/{len(videos)}: {video.name} ===", flush=True)
        if args.command == "inspect":
            context = inspect_video(
                video,
                options,
                probe_only=args.probe_only,
                dry_run=args.dry_run,
            )
        elif args.command == "plan":
            context = plan_video(
                video,
                options,
                include_inspection=args.include_inspection,
            )
        else:
            context = run_video(
                video,
                options,
                skip_inspection=args.skip_inspection,
                dry_run=args.dry_run,
            )
        print(
            f"[manifest] {context.manifest_path} "
            f"[route] {context.routing()['pipeline_id']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
