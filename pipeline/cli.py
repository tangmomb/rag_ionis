from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .catalog import TASKS
from .context import PipelineContext
from .contracts import PlannedTask
from .discovery import select_videos
from .executor import execute_tasks
from .manifest import write_manifest
from .options import PipelineOptions
from .orchestrator import inspect_video, plan_video, run_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "downloads" / "youtube" / "init"


def task_identifier(value: str) -> str:
    task_id = value.strip()
    if task_id not in TASKS:
        raise argparse.ArgumentTypeError(
            f"tache inconnue: {value!r}; utilise `python -m pipeline task --list`."
        )
    return task_id


def print_task_catalog() -> None:
    width = max(len(task_id) for task_id in TASKS)
    for task_id, spec in TASKS.items():
        print(
            f"{task_id:<{width}}  {spec.phase:<10}  {spec.title}",
            flush=True,
        )


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        help="Reutilise les faits de routage deja presents dans video_manifest.json.",
    )

    task_parser = commands.add_parser(
        "task",
        help="Execute une tache du catalogue sur les videos selectionnees.",
    )
    task_parser.add_argument(
        "task_id",
        nargs="?",
        type=task_identifier,
        metavar="TASK_ID",
        help="Identifiant exact de la tache a executer.",
    )
    add_common_options(task_parser)
    task_parser.add_argument(
        "--list",
        dest="list_tasks",
        action="store_true",
        help="Affiche les identifiants de taches disponibles puis quitte.",
    )

    args = parser.parse_args(argv)
    if (
        args.command == "task"
        and not args.list_tasks
        and args.task_id is None
    ):
        task_parser.error(
            "TASK_ID est requis (ou utilise --list pour afficher le catalogue)."
        )
    return args


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
    if args.command == "task" and args.list_tasks:
        print_task_catalog()
        return

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
        elif args.command == "task":
            context = PipelineContext.inspect(video, options)
            context.set_plan(
                [PlannedTask(args.task_id, "manual_cli")]
            )
            write_manifest(context)
            execute_tasks(context, dry_run=args.dry_run)
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
