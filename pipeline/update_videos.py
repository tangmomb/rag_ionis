from __future__ import annotations

from .support.environment import load_project_env
from .update_runs import run_videos_only
from .update_runs import PROJECT_ROOT


def main() -> None:
    load_project_env(PROJECT_ROOT)
    metrics = run_videos_only()
    print(
        "Mise a jour des videos terminee: "
        f"{metrics['new_videos']} nouvelle(s), "
        f"{metrics['pipeline_videos_completed']} pipeline(s) termine(s), "
        f"{metrics['errors_count']} erreur(s)."
    )


if __name__ == "__main__":
    main()
