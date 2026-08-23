from __future__ import annotations

from .support.environment import load_project_env
from .update_runs import run_stats_only
from .update_runs import PROJECT_ROOT


def main() -> None:
    load_project_env(PROJECT_ROOT)
    metrics = run_stats_only()
    print(
        "Mise a jour des stats terminee: "
        f"{metrics['videos_updated']} video(s), "
        f"{metrics['stats_snapshots']} snapshot(s), "
        f"{metrics['errors_count']} erreur(s)."
    )


if __name__ == "__main__":
    main()
