from __future__ import annotations

from dotenv import load_dotenv

from .update_runs import run_stats_only


def main() -> None:
    load_dotenv(override=True)
    metrics = run_stats_only()
    print(
        "Mise a jour des stats terminee: "
        f"{metrics['videos_updated']} video(s), "
        f"{metrics['stats_snapshots']} snapshot(s), "
        f"{metrics['errors_count']} erreur(s)."
    )


if __name__ == "__main__":
    main()
