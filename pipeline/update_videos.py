from __future__ import annotations

from dotenv import load_dotenv

from .update_runs import run_videos_only


def main() -> None:
    load_dotenv(override=True)
    metrics = run_videos_only()
    print(
        "Mise a jour des videos terminee: "
        f"{metrics['new_videos']} nouvelle(s), "
        f"{metrics['pipeline_videos_completed']} pipeline(s) termine(s), "
        f"{metrics['errors_count']} erreur(s)."
    )


if __name__ == "__main__":
    main()
