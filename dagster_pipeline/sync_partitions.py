from __future__ import annotations

import dagster as dg

from dagster_pipeline.runtime import VIDEO_PARTITIONS, discover_local_videos


def main() -> None:
    discovered = sorted(discover_local_videos())
    with dg.DagsterInstance.get() as instance:
        existing = set(instance.get_dynamic_partitions(VIDEO_PARTITIONS.name))
        new_partitions = [video_id for video_id in discovered if video_id not in existing]
        if new_partitions:
            instance.add_dynamic_partitions(VIDEO_PARTITIONS.name, new_partitions)
    print(f"Partitions vidéo: {len(discovered)} détectée(s), {len(new_partitions)} ajoutée(s).")


if __name__ == "__main__":
    main()
