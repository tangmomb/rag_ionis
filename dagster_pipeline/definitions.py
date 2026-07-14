import dagster as dg

from dagster_pipeline.assets import (
    ALL_ASSETS,
    ALL_CHECKS,
    PROCESSING_GROUP,
    PUBLICATION_GROUP,
)
from dagster_pipeline.ingestion import importer_videos
from dagster_pipeline.runtime import (
    VIDEO_PARTITIONS,
    PipelineSettings,
    discover_local_videos,
)


@dg.sensor(minimum_interval_seconds=15, default_status=dg.DefaultSensorStatus.RUNNING)
def discover_video_partitions(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    discovered = set(discover_local_videos())
    existing = set(context.instance.get_dynamic_partitions(VIDEO_PARTITIONS.name))
    new_partitions = sorted(discovered - existing)
    if not new_partitions:
        return dg.SensorResult(skip_reason="Aucune nouvelle vidéo locale.")
    return dg.SensorResult(
        dynamic_partitions_requests=[VIDEO_PARTITIONS.build_add_request(new_partitions)],
    )


traiter_video = dg.define_asset_job(
    name="traiter_video",
    selection=dg.AssetSelection.groups(PROCESSING_GROUP),
    partitions_def=VIDEO_PARTITIONS,
    run_tags={"pipeline_job": "traiter_video"},
    executor_def=dg.multiprocess_executor.configured(
        {"max_concurrent": 1},
        name="traiter_video_one_at_a_time",
    ),
    description=(
        "Exécuter toutes les étapes 03 à 24 pour une vidéo, avec une seule étape active à la fois. "
        "La branche has_sub/no_sub est choisie automatiquement après la Step 09."
    ),
)

publier_video = dg.define_asset_job(
    name="publier_video",
    selection=dg.AssetSelection.groups(PUBLICATION_GROUP),
    partitions_def=VIDEO_PARTITIONS,
    description="Publier dans S3 puis mettre à jour PostgreSQL/pgvector pour une vidéo.",
)

pipeline_video_complet = dg.define_asset_job(
    name="pipeline_video_complet",
    selection=dg.AssetSelection.groups(PROCESSING_GROUP, PUBLICATION_GROUP),
    partitions_def=VIDEO_PARTITIONS,
    description="Traiter puis publier une vidéo de bout en bout.",
)


defs = dg.Definitions(
    assets=ALL_ASSETS,
    asset_checks=ALL_CHECKS,
    sensors=[discover_video_partitions],
    jobs=[importer_videos, traiter_video, publier_video, pipeline_video_complet],
    resources={"pipeline_settings": PipelineSettings()},
)
