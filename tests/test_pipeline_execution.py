from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.catalog import TASKS, TaskSpec
from pipeline.context import PipelineContext
from pipeline.contracts import (
    PlannedTask,
    RoutingFacts,
    RunStatus,
    TaskExecution,
    TaskResult,
    TaskStatus,
    VideoType,
)
from pipeline.executor import PipelineBlockedError, execute_tasks
from pipeline.manifest import (
    ManifestValidationError,
    SCHEMA_VERSION,
    build_manifest,
    read_manifest,
    write_manifest,
)
from pipeline.options import PipelineOptions


class PipelineExecutionTests(unittest.TestCase):
    def make_context(
        self,
        root: Path,
        *,
        options: PipelineOptions | None = None,
        routing_ready: bool = True,
    ) -> PipelineContext:
        video_dir = root / "abcdefghijk"
        metadata_dir = video_dir / "metadata"
        metadata_dir.mkdir(parents=True)
        video = video_dir / "abcdefghijk.mp4"
        video.touch()
        facts = (
            {
                "has_subtitles": False,
                "video_type": "interview",
            }
            if routing_ready
            else {}
        )
        (metadata_dir / "youtube_video_metadata.json").write_text(
            json.dumps({"duration_seconds": 180}),
            encoding="utf-8",
        )
        context = PipelineContext(
            video_path=video,
            options=options or PipelineOptions(),
            media={
                "path": video.resolve().as_posix(),
                "filename": video.name,
                "duration_seconds": 180.0,
                "width": 1280,
                "height": 720,
                "fps": 25.0,
                "video_codec": "h264",
                "audio_codec": "aac",
                "has_audio": True,
            },
            routing_facts=RoutingFacts.from_dict(facts),
        )
        return context

    def set_test_plan(
        self,
        context: PipelineContext,
        *specs: TaskSpec,
    ) -> None:
        context.set_plan(
            PlannedTask(spec.id, "test")
            for spec in specs
        )

    @staticmethod
    def mark_completed_checkpoint(
        context: PipelineContext,
        task_id: str,
    ) -> None:
        context.execution.start_pipeline()
        context.execution.start_task(task_id)
        context.execution.complete_task(task_id)
        context.execution.complete_pipeline()

    def test_task_result_has_explicit_terminal_statuses(self) -> None:
        succeeded = TaskResult.succeeded("value", artifacts=["one.txt"])
        cached = TaskResult.cached(reason="cache valide")
        blocked = TaskResult.blocked("prerequis absent")

        self.assertIs(succeeded.status, TaskStatus.SUCCEEDED)
        self.assertEqual(succeeded.artifacts, ("one.txt",))
        self.assertIs(cached.status, TaskStatus.CACHED)
        self.assertIs(blocked.status, TaskStatus.BLOCKED)
        with self.assertRaisesRegex(ValueError, "statut terminal"):
            TaskResult(TaskStatus.RUNNING)

    def test_new_attempt_clears_the_previous_artifact_fingerprint(self) -> None:
        execution = TaskExecution(
            status=TaskStatus.SUCCEEDED,
            artifact_fingerprint="previous",
        )

        execution.start()
        self.assertIsNone(execution.artifact_fingerprint)

        execution.finish(TaskStatus.SUCCEEDED)
        self.assertIsNone(execution.artifact_fingerprint)

    def test_executor_applies_results_and_checkpoints_each_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            artifact = context.outputs_dir / "test.txt"

            def handler(pipeline_context: PipelineContext) -> TaskResult:
                self.assertIs(pipeline_context, context)
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text("ok", encoding="utf-8")
                return TaskResult.succeeded(artifacts=[artifact])

            spec = TaskSpec(
                "test.handler",
                "processing",
                "Test handler",
                handler,
            )
            checkpoints: list[tuple[str, str]] = []

            def capture(current: PipelineContext) -> Path:
                task = current.execution.tasks[spec.id]
                checkpoints.append(
                    (
                        current.execution.status.value,
                        task.status.value,
                    )
                )
                return current.manifest_path

            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                with patch(
                    "pipeline.executor.write_manifest",
                    side_effect=capture,
                ):
                    execute_tasks(context)

            self.assertEqual(
                checkpoints,
                [
                    ("running", "not_started"),
                    ("running", "running"),
                    ("running", "completed"),
                    ("completed", "completed"),
                ],
            )
            self.assertEqual(
                context.artifacts.by_task[spec.id],
                ["outputs/test.txt"],
            )

    def test_blocked_result_stops_dependent_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            downstream = Mock(return_value=TaskResult.succeeded())
            blocked_spec = TaskSpec(
                "test.blocked",
                "processing",
                "Blocked",
                lambda _context: TaskResult.blocked("source absente"),
            )
            downstream_spec = TaskSpec(
                "test.downstream",
                "processing",
                "Downstream",
                lambda pipeline_context: downstream(pipeline_context),
            )
            with patch.dict(
                TASKS,
                {
                    blocked_spec.id: blocked_spec,
                    downstream_spec.id: downstream_spec,
                },
            ):
                self.set_test_plan(
                    context,
                    blocked_spec,
                    downstream_spec,
                )
                with self.assertRaisesRegex(
                    PipelineBlockedError,
                    "source absente",
                ):
                    execute_tasks(context)

            downstream.assert_not_called()
            self.assertIs(context.execution.status, RunStatus.BLOCKED)
            self.assertIs(
                context.execution.tasks[blocked_spec.id].status,
                TaskStatus.BLOCKED,
            )
            self.assertIs(
                context.execution.tasks[downstream_spec.id].status,
                TaskStatus.SKIPPED,
            )

    def test_handler_contract_violation_is_a_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            spec = TaskSpec(
                "test.no_output",
                "processing",
                "No output",
                lambda _context: None,
            )
            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                with self.assertRaisesRegex(
                    TypeError,
                    "doit retourner TaskResult",
                ):
                    execute_tasks(context)

        self.assertIs(context.execution.status, RunStatus.FAILED)
        self.assertIs(
            context.execution.tasks[spec.id].status,
            TaskStatus.FAILED,
        )

    def test_failure_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))

            def failing_handler(pipeline_context: PipelineContext) -> TaskResult:
                raise RuntimeError(f"boom:{pipeline_context.video_id}")

            spec = TaskSpec(
                "test.failure",
                "processing",
                "Test failure",
                failing_handler,
            )
            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "boom:abcdefghijk",
                ):
                    execute_tasks(context)

            manifest = json.loads(
                context.manifest_path.read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["execution"]["status"], "failed")
        failure = manifest["execution"]["tasks"][spec.id]
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["error"], "boom:abcdefghijk")

    def test_valid_checkpoint_is_resumed_without_calling_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            context = self.make_context(root)
            artifact = context.outputs_dir / "resume.txt"
            handler = Mock()

            def create_artifact(
                pipeline_context: PipelineContext,
            ) -> TaskResult:
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text("ok", encoding="utf-8")
                handler(pipeline_context)
                return TaskResult.succeeded(artifacts=[artifact])

            spec = TaskSpec(
                "test.resume",
                "processing",
                "Resume",
                create_artifact,
            )
            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                execute_tasks(context)
                media = dict(context.media)
                with patch(
                    "pipeline.context.probe_video",
                    return_value=media,
                ):
                    resumed = PipelineContext.inspect(
                        context.video_path,
                        context.options,
                    )
                execute_tasks(resumed)

            self.assertEqual(handler.call_count, 1)
            self.assertIs(
                resumed.execution.tasks[spec.id].status,
                TaskStatus.CACHED,
            )
            self.assertEqual(
                resumed.execution.tasks[spec.id].attempts,
                1,
            )

    def test_missing_checkpoint_artifact_reexecutes_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            context = self.make_context(root)
            artifact = context.outputs_dir / "rebuild.txt"
            calls = 0

            def create_artifact(
                _pipeline_context: PipelineContext,
            ) -> TaskResult:
                nonlocal calls
                calls += 1
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(str(calls), encoding="utf-8")
                return TaskResult.succeeded(artifacts=[artifact])

            spec = TaskSpec(
                "test.rebuild",
                "processing",
                "Rebuild",
                create_artifact,
            )
            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                execute_tasks(context)
                artifact.unlink()
                media = dict(context.media)
                with patch(
                    "pipeline.context.probe_video",
                    return_value=media,
                ):
                    resumed = PipelineContext.inspect(
                        context.video_path,
                        context.options,
                    )
                execute_tasks(resumed)

            self.assertEqual(calls, 2)
            self.assertEqual(artifact.read_text(encoding="utf-8"), "2")
            self.assertEqual(
                resumed.execution.tasks[spec.id].attempts,
                2,
            )

    def test_modified_checkpoint_artifact_reexecutes_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            artifact = context.outputs_dir / "fingerprinted.txt"
            calls = 0

            def create_artifact(
                _context: PipelineContext,
            ) -> TaskResult:
                nonlocal calls
                calls += 1
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(f"generated-{calls}", encoding="utf-8")
                return TaskResult.succeeded(artifacts=[artifact])

            spec = TaskSpec(
                "test.fingerprinted",
                "processing",
                "Fingerprint",
                create_artifact,
            )
            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                execute_tasks(context)
                artifact.write_text("overwritten", encoding="utf-8")
                execute_tasks(context)

            self.assertEqual(calls, 2)
            self.assertEqual(
                artifact.read_text(encoding="utf-8"),
                "generated-2",
            )
            self.assertEqual(
                context.execution.tasks[spec.id].attempts,
                2,
            )

    def test_false_postcondition_cannot_be_bypassed_by_an_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            artifact = context.outputs_dir / "stale.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("stale", encoding="utf-8")

            def handler(_context: PipelineContext) -> TaskResult:
                return TaskResult.succeeded(artifacts=[artifact])

            spec = TaskSpec(
                "test.strict_postcondition",
                "processing",
                "Strict postcondition",
                handler,
                postcondition=lambda _context: False,
            )
            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                context.record_artifacts(spec.id, artifact)
                self.mark_completed_checkpoint(context, spec.id)

                with self.assertRaises(PipelineBlockedError):
                    execute_tasks(context)

            self.assertIs(
                context.execution.tasks[spec.id].status,
                TaskStatus.BLOCKED,
            )
            self.assertEqual(
                context.execution.tasks[spec.id].attempts,
                2,
            )

    def test_empty_artifact_directory_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            artifact_dir = context.outputs_dir / "empty-artifact"
            artifact_dir.mkdir(parents=True)
            handler = Mock()

            def rebuild(_context: PipelineContext) -> TaskResult:
                handler()
                output = artifact_dir / "rebuilt.txt"
                output.write_text("rebuilt", encoding="utf-8")
                return TaskResult.succeeded(artifacts=[artifact_dir])

            spec = TaskSpec(
                "test.empty_directory",
                "processing",
                "Empty directory",
                rebuild,
            )
            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                context.record_artifacts(spec.id, artifact_dir)
                self.mark_completed_checkpoint(context, spec.id)
                execute_tasks(context)

            handler.assert_called_once_with()
            self.assertTrue((artifact_dir / "rebuilt.txt").exists())
            self.assertIs(
                context.execution.tasks[spec.id].status,
                TaskStatus.SUCCEEDED,
            )
            self.assertEqual(
                context.execution.tasks[spec.id].attempts,
                2,
            )

    def test_replayed_upstream_task_invalidates_downstream_checkpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            calls = {"upstream": 0, "downstream": 0}
            upstream_artifact = context.outputs_dir / "upstream.txt"
            downstream_artifact = context.outputs_dir / "downstream.txt"

            def upstream(_context: PipelineContext) -> TaskResult:
                calls["upstream"] += 1
                upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
                upstream_artifact.write_text(
                    str(calls["upstream"]),
                    encoding="utf-8",
                )
                return TaskResult.succeeded(artifacts=[upstream_artifact])

            def downstream(_context: PipelineContext) -> TaskResult:
                calls["downstream"] += 1
                downstream_artifact.write_text(
                    str(calls["downstream"]),
                    encoding="utf-8",
                )
                return TaskResult.succeeded(artifacts=[downstream_artifact])

            upstream_spec = TaskSpec(
                "test.upstream",
                "processing",
                "Upstream",
                upstream,
            )
            downstream_spec = TaskSpec(
                "test.downstream_checkpoint",
                "processing",
                "Downstream checkpoint",
                downstream,
            )
            with patch.dict(
                TASKS,
                {
                    upstream_spec.id: upstream_spec,
                    downstream_spec.id: downstream_spec,
                },
            ):
                self.set_test_plan(
                    context,
                    upstream_spec,
                    downstream_spec,
                )
                execute_tasks(context)
                upstream_artifact.unlink()
                execute_tasks(context)

            self.assertEqual(calls, {"upstream": 2, "downstream": 2})
            self.assertEqual(
                downstream_artifact.read_text(encoding="utf-8"),
                "2",
            )
            self.assertEqual(
                context.execution.tasks[downstream_spec.id].attempts,
                2,
            )

    def test_unknown_task_is_rejected_before_execution_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            context.plan = [PlannedTask("test.unknown", "test")]

            with self.assertRaisesRegex(ValueError, "test.unknown"):
                execute_tasks(context)

            self.assertIs(
                context.execution.status,
                RunStatus.NOT_STARTED,
            )
            self.assertFalse(context.manifest_path.exists())

    def test_plan_change_creates_run_id_and_archives_previous_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            first = TASKS["frames.extract"]
            second = TASKS["frames.classify"]
            context.set_plan([PlannedTask(first.id, "first")])
            context.execution.start_pipeline()
            context.execution.complete_task(first.id)
            first_id = context.execution.run_id
            first_hash = context.execution.plan_hash

            context.set_plan([PlannedTask(second.id, "second")])

            self.assertNotEqual(context.execution.run_id, first_id)
            self.assertNotEqual(context.execution.plan_hash, first_hash)
            self.assertEqual(
                [item.run_id for item in context.execution_history],
                [first_id],
            )
            manifest = build_manifest(context)
            self.assertEqual(
                manifest["plan"]["hash"],
                context.execution.plan_hash,
            )
            self.assertEqual(
                manifest["execution_history"][0]["run_id"],
                first_id,
            )
            self.assertEqual(
                set(manifest["artifacts"]),
                {"by_task"},
            )

    def test_video_size_and_mtime_are_part_of_the_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            task = PlannedTask("frames.extract", "source-fingerprint")
            context.set_plan([task])
            initial_hash = context.execution.plan_hash

            context.video_path.write_bytes(b"new-video-content")
            size_changed_hash = context.compute_plan_hash([task])

            stat = context.video_path.stat()
            os.utime(
                context.video_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            mtime_changed_hash = context.compute_plan_hash([task])

        self.assertNotEqual(initial_hash, size_changed_hash)
        self.assertNotEqual(size_changed_hash, mtime_changed_hash)

    def test_switching_plans_restores_the_matching_run_from_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            inspection_spec = TaskSpec(
                "test.inspection_run",
                "inspection",
                "Inspection run",
                lambda _context: TaskResult.succeeded(),
            )
            processing_spec = TaskSpec(
                "test.processing_run",
                "processing",
                "Processing run",
                lambda _context: TaskResult.succeeded(),
            )
            with patch.dict(
                TASKS,
                {
                    inspection_spec.id: inspection_spec,
                    processing_spec.id: processing_spec,
                },
            ):
                self.set_test_plan(context, inspection_spec)
                self.mark_completed_checkpoint(context, inspection_spec.id)
                inspection_run_id = context.execution.run_id
                inspection_plan_hash = context.execution.plan_hash

                self.set_test_plan(context, processing_spec)
                self.mark_completed_checkpoint(context, processing_spec.id)
                processing_run_id = context.execution.run_id
                processing_plan_hash = context.execution.plan_hash

                self.set_test_plan(context, inspection_spec)
                self.assertEqual(context.execution.run_id, inspection_run_id)
                self.assertEqual(
                    context.execution.plan_hash,
                    inspection_plan_hash,
                )
                self.assertIs(
                    context.execution.status,
                    RunStatus.COMPLETED,
                )

                self.set_test_plan(context, processing_spec)

            self.assertEqual(context.execution.run_id, processing_run_id)
            self.assertEqual(
                context.execution.plan_hash,
                processing_plan_hash,
            )
            self.assertIs(
                context.execution.status,
                RunStatus.COMPLETED,
            )
            self.assertEqual(
                [item.run_id for item in context.execution_history],
                [inspection_run_id],
            )

    def test_restored_plan_rebuilds_an_artifact_overwritten_by_another_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))
            artifact = context.outputs_dir / "profile.txt"
            calls: list[str] = []

            def handler(current: PipelineContext) -> TaskResult:
                profile = current.options.correction_mode
                calls.append(profile)
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(profile, encoding="utf-8")
                return TaskResult.succeeded(artifacts=[artifact])

            spec = TaskSpec(
                "test.profiled_output",
                "processing",
                "Profiled output",
                handler,
            )
            original_options = context.options
            with patch.dict(TASKS, {spec.id: spec}):
                self.set_test_plan(context, spec)
                execute_tasks(context)
                original_run_id = context.execution.run_id

                context.options = replace(
                    original_options,
                    correction_mode="aggressive",
                )
                self.set_test_plan(context, spec)
                execute_tasks(context)

                context.options = original_options
                self.set_test_plan(context, spec)
                self.assertEqual(context.execution.run_id, original_run_id)
                execute_tasks(context)

            self.assertEqual(calls, ["balanced", "aggressive", "balanced"])
            self.assertEqual(
                artifact.read_text(encoding="utf-8"),
                "balanced",
            )
            self.assertEqual(
                context.execution.tasks[spec.id].attempts,
                2,
            )

    def test_task_version_change_invalidates_the_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(Path(temporary_directory))

            def handler(_context: PipelineContext) -> TaskResult:
                return TaskResult.succeeded()

            version_one = TaskSpec(
                "test.versioned",
                "processing",
                "Versioned",
                handler,
                version="1",
            )
            version_two = TaskSpec(
                "test.versioned",
                "processing",
                "Versioned",
                handler,
                version="2",
            )
            plan = [PlannedTask(version_one.id, "test")]
            with patch.dict(TASKS, {version_one.id: version_one}):
                context.set_plan(plan)
                first_hash = context.execution.plan_hash
                context.execution.start_pipeline()
            with patch.dict(TASKS, {version_two.id: version_two}):
                context.set_plan(plan)

        self.assertNotEqual(context.execution.plan_hash, first_hash)
        self.assertEqual(len(context.execution_history), 1)


class ManifestCompatibilityTests(unittest.TestCase):
    def inspect_context(self, video: Path) -> PipelineContext:
        metadata_dir = video.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "youtube_video_metadata.json").write_text(
            json.dumps({"duration_seconds": 180}),
            encoding="utf-8",
        )
        media = {
            "path": video.resolve().as_posix(),
            "filename": video.name,
        }
        with patch("pipeline.context.probe_video", return_value=media):
            return PipelineContext.inspect(video)

    def test_read_manifest_accepts_v3_and_v4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for version in (3, SCHEMA_VERSION):
                target = root / f"manifest-v{version}.json"
                payload = {"schema_version": version}
                if version == SCHEMA_VERSION:
                    payload.update(
                        {
                            "routing_facts": {},
                            "route": {
                                "status": "needs_content_inspection",
                                "missing_facts": [
                                    "has_subtitles",
                                    "video_type",
                                ],
                            },
                            "artifacts": {"by_task": {}},
                            "plan": {
                                "hash": "",
                                "tasks": [],
                            },
                            "execution": {
                                "run_id": "test-run",
                                "plan_hash": "",
                                "status": "not_started",
                                "tasks": {},
                            },
                        }
                    )
                target.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                self.assertEqual(
                    read_manifest(target)["schema_version"],
                    version,
                )

    def test_force_is_not_reactivated_from_a_manifest(self) -> None:
        restored = PipelineOptions.from_dict({"force": True})

        self.assertFalse(restored.force)

    def test_read_manifest_rejects_corruption_and_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corrupt = root / "corrupt.json"
            corrupt.write_text("{", encoding="utf-8")
            unknown = root / "unknown.json"
            unknown.write_text(
                json.dumps({"schema_version": 99}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ManifestValidationError,
                "JSON invalide",
            ):
                read_manifest(corrupt)
            with self.assertRaisesRegex(
                ManifestValidationError,
                "non supporte",
            ):
                read_manifest(unknown)

    def test_inspect_accepts_a_new_video_without_analysis_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "abcdefghijk"
            video_dir.mkdir()
            video = video_dir / "abcdefghijk.mp4"
            video.touch()

            context = self.inspect_context(video)

        self.assertEqual(context.routing_facts, RoutingFacts())

    def test_legacy_analysis_is_migrated_then_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "abcdefghijk"
            metadata_dir = video_dir / "metadata"
            metadata_dir.mkdir(parents=True)
            video = video_dir / "abcdefghijk.mp4"
            video.touch()
            legacy = metadata_dir / "pipeline_analysis.json"
            legacy.write_text(
                json.dumps(
                    {
                        "has_subtitles": True,
                        "video_type": "motion_design",
                    }
                ),
                encoding="utf-8",
            )

            context = self.inspect_context(video)
            write_manifest(context)
            manifest = read_manifest(context.manifest_path)
            self.assertFalse(legacy.exists())

        self.assertTrue(manifest["routing_facts"]["has_subtitles"])
        self.assertEqual(
            manifest["routing_facts"]["video_type"],
            "motion_design",
        )

    def test_inspect_migrates_v3_features_to_routing_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_dir = Path(temporary_directory) / "abcdefghijk"
            metadata_dir = video_dir / "metadata"
            metadata_dir.mkdir(parents=True)
            video = video_dir / "abcdefghijk.mp4"
            video.touch()
            (metadata_dir / "video_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "features": {
                            "has_subtitles": {
                                "value": True,
                                "details": {"ratio": 0.75},
                            },
                            "video_type": "motion_design",
                        },
                    }
                ),
                encoding="utf-8",
            )

            context = self.inspect_context(video)

        self.assertTrue(context.has_subtitles)
        self.assertEqual(context.video_type, "motion_design")
        self.assertEqual(
            context.routing_facts.has_subtitles_details,
            {"ratio": 0.75},
        )


class OrchestratorPublicApiTests(unittest.TestCase):
    def make_context(
        self,
        root: Path,
        *,
        ready: bool,
    ) -> PipelineContext:
        video_dir = root / "abcdefghijk"
        video_dir.mkdir(parents=True)
        video = video_dir / "abcdefghijk.mp4"
        video.touch()
        facts = (
            RoutingFacts(False, VideoType.INTERVIEW)
            if ready
            else RoutingFacts()
        )
        context = PipelineContext(
            video_path=video,
            media={"duration_seconds": 180.0},
            routing_facts=facts,
        )
        return context

    def test_plan_video_persists_the_plan_on_the_context(self) -> None:
        from pipeline.orchestrator import plan_video

        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(
                Path(temporary_directory),
                ready=True,
            )
            with (
                patch(
                    "pipeline.orchestrator.PipelineContext.inspect",
                    return_value=context,
                ),
                patch("pipeline.orchestrator.write_manifest") as write,
            ):
                returned = plan_video(
                    context.video_path,
                    PipelineOptions(),
                )

        self.assertIs(returned, context)
        self.assertTrue(context.plan)
        self.assertTrue(context.execution.plan_hash)
        write.assert_called_once_with(context)

    def test_inspect_video_probe_only_does_not_execute_handlers(self) -> None:
        from pipeline.orchestrator import inspect_video

        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(
                Path(temporary_directory),
                ready=False,
            )
            with (
                patch(
                    "pipeline.orchestrator.PipelineContext.inspect",
                    return_value=context,
                ),
                patch("pipeline.orchestrator.execute_tasks") as execute,
                patch("pipeline.orchestrator.write_manifest") as write,
            ):
                returned = inspect_video(
                    context.video_path,
                    PipelineOptions(),
                    probe_only=True,
                )

        self.assertIs(returned, context)
        self.assertEqual(context.plan[0].id, "frames.extract")
        execute.assert_not_called()
        write.assert_called_once_with(context)

    def test_run_video_reuses_one_context_for_inspection_and_processing(
        self,
    ) -> None:
        from pipeline.orchestrator import run_video

        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(
                Path(temporary_directory),
                ready=False,
            )
            executions: list[list[str]] = []

            def execute(current: PipelineContext, *, dry_run: bool) -> None:
                self.assertIs(current, context)
                self.assertFalse(dry_run)
                executions.append([task.id for task in current.plan])
                if len(executions) == 1:
                    current.routing_facts = RoutingFacts(
                        False,
                        VideoType.INTERVIEW,
                    )

            with (
                patch(
                    "pipeline.orchestrator.PipelineContext.inspect",
                    return_value=context,
                ) as inspect,
                patch(
                    "pipeline.orchestrator.execute_tasks",
                    side_effect=execute,
                ),
                patch("pipeline.orchestrator.write_manifest"),
            ):
                returned = run_video(
                    context.video_path,
                    PipelineOptions(),
                )

        self.assertIs(returned, context)
        inspect.assert_called_once()
        self.assertEqual(len(executions), 2)
        self.assertEqual(executions[0][0], "frames.extract")
        self.assertIn("transcript.whisper", executions[1])

    def test_force_run_deletes_outputs_before_inspection(self) -> None:
        from pipeline.orchestrator import run_video

        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(
                Path(temporary_directory),
                ready=True,
            )
            obsolete = context.outputs_dir / "legacy" / "obsolete.txt"
            obsolete.parent.mkdir(parents=True)
            obsolete.write_text("obsolete", encoding="utf-8")

            def inspect(video_path, options):
                self.assertFalse(context.outputs_dir.exists())
                return context

            with (
                patch(
                    "pipeline.orchestrator.PipelineContext.inspect",
                    side_effect=inspect,
                ),
                patch("pipeline.orchestrator.execute_tasks"),
                patch("pipeline.orchestrator.write_manifest"),
            ):
                run_video(
                    context.video_path,
                    PipelineOptions(force=True),
                    skip_inspection=True,
                )

            self.assertFalse(obsolete.exists())

    def test_force_dry_run_does_not_delete_outputs(self) -> None:
        from pipeline.orchestrator import run_video

        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self.make_context(
                Path(temporary_directory),
                ready=True,
            )
            preserved = context.outputs_dir / "custom.txt"
            preserved.parent.mkdir(parents=True)
            preserved.write_text("keep", encoding="utf-8")

            with (
                patch(
                    "pipeline.orchestrator.PipelineContext.inspect",
                    return_value=context,
                ),
                patch("pipeline.orchestrator.execute_tasks"),
                patch("pipeline.orchestrator.write_manifest"),
            ):
                run_video(
                    context.video_path,
                    PipelineOptions(force=True),
                    skip_inspection=True,
                    dry_run=True,
                )

            self.assertEqual(preserved.read_text(encoding="utf-8"), "keep")


class PipelineCliTests(unittest.TestCase):
    def test_review_scope_none_is_accepted(self) -> None:
        from pipeline.cli import parse_args

        args = parse_args(
            ["run", "abcdefghijk", "--review-scope", "none"]
        )

        self.assertEqual(args.review_scope, "none")

    def test_run_batch_alias_selects_global_openai_batch_mode(self) -> None:
        from pipeline.cli import parse_args

        args = parse_args(["run", "abcdefghijk", "--batch"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.openai_mode, "batch")

    def test_task_command_parses_task_id_and_selector(self) -> None:
        from pipeline.cli import parse_args

        args = parse_args(
            [
                "task",
                "frames.extract",
                "abcdefghijk",
                "--dry-run",
            ]
        )

        self.assertEqual(args.command, "task")
        self.assertEqual(args.task_id, "frames.extract")
        self.assertEqual(args.selector, "abcdefghijk")
        self.assertTrue(args.dry_run)

    def test_task_list_does_not_require_a_task_id(self) -> None:
        from pipeline.cli import parse_args

        args = parse_args(["task", "--list"])

        self.assertTrue(args.list_tasks)
        self.assertIsNone(args.task_id)


if __name__ == "__main__":
    unittest.main()
