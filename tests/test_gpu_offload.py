from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline import catalog, gpu_offload
from pipeline.contracts import TaskResult, TaskStatus
from pipeline.support import remote_backend, scaleway_execution


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class FakeContext:
    video_path: Path
    force_rebuild: bool = True
    transcripts_dir_name: str = "transcripts_whisper"
    video_type: str | None = "interview"


@dataclass(frozen=True)
class FakeSpec:
    id: str
    handler: object
    requires_gpu: bool = True


def fake_classify(context) -> TaskResult:
    """Moves every frame into footage/ and writes a manifest, like the real step."""
    images = Path(context.video_path).parent / "outputs" / "images"
    (images / "footage").mkdir(parents=True, exist_ok=True)
    for frame in sorted(images.glob("*.jpg")):
        frame.rename(images / "footage" / frame.name)
    manifest = images / "frame_classification_manifest.json"
    manifest.write_text(json.dumps({"footage_count": 2}), encoding="utf-8")
    return TaskResult.succeeded(manifest, artifacts=(manifest,), reason="Frames classifiees.")


def fake_whisper(context) -> TaskResult:
    directory = Path(context.video_path).parent / "outputs" / context.transcripts_dir_name
    audio = directory / "audio" / "video.mp3"
    assert audio.read_bytes() == b"mp3", "the audio must reach the GPU"
    target = directory / "transcript_1_brut.txt"
    target.write_text(f"type={context.video_type}", encoding="utf-8")
    return TaskResult.succeeded({"transcript": target}, artifacts=(target,))


class InlineCall:
    """A Modal call that runs the remote side in another directory at once."""

    object_id = "fc-test"

    def __init__(self, arguments: tuple, remote_root: Path):
        task_id, video_dir_name, video_name, archive, task_context = arguments
        self.output = gpu_offload.run_task_in_directory(
            task_id, remote_root / video_dir_name, video_name, archive, task_context
        )
        self.cancel = Mock()

    def get(self, timeout: float):
        return self.output


def inline_function(remote_root: Path) -> Mock:
    function = Mock()
    function.spawn.side_effect = lambda *arguments: InlineCall(arguments, remote_root)
    return function


def make_video(root: Path) -> Path:
    video_dir = root / "archive" / "abc123"
    images = video_dir / "outputs" / "images"
    images.mkdir(parents=True)
    (images / "frame_0001.jpg").write_bytes(b"one")
    (images / "frame_0002.jpg").write_bytes(b"two")
    video = video_dir / "abc123.mp4"
    video.write_bytes(b"video bytes stay local")
    return video


class ArchiveTests(unittest.TestCase):
    def test_pack_and_unpack_keep_relative_layout_and_skip_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            (root / "a" / "skip").mkdir(parents=True)
            (root / "a" / "keep.txt").write_text("keep", encoding="utf-8")
            (root / "a" / "skip" / "big.mp3").write_text("skip", encoding="utf-8")

            data = gpu_offload.pack(root, [root / "a"], excluded=[root / "a" / "skip"])
            target = Path(directory) / "target"
            count = gpu_offload.unpack(data, target)

            self.assertEqual(count, 1)
            self.assertEqual((target / "a" / "keep.txt").read_text(encoding="utf-8"), "keep")
            self.assertFalse((target / "a" / "skip").exists())

    def test_unpack_ignores_paths_leaving_the_directory(self) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo("../escape.txt")
            info.size = 3
            archive.addfile(info, io.BytesIO(b"bad"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "inside"
            self.assertEqual(gpu_offload.unpack(buffer.getvalue(), target), 0)
            self.assertFalse((Path(directory) / "escape.txt").exists())

    def test_values_with_paths_round_trip(self) -> None:
        root = Path("/remote/video")
        encoded = gpu_offload.encode_value(
            {"files": [root / "outputs" / "a.json"], "count": 2}, root
        )
        self.assertEqual(json.loads(json.dumps(encoded))["count"], 2)
        local = Path("local") / "video"
        decoded = gpu_offload.decode_value(encoded, local)
        self.assertEqual(decoded["files"], [local / "outputs" / "a.json"])


class RemoteTaskTests(unittest.TestCase):
    def test_classification_replaces_local_frames_and_never_sends_the_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = make_video(root)
            function = inline_function(root / "remote")
            spec = FakeSpec("frames.classify", fake_classify)

            with patch.dict(catalog.TASKS, {"frames.classify": spec}):
                result = gpu_offload.run_task_remotely(spec, FakeContext(video), function=function)

            images = video.parent / "outputs" / "images"
            self.assertIs(result.status, TaskStatus.SUCCEEDED)
            self.assertEqual(sorted(p.name for p in images.glob("*.jpg")), [])
            self.assertEqual((images / "footage" / "frame_0001.jpg").read_bytes(), b"one")
            self.assertEqual(result.value, images / "frame_classification_manifest.json")
            self.assertEqual(result.artifacts, (images / "frame_classification_manifest.json",))
            remote_video = root / "remote" / "abc123" / "abc123.mp4"
            self.assertEqual(remote_video.read_bytes(), b"")

    def test_whisper_sends_audio_only_and_keeps_local_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = make_video(root)
            audio_dir = video.parent / "outputs" / "transcripts_whisper" / "audio"

            def extract_audio(video_path, target_dir):
                (target_dir / "video.mp3").write_bytes(b"mp3")

            spec = FakeSpec("transcript.whisper", fake_whisper)
            with (
                patch.dict(catalog.TASKS, {"transcript.whisper": spec}),
                patch(
                    "pipeline.steps.transcripts.transcribe_with_whisper.extract_audio",
                    side_effect=extract_audio,
                ),
            ):
                result = gpu_offload.run_task_remotely(
                    spec,
                    FakeContext(video, video_type="long_video"),
                    function=inline_function(root / "remote"),
                )

            transcript = video.parent / "outputs" / "transcripts_whisper" / "transcript_1_brut.txt"
            self.assertEqual(transcript.read_text(encoding="utf-8"), "type=long_video")
            self.assertEqual((audio_dir / "video.mp3").read_bytes(), b"mp3")
            self.assertEqual(result.value, {"transcript": transcript})

    def test_blocked_remote_task_leaves_local_files_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = make_video(root)
            spec = FakeSpec("frames.classify", lambda context: TaskResult.blocked("Aucune frame."))

            with patch.dict(catalog.TASKS, {"frames.classify": spec}):
                result = gpu_offload.run_task_remotely(
                    spec, FakeContext(video), function=inline_function(root / "remote")
                )

            self.assertIs(result.status, TaskStatus.BLOCKED)
            self.assertEqual(result.reason, "Aucune frame.")
            self.assertTrue((video.parent / "outputs" / "images" / "frame_0001.jpg").exists())


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class WaitingCall:
    object_id = "fc-wait"

    def __init__(self, clock: FakeClock, outcomes: list, *, elapse: bool = True):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.elapse = elapse
        self.cancel = Mock()

    def get(self, timeout: float):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            if self.elapse and isinstance(outcome, TimeoutError):
                self.clock.now += timeout
            raise outcome
        return outcome


class CallRemoteTests(unittest.TestCase):
    def call(self, outcomes: list, *, elapse: bool = True, timeout: str = "600"):
        clock = FakeClock()
        waiting = WaitingCall(clock, outcomes, elapse=elapse)
        function = Mock()
        function.spawn.return_value = waiting
        with patch.dict(
            "os.environ",
            {"MODAL_POLL_SECONDS": "60", "MODAL_TASK_TIMEOUT_SECONDS": timeout},
        ):
            try:
                return gpu_offload.call_remote(function, (), label="task", clock=clock), waiting
            except gpu_offload.GpuOffloadError as error:
                return error, waiting

    def test_waits_while_the_task_runs(self) -> None:
        output, waiting = self.call([TimeoutError(), {"status": "completed"}])
        self.assertEqual(output, {"status": "completed"})
        waiting.cancel.assert_not_called()

    def test_remote_failure_cancels_the_call(self) -> None:
        error, waiting = self.call([ValueError("CUDA out of memory")])
        self.assertIn("ValueError: CUDA out of memory", str(error))
        waiting.cancel.assert_called_once()

    def test_remote_timeout_raised_at_once_is_a_failure(self) -> None:
        error, _ = self.call([TimeoutError("socket")], elapse=False)
        self.assertIn("TimeoutError: socket", str(error))

    def test_deadline_cancels_the_call(self) -> None:
        error, waiting = self.call([TimeoutError()] * 5, timeout="150")
        self.assertIn("Delai depasse", str(error))
        waiting.cancel.assert_called_once()


class BackendSelectionTests(unittest.TestCase):
    def test_only_gpu_tasks_are_offloaded_with_the_modal_backend(self) -> None:
        handler = Mock(return_value=TaskResult.succeeded())
        with (
            patch.dict("os.environ", {"PIPELINE_EXECUTION_BACKEND": "modal", "PIPELINE_GPU_WORKER": ""}),
            patch.object(gpu_offload, "run_task_remotely", return_value="remote") as remote,
        ):
            self.assertEqual(gpu_offload.run_handler(FakeSpec("x", handler), "ctx"), "remote")
            gpu_offload.run_handler(FakeSpec("y", handler, requires_gpu=False), "ctx")

        remote.assert_called_once()
        handler.assert_called_once_with("ctx")

    def test_gpu_worker_and_other_backends_run_handlers_locally(self) -> None:
        handler = Mock(return_value=TaskResult.succeeded())
        for environment in (
            {"PIPELINE_EXECUTION_BACKEND": "modal", "PIPELINE_GPU_WORKER": "1"},
            {"PIPELINE_EXECUTION_BACKEND": "local", "PIPELINE_GPU_WORKER": ""},
        ):
            with patch.dict("os.environ", environment):
                self.assertFalse(gpu_offload.gpu_offload_enabled())
                gpu_offload.run_handler(FakeSpec("x", handler), "ctx")
        self.assertEqual(handler.call_count, 2)

    def test_modal_backend_runs_the_pipeline_locally(self) -> None:
        with patch.dict(
            "os.environ", {"PIPELINE_EXECUTION_BACKEND": "modal", "SCALEWAY_WORKER": ""}
        ):
            self.assertEqual(remote_backend.configured_backend(), "modal")
            self.assertFalse(scaleway_execution.should_delegate_to_scaleway("run"))

    def test_unknown_backend_is_rejected(self) -> None:
        with patch.dict("os.environ", {"PIPELINE_EXECUTION_BACKEND": "runpod"}):
            with self.assertRaisesRegex(RuntimeError, "'modal'"):
                remote_backend.configured_backend()

    def test_every_gpu_task_has_its_files_defined(self) -> None:
        context = FakeContext(Path("archive/abc/abc.mp4"))
        for task_id, spec in catalog.TASKS.items():
            if spec.requires_gpu:
                files = gpu_offload.task_files(task_id, context)
                self.assertTrue(files.inputs and files.outputs, task_id)


class DeploymentFileTests(unittest.TestCase):
    def test_modal_app_needs_no_secret_and_runs_the_offloaded_tasks(self) -> None:
        source = (ROOT_DIR / "deploy" / "modal_app.py").read_text(encoding="utf-8")
        self.assertNotIn("Secret", source)
        self.assertIn("def run_gpu_task(", source)
        self.assertIn("run_task_in_directory", source)
        self.assertIn('"PIPELINE_GPU_WORKER": "1"', source)
        self.assertEqual(gpu_offload.FUNCTION_NAME, "run_gpu_task")

    def test_youtube_downloads_use_the_configured_proxy(self) -> None:
        from pipeline.ingest import download_videos

        downloader = Mock()
        downloader.__enter__ = Mock(return_value=downloader)
        downloader.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory, (
            patch.dict("os.environ", {"YT_PROXY": "socks5://host.docker.internal:1080"})
        ), patch.object(download_videos, "ffmpeg_exe", return_value=Path("ffmpeg")), patch.object(
            download_videos.yt_dlp, "YoutubeDL", return_value=downloader
        ) as youtube_dl:
            with self.assertRaises(FileNotFoundError):
                download_videos.download_video(
                    ("abc", "Titre", "https://www.youtube.com/watch?v=abc", {}),
                    Path(directory),
                    Path(directory),
                    force=True,
                    sync_cached_metadata=False,
                )

        options = youtube_dl.call_args.args[0]
        self.assertEqual(options["proxy"], "socks5://host.docker.internal:1080")


if __name__ == "__main__":
    unittest.main()
