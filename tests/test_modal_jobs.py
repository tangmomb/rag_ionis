from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.support import remote_backend, scaleway_execution
from pipeline.support.modal_jobs import ModalClient, ModalConfig, ModalJobError
from pipeline.support.scaleway_jobs import ScalewayJobError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeCall:
    """Each item of ``outcomes`` is a value to return or an exception to raise."""

    object_id = "fc-123"

    def __init__(self, clock: FakeClock, outcomes: list, *, elapse: bool = True):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.elapse = elapse
        self.cancel = Mock()
        self.timeouts: list[float] = []

    def get(self, timeout: float):
        self.timeouts.append(timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            if self.elapse and isinstance(outcome, TimeoutError):
                self.clock.now += timeout
            raise outcome
        return outcome


def client_for(call: FakeCall, clock: FakeClock, **config) -> tuple[ModalClient, Mock]:
    function = Mock()
    function.spawn.return_value = call
    options = {"poll_seconds": 60.0, "timeout_seconds": 600.0, **config}
    return ModalClient(ModalConfig(**options), function=function, clock=clock), function


JOB = {"operation": "pipeline_command", "control": {"job_id": "job-1"}}


class ModalClientTests(unittest.TestCase):
    def test_returns_output_after_waiting_for_a_running_job(self) -> None:
        clock = FakeClock()
        output = {"video_id": "abc", "status": "completed"}
        call = FakeCall(clock, [TimeoutError(), TimeoutError(), output])
        client, function = client_for(call, clock)

        job_id, result = client.run(JOB, object_store=object(), bucket="bucket")

        self.assertEqual((job_id, result), ("job-1", output))
        function.spawn.assert_called_once_with(JOB)
        call.cancel.assert_not_called()

    def test_remote_failure_is_a_scaleway_compatible_error_and_cancels(self) -> None:
        clock = FakeClock()
        call = FakeCall(clock, [ValueError("Identifiant video invalide.")])
        client, _ = client_for(call, clock)

        with self.assertRaisesRegex(ModalJobError, "ValueError: Identifiant video"):
            client.run(JOB)

        self.assertTrue(issubclass(ModalJobError, ScalewayJobError))
        call.cancel.assert_called_once()

    def test_remote_timeout_raised_immediately_is_not_mistaken_for_waiting(self) -> None:
        clock = FakeClock()
        call = FakeCall(clock, [TimeoutError("socket")], elapse=False)
        client, _ = client_for(call, clock)

        with self.assertRaisesRegex(ModalJobError, "TimeoutError: socket"):
            client.run(JOB)

    def test_function_timeout_is_a_failure(self) -> None:
        FunctionTimeoutError = type("FunctionTimeoutError", (TimeoutError,), {})
        clock = FakeClock()
        call = FakeCall(clock, [FunctionTimeoutError("6 h")])
        client, _ = client_for(call, clock)

        with self.assertRaisesRegex(ModalJobError, "FunctionTimeoutError"):
            client.run(JOB)

    def test_overall_deadline_cancels_the_call(self) -> None:
        clock = FakeClock()
        call = FakeCall(clock, [TimeoutError()] * 10)
        client, _ = client_for(call, clock, timeout_seconds=150.0)

        with self.assertRaisesRegex(ModalJobError, "Delai depasse"):
            client.run(JOB)

        self.assertEqual(call.timeouts, [60.0, 60.0, 30.0])
        call.cancel.assert_called_once()

    def test_non_dict_output_is_rejected(self) -> None:
        clock = FakeClock()
        client, _ = client_for(FakeCall(clock, ["done"]), clock)

        with self.assertRaisesRegex(ModalJobError, "sortie invalide"):
            client.run(JOB)

    def test_config_reads_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MODAL_APP_NAME": "custom-app",
                "MODAL_ENVIRONMENT": "staging",
                "MODAL_POLL_SECONDS": "30",
                "MODAL_JOB_TIMEOUT_SECONDS": "3600",
            },
        ):
            config = ModalConfig.from_env()

        self.assertEqual(config.app_name, "custom-app")
        self.assertEqual(config.function_name, "process_job")
        self.assertEqual(config.environment_name, "staging")
        self.assertEqual((config.poll_seconds, config.timeout_seconds), (30.0, 3600.0))


class ModalBackendSelectionTests(unittest.TestCase):
    def test_modal_backend_delegates_gpu_commands(self) -> None:
        with patch.dict(
            "os.environ",
            {"PIPELINE_EXECUTION_BACKEND": "modal", "SCALEWAY_WORKER": ""},
        ):
            self.assertTrue(scaleway_execution.should_delegate_to_scaleway("run"))
            self.assertEqual(
                remote_backend.remote_job_client("modal").__class__, ModalClient
            )

    def test_unknown_backend_is_rejected(self) -> None:
        with patch.dict("os.environ", {"PIPELINE_EXECUTION_BACKEND": "runpod"}):
            with self.assertRaisesRegex(RuntimeError, "'modal'"):
                remote_backend.configured_backend()

    def test_modal_backend_allows_several_concurrent_jobs_setting(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "PIPELINE_EXECUTION_BACKEND": "modal",
                    "SCALEWAY_WORKER": "",
                    "SCALEWAY_MAX_CONCURRENT_JOBS": "4",
                },
            ),
            patch.object(
                scaleway_execution,
                "run_video_on_scaleway",
                return_value={"backend": "modal"},
            ) as run_one,
        ):
            results = scaleway_execution.run_videos_on_scaleway(
                [Path("a/video.mp4")],
                Mock(),
                scaleway_execution.RemoteCommand("run"),
            )

        self.assertEqual(results, [{"backend": "modal"}])
        run_one.assert_called_once()


class ModalAppFileTests(unittest.TestCase):
    def test_app_reuses_worker_entrypoint_on_an_l4(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "deploy" / "modal_app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('gpu="L4"', source)
        self.assertIn("from pipeline.workers.scaleway_ingestion import process_job", source)
        self.assertIn('volumes={"/models": models_volume}', source)


if __name__ == "__main__":
    unittest.main()
