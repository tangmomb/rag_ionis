from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.support import json_io


class JsonIoTests(unittest.TestCase):
    def test_write_json_retries_a_transient_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "video_manifest.json"
            real_replace = json_io.os.replace
            attempts = 0

            def replace_after_transient_lock(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("temporary Windows file lock")
                return real_replace(source, destination)

            with (
                patch.object(json_io.os, "replace", side_effect=replace_after_transient_lock),
                patch.object(json_io.time, "sleep") as sleep,
            ):
                result = json_io.write_json(target, {"status": "done"})

            self.assertEqual(result, target)
            self.assertEqual(json_io.read_json(target), {"status": "done"})
            self.assertEqual(attempts, 2)
            sleep.assert_called_once_with(0.05)


if __name__ == "__main__":
    unittest.main()
