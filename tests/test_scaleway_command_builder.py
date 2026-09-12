from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "utils" / "app_command_builder" / "app.js"
INDEX_PATH = ROOT_DIR / "utils" / "app_command_builder" / "index.html"


class ScalewayCommandBuilderTests(unittest.TestCase):
    def test_actions_include_image_lifecycle_and_worker_info(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        index = INDEX_PATH.read_text(encoding="utf-8")

        self.assertIn('"Scaleway"', source)
        self.assertIn('id: "scaleway-build-image"', source)
        self.assertIn('rg.fr-par.scw.cloud/rag-ionis/rag-ionis-scaleway', source)
        self.assertNotIn('rg.fr-par.scw.cloud/NAMESPACE/rag-ionis-scaleway', source)
        self.assertIn('id: "scaleway-publish-image"', source)
        self.assertIn('$env:SCALEWAY_IMAGE_REPOSITORY = $repository', source)
        self.assertIn('bash deploy/publish-scaleway-image.sh $commitTag', source)
        self.assertIn("cache distant", source)
        self.assertIn('id: "scaleway-inspect-image"', source)
        self.assertIn('docker buildx imagetools inspect', source)
        self.assertIn('id: "scaleway-pull-latest"', source)
        self.assertIn('id: "scaleway-connect"', source)
        self.assertIn('title: "Se connecter à la VM Scaleway"', source)
        self.assertIn('command: `ssh${identityOption}', source)
        self.assertIn('id: "scaleway-worker-info"', source)
        worker_info = source.split('id: "scaleway-worker-info"', 1)[1].split(
            '];', 1
        )[0]
        self.assertIn('fields: scalewayWorkerConnectionFields', worker_info)
        self.assertIn('id: "scalewayWorkerHost"', source)
        self.assertIn(r'C:\Users\rgb\.ssh\rag_ionis_scaleway_admin', source)
        self.assertIn('value: "51.159.135.156"', source)
        self.assertIn('value: "root"', source)
        self.assertNotIn('ubuntu@179.237.98.117', worker_info)
        self.assertIn('sudo systemctl cat rag-ionis-scaleway-worker', source)
        self.assertIn('/etc/rag-ionis/scaleway-worker.env', source)
        self.assertIn('sudo cat /etc/rag-ionis/scaleway-worker.env', source)
        self.assertIn('clés S3 et API en clair', source)
        self.assertIn('app.js?v=20260822-vps-update-stats', index)


if __name__ == "__main__":
    unittest.main()
