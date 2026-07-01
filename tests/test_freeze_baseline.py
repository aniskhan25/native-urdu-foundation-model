import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.freeze_baseline import artifact_records
from scripts.freeze_baseline import create_lock
from scripts.freeze_baseline import verify_lock


class FreezeBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.artifact = self.root / "model.pt"
        self.artifact.write_bytes(b"checkpoint")
        self.spec = self.root / "baseline.yaml"
        self.spec.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "baseline_id": "test_baseline",
                    "artifacts": [{"id": "model", "path": "model.pt"}],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_artifact_records_hash_relative_files(self):
        spec = yaml.safe_load(self.spec.read_text(encoding="utf-8"))
        records = artifact_records(spec, self.root)
        self.assertEqual(records[0]["size_bytes"], 10)
        self.assertEqual(records[0]["sha256"], hashlib.sha256(b"checkpoint").hexdigest())

    @patch("scripts.freeze_baseline.repository_state")
    def test_create_and_verify_lock(self, repository_state_mock):
        repository_state_mock.return_value = {"commit": "abc123", "branch": "main", "dirty": False}
        lock = create_lock(self.spec, self.root)
        lock_path = self.root / "baseline.lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")

        verify_lock(self.spec, lock_path, self.root)

        self.artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Artifact metadata"):
            verify_lock(self.spec, lock_path, self.root)

    @patch("scripts.freeze_baseline.repository_state")
    def test_dirty_repository_is_rejected(self, repository_state_mock):
        repository_state_mock.return_value = {"commit": "abc123", "branch": "main", "dirty": True}
        with self.assertRaisesRegex(RuntimeError, "dirty repository"):
            create_lock(self.spec, self.root)

    def test_duplicate_artifact_ids_are_rejected(self):
        spec = {
            "artifacts": [
                {"id": "model", "path": str(self.artifact)},
                {"id": "model", "path": str(self.artifact)},
            ]
        }
        with self.assertRaisesRegex(ValueError, "Duplicate artifact id"):
            artifact_records(spec, self.root)


if __name__ == "__main__":
    unittest.main()
