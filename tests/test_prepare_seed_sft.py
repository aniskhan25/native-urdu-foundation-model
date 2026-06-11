import json
import tempfile
import unittest
from pathlib import Path

from sft.prepare_seed_sft import SEED_RECORDS
from sft.prepare_seed_sft import normalize_record
from sft.prepare_seed_sft import split_records
from sft.prepare_seed_sft import validate_records
from sft.prepare_seed_sft import write_jsonl


class PrepareSeedSftTests(unittest.TestCase):
    def test_seed_records_have_required_fields(self):
        records = [normalize_record(record) for record in SEED_RECORDS]

        validate_records(records)
        self.assertGreaterEqual(len(records), 20)

    def test_split_is_deterministic(self):
        records = [normalize_record(record) for record in SEED_RECORDS]

        train_a, validation_a = split_records(records, validation_fraction=0.2, seed=7)
        train_b, validation_b = split_records(records, validation_fraction=0.2, seed=7)

        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertGreater(len(train_a), len(validation_a))

    def test_rejects_duplicate_prompts(self):
        record = normalize_record(SEED_RECORDS[0])

        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            validate_records([record, record])

    def test_write_jsonl_outputs_valid_records(self):
        records = [normalize_record(record) for record in SEED_RECORDS[:2]]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            write_jsonl(path, records)

            loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(loaded, records)


if __name__ == "__main__":
    unittest.main()
