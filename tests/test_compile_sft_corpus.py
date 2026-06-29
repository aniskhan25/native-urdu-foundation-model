import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sft.compile_sft_corpus import accept_records
from sft.compile_sft_corpus import canonicalize_record
from sft.compile_sft_corpus import compile_corpus
from sft.compile_sft_corpus import load_config
from sft.compile_sft_corpus import record_matches_filters
from sft.compile_sft_corpus import split_records


QUALITY = {
    "min_prompt_chars": 5,
    "min_response_chars": 5,
    "max_prompt_chars": 1000,
    "max_response_chars": 1000,
    "min_combined_urdu_script_ratio": 0.3,
}


class CompileSftCorpusTests(unittest.TestCase):
    def test_rejects_training_config_with_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.yaml"
            path.write_text("training:\n  lr: 0.00002\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "set SFT_SOURCE_CONFIG"):
                load_config(path)

    def test_canonicalizes_configured_fields(self):
        source = {
            "id": "aya",
            "prompt_field": "inputs",
            "response_field": "targets",
            "category_field": "annotation_type",
            "license": "apache-2.0",
            "provenance": "human",
        }

        record = canonicalize_record(
            {"inputs": "  سوال کیا ہے؟  ", "targets": "  یہ جواب ہے۔  ", "annotation_type": "original"},
            source,
        )

        self.assertEqual(record["prompt"], "سوال کیا ہے؟")
        self.assertEqual(record["response"], "یہ جواب ہے۔")
        self.assertEqual(record["category"], "original")
        self.assertEqual(record["license"], "apache-2.0")

    def test_matches_exact_source_filters(self):
        self.assertTrue(record_matches_filters({"language_code": "urd"}, {"language_code": "urd"}))
        self.assertFalse(record_matches_filters({"language_code": "eng"}, {"language_code": "urd"}))

    def test_rejects_eval_overlap_and_duplicate_prompt(self):
        records = [
            {"prompt": "محفوظ سوال", "response": "پہلا مناسب جواب", "source": "a", "category": "qa"},
            {"prompt": "محفوظ سوال", "response": "دوسرا مناسب جواب", "source": "b", "category": "qa"},
            {"prompt": "امتحانی سوال", "response": "یہ جواب شامل نہیں ہونا چاہیے", "source": "a", "category": "qa"},
        ]

        accepted, rejected = accept_records(records, QUALITY, {"امتحانی سوال"})

        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected["duplicate_prompt"], 1)
        self.assertEqual(rejected["eval_overlap"], 1)

    def test_split_is_deterministic_and_keeps_groups(self):
        records = [
            {
                "prompt": f"سوال نمبر {index}",
                "response": f"جواب نمبر {index}",
                "source": "source-a" if index < 20 else "source-b",
                "category": "qa",
            }
            for index in range(40)
        ]

        train_a, validation_a = split_records(records, validation_fraction=0.1, seed=7)
        train_b, validation_b = split_records(records, validation_fraction=0.1, seed=7)

        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertEqual(len(train_a), 36)
        self.assertEqual(len(validation_a), 4)
        self.assertEqual({record["source"] for record in validation_a}, {"source-a", "source-b"})

    def test_split_keeps_validation_nonempty_for_small_pilot(self):
        records = [
            {"prompt": "پہلا سوال", "response": "پہلا جواب", "source": "source", "category": "qa"},
            {"prompt": "دوسرا سوال", "response": "دوسرا جواب", "source": "source", "category": "qa"},
        ]

        train, validation = split_records(records, validation_fraction=0.05, seed=7)

        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)

    def test_compile_corpus_filters_language_and_tracks_sources(self):
        config = {
            "include_curated_seed": False,
            "quality": QUALITY,
            "exclude_prompt_files": [],
            "sources": [
                {
                    "id": "aya",
                    "dataset": "example/aya",
                    "prompt_field": "inputs",
                    "response_field": "targets",
                    "category_field": "annotation_type",
                    "filters": {"language_code": "urd"},
                    "max_records": 10,
                    "max_scanned": 10,
                    "license": "apache-2.0",
                    "provenance": "human",
                }
            ],
        }
        rows = [
            {"inputs": "یہ اردو سوال ہے", "targets": "یہ اردو جواب مناسب ہے", "language_code": "urd", "annotation_type": "original"},
            {"inputs": "English question", "targets": "English answer", "language_code": "eng", "annotation_type": "original"},
        ]

        with tempfile.TemporaryDirectory() as tmp, patch(
            "sft.compile_sft_corpus.iter_hf_source", return_value=iter(rows)
        ):
            records, summary = compile_corpus(config, Path(tmp) / "config.yaml")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "aya")
        self.assertEqual(summary["sources"]["aya"]["seen"], 2)
        self.assertEqual(summary["sources"]["aya"]["matched"], 1)
        self.assertEqual(summary["sources"]["aya"]["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
