import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sft.compile_sft_corpus import accept_records
from sft.compile_sft_corpus import apply_category_limits
from sft.compile_sft_corpus import canonicalize_record
from sft.compile_sft_corpus import category_limit_shortages
from sft.compile_sft_corpus import compile_corpus
from sft.compile_sft_corpus import load_config
from sft.compile_sft_corpus import record_matches_filters
from sft.compile_sft_corpus import split_records
from sft.compile_sft_corpus import stratified_review_sample


QUALITY = {
    "min_prompt_chars": 5,
    "min_response_chars": 5,
    "max_prompt_chars": 1000,
    "max_response_chars": 1000,
    "min_combined_urdu_script_ratio": 0.3,
    "max_pashto_specific_chars": 1,
    "max_response_repeated_4gram_ratio": 0.08,
    "max_response_repeated_6gram_ratio": 0.04,
    "max_response_longest_repeated_ngram": 6,
    "max_response_url_hits": 1,
    "max_response_boilerplate_hits": 0,
    "excluded_source_categories": {},
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

    def test_translates_synthetic_reasoning_labels(self):
        source = {
            "id": "synthetic",
            "prompt_field": "instruction",
            "response_field": "output",
            "license": "cc-by-sa-4.0",
        }

        record = canonicalize_record(
            {
                "instruction": "دو جمع دو کتنے ہوتے ہیں؟",
                "output": "Reasoning: دو میں دو جمع کریں۔ Answer: چار",
            },
            source,
        )

        self.assertEqual(record["response"], "حل: دو میں دو جمع کریں۔ جواب: چار")

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

    def test_rejects_duplicate_response(self):
        records = [
            {"prompt": "پہلا مختلف سوال", "response": "ایک ہی مناسب جواب", "source": "a", "category": "qa"},
            {"prompt": "دوسرا مختلف سوال", "response": "ایک ہی مناسب جواب", "source": "a", "category": "qa"},
        ]

        accepted, rejected = accept_records(records, QUALITY, set())

        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected["duplicate_response"], 1)

    def test_rejects_excluded_source_category(self):
        quality = dict(QUALITY, excluded_source_categories={"synthetic": ["qa"]})
        records = [
            {
                "prompt": "ایک مصنوعی سوال",
                "response": "ایک مصنوعی جواب",
                "source": "synthetic",
                "category": "qa",
            }
        ]

        accepted, rejected = accept_records(records, quality, set())

        self.assertEqual(accepted, [])
        self.assertEqual(rejected["excluded_source_category"], 1)

    def test_rejects_pashto_mislabeled_as_urdu(self):
        records = [
            {
                "prompt": "پوښتنہ پہ کومہ لار بہ وګرځی؟",
                "response": "ځواب پہ پښتو ژبہ کې دی۔",
                "source": "aya",
                "category": "qa",
            }
        ]

        accepted, rejected = accept_records(records, QUALITY, set())

        self.assertEqual(accepted, [])
        self.assertEqual(rejected["pashto_specific_chars"], 1)

    def test_rejects_oversized_response(self):
        quality = dict(QUALITY, max_response_chars=20)
        records = [
            {
                "prompt": "یہ ایک مناسب سوال ہے",
                "response": "یہ جواب مقررہ حد سے بہت زیادہ طویل ہے",
                "source": "aya",
                "category": "qa",
            }
        ]

        accepted, rejected = accept_records(records, quality, set())

        self.assertEqual(accepted, [])
        self.assertEqual(rejected["long_response"], 1)

    def test_rejects_repeated_response_span(self):
        records = [
            {
                "prompt": "اس سوال کا واضح جواب دیں",
                "response": "یہ ایک بار بار آنے والا جملہ ہے جو یہ ایک بار بار آنے والا جملہ ہے",
                "source": "aya",
                "category": "qa",
            }
        ]

        accepted, rejected = accept_records(records, QUALITY, set())

        self.assertEqual(accepted, [])
        self.assertTrue(any(reason.startswith("response_repetition") for reason in rejected))

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

    def test_category_limits_are_deterministic_and_leave_other_sources_unrestricted(self):
        records = [
            {
                "prompt": f"سوال {source} {category} {index}",
                "response": f"جواب {source} {category} {index}",
                "source": source,
                "category": category,
            }
            for source, category, count in (
                ("synthetic", "reasoning", 5),
                ("synthetic", "ethics", 4),
                ("curated", "math", 3),
            )
            for index in range(count)
        ]
        limits = {"synthetic": {"reasoning": 2}}

        selected_a, rejected_a = apply_category_limits(records, limits, seed=7)
        selected_b, rejected_b = apply_category_limits(records, limits, seed=7)

        self.assertEqual(selected_a, selected_b)
        self.assertEqual(rejected_a, rejected_b)
        self.assertEqual(len(selected_a), 5)
        self.assertEqual(sum(record["source"] == "curated" for record in selected_a), 3)
        self.assertEqual(sum(record["category"] == "reasoning" for record in selected_a), 2)
        self.assertEqual(rejected_a["category_limit"], 3)
        self.assertEqual(rejected_a["category_not_selected"], 4)

    def test_review_sample_covers_each_group(self):
        records = [
            {
                "prompt": f"سوال {source} {category} {index}",
                "response": f"جواب {source} {category} {index}",
                "source": source,
                "category": category,
            }
            for source, category in (("a", "math"), ("a", "translation"), ("b", "story"))
            for index in range(10)
        ]

        sample_a = stratified_review_sample(records, size=9, seed=11)
        sample_b = stratified_review_sample(records, size=9, seed=11)

        self.assertEqual(sample_a, sample_b)
        self.assertEqual(len(sample_a), 9)
        self.assertEqual({(record["source"], record["category"]) for record in sample_a}, {
            ("a", "math"),
            ("a", "translation"),
            ("b", "story"),
        })

    def test_category_limit_shortages_reports_missing_records(self):
        records = [
            {"prompt": "سوال", "response": "جواب", "source": "a", "category": "math"},
        ]

        shortages = category_limit_shortages(records, {"a": {"math": 3, "story": 2}})

        self.assertEqual(shortages, {"a/math": 2, "a/story": 2})

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

    def test_compile_corpus_fails_when_required_category_quota_is_short(self):
        config = {
            "include_curated_seed": False,
            "require_full_category_limits": True,
            "category_limits": {"synthetic": {"reasoning": 2}},
            "quality": QUALITY,
            "exclude_prompt_files": [],
            "sources": [
                {
                    "id": "synthetic",
                    "dataset": "example/synthetic",
                    "prompt_field": "instruction",
                    "response_field": "output",
                    "category_field": "category",
                    "max_records": 10,
                    "license": "test",
                    "provenance": "test",
                }
            ],
        }
        rows = [
            {
                "instruction": "یہ ایک مناسب سوال ہے",
                "output": "یہ ایک مناسب اور مکمل جواب ہے",
                "category": "reasoning",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp, patch(
            "sft.compile_sft_corpus.iter_hf_source", return_value=iter(rows)
        ):
            with self.assertRaisesRegex(ValueError, "synthetic/reasoning"):
                compile_corpus(config, Path(tmp) / "config.yaml")


if __name__ == "__main__":
    unittest.main()
