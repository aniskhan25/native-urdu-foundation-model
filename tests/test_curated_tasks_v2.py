import unittest
from pathlib import Path

import yaml

from data_pipeline.normalize_urdu import normalize_urdu
from eval.generate_samples import load_prompts
from sft.compile_sft_corpus import compile_corpus
from sft.curated_tasks_v2 import CURATED_TASK_RECORDS
from sft.curated_tasks_v2 import EXPECTED_CATEGORY_COUNTS
from sft.curated_tasks_v2 import category_counts


class CuratedTasksV2Tests(unittest.TestCase):
    def test_generated_counts_match_contract(self):
        self.assertEqual(len(CURATED_TASK_RECORDS), 600)
        self.assertEqual(category_counts(), EXPECTED_CATEGORY_COUNTS)

    def test_generated_prompts_and_responses_are_unique(self):
        prompts = [normalize_urdu(record["prompt"]) for record in CURATED_TASK_RECORDS]
        responses = [normalize_urdu(record["response"]) for record in CURATED_TASK_RECORDS]

        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertEqual(len(responses), len(set(responses)))

    def test_generated_prompts_do_not_overlap_heldout_evaluation(self):
        generated = {normalize_urdu(record["prompt"]) for record in CURATED_TASK_RECORDS}
        heldout = {
            normalize_urdu(prompt)
            for prompt in load_prompts(Path("eval/prompts_sft_heldout.txt"))
        }

        self.assertFalse(generated & heldout)

    def test_math_generator_has_expected_first_results(self):
        first_batch = CURATED_TASK_RECORDS[:6]

        self.assertIn("125 + 34 = 159", first_batch[0]["response"])
        self.assertIn("600 - 120 = 480", first_batch[1]["response"])
        self.assertIn("3 × 12 = 36", first_batch[2]["response"])
        self.assertIn("14 ÷ 2 = 7", first_batch[3]["response"])
        self.assertIn("1000 - 50 = 950", first_batch[4]["response"])
        self.assertIn("80 ÷ 2 = 40", first_batch[5]["response"])

    def test_v2_config_compiles_exact_local_corpus(self):
        config_path = Path("configs/sft_sources_v2.yaml").resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        records, summary = compile_corpus(config, config_path)

        self.assertEqual(len(records), 630)
        self.assertEqual(summary["sources"]["curated_tasks_v2"]["accepted"], 600)
        self.assertEqual(summary["sources"]["curated_seed_v1"]["accepted"], 30)
        self.assertEqual(summary["category_limit_shortages"], {})


if __name__ == "__main__":
    unittest.main()
