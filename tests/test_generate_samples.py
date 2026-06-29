import unittest

from eval.generate_samples import prompt_for_model
from eval.generate_samples import repeated_ngram_next_tokens


class GenerateSamplesTests(unittest.TestCase):
    def test_raw_prompt_template_leaves_prompt_unchanged(self):
        self.assertEqual(prompt_for_model("سوال", "raw"), "سوال")

    def test_urdu_sft_prompt_template_matches_training_format(self):
        self.assertEqual(prompt_for_model("سوال", "urdu_sft"), "ہدایت:\nسوال\n\nجواب:\n")

    def test_rejects_unknown_prompt_template(self):
        with self.assertRaisesRegex(ValueError, "Unknown prompt template"):
            prompt_for_model("سوال", "unknown")

    def test_repeated_ngram_next_tokens_blocks_seen_completion(self):
        self.assertEqual(repeated_ngram_next_tokens([1, 2, 3, 1, 2], 3), {3})

    def test_repeated_ngram_next_tokens_allows_unseen_prefix(self):
        self.assertEqual(repeated_ngram_next_tokens([1, 2, 3, 2, 1], 3), set())

    def test_repeated_ngram_next_tokens_ignores_disabled_setting(self):
        self.assertEqual(repeated_ngram_next_tokens([1, 2, 1, 2], 0), set())


if __name__ == "__main__":
    unittest.main()
