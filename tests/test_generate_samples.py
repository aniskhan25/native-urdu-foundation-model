import unittest

from eval.generate_samples import repeated_ngram_next_tokens


class GenerateSamplesTests(unittest.TestCase):
    def test_repeated_ngram_next_tokens_blocks_seen_completion(self):
        self.assertEqual(repeated_ngram_next_tokens([1, 2, 3, 1, 2], 3), {3})

    def test_repeated_ngram_next_tokens_allows_unseen_prefix(self):
        self.assertEqual(repeated_ngram_next_tokens([1, 2, 3, 2, 1], 3), set())

    def test_repeated_ngram_next_tokens_ignores_disabled_setting(self):
        self.assertEqual(repeated_ngram_next_tokens([1, 2, 1, 2], 0), set())


if __name__ == "__main__":
    unittest.main()
