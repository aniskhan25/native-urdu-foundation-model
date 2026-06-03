import unittest

from data_pipeline.quality_filter import passes_quality, quality_metrics


def passes(text: str) -> bool:
    return passes_quality(
        quality_metrics(text),
        min_chars=40,
        min_script_ratio=0.65,
        max_repeated_char_ratio=0.20,
        max_repeated_4gram_ratio=0.08,
        max_repeated_6gram_ratio=0.04,
        max_longest_repeated_ngram=8,
        max_symbol_ratio=0.15,
        max_url_hits=3,
        max_boilerplate_hits=2,
    )


class QualityFilterTests(unittest.TestCase):
    def test_accepts_plain_urdu_text(self):
        text = "پاکستان میں تعلیم اور تحقیق کی اہمیت بہت زیادہ ہے۔ اچھی تعلیم معاشرے کو بہتر بناتی ہے۔"
        self.assertTrue(passes(text))

    def test_rejects_repeated_phrase_loops(self):
        text = "دکان کے سامنے کھڑے ہوکر " * 20
        metrics = quality_metrics(text)
        self.assertGreater(metrics.repeated_4gram_ratio, 0.08)
        self.assertFalse(passes(text))

    def test_rejects_url_artifacts(self):
        text = "یہ ایک اردو مضمون ہے example.com www.test.pk news.org link.net page.com مزید اردو متن۔"
        metrics = quality_metrics(text)
        self.assertGreater(metrics.url_hits, 3)
        self.assertFalse(passes(text))

    def test_rejects_ui_boilerplate(self):
        text = "کتاب کا عنوان تلاش کریں اور پھر بٹن پر کلک کریں۔ اگر لنک دستیاب ہے تو download کریں۔"
        metrics = quality_metrics(text)
        self.assertGreater(metrics.boilerplate_hits, 2)
        self.assertFalse(passes(text))


if __name__ == "__main__":
    unittest.main()
