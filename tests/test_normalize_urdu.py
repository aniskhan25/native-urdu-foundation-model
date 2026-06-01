import unittest

from data_pipeline.normalize_urdu import normalize_urdu, text_stats


class NormalizeUrduTests(unittest.TestCase):
    def test_maps_arabic_variants_to_urdu_codepoints(self):
        self.assertEqual(normalize_urdu("كيا يه كتاب ہے؟"), "کیا یہ کتاب ہے؟")

    def test_composes_common_decomposed_forms_before_diacritic_removal(self):
        self.assertEqual(normalize_urdu("ا\u0653 و\u0654 ی\u0654"), "آ ؤ ئ")

    def test_normalizes_digits_and_whitespace(self):
        self.assertEqual(normalize_urdu("  سال ۲۰۲۶\nہے  "), "سال 2026 ہے")

    def test_removes_bidi_controls_and_tatweel(self):
        self.assertEqual(normalize_urdu("ا\u200fر\u0640دو"), "اردو")

    def test_script_ratio(self):
        stats = text_stats(normalize_urdu("پاکستان ایک خوبصورت ملک ہے۔"))
        self.assertGreater(stats.urdu_script_ratio, 0.95)


if __name__ == "__main__":
    unittest.main()

