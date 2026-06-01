import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline.compile_corpus import strip_xml_text
from tokenizer.train_urdu_bpe_tokenizer import iter_training_texts


class PipelineExtractionTests(unittest.TestCase):
    def test_tokenizer_reads_jsonl_text_field_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "doc_id": "metadata-should-not-be-read",
                        "normalized_text": "پاکستان ایک خوبصورت ملک ہے۔",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                list(iter_training_texts([str(path)], text_key="normalized_text")),
                ["پاکستان ایک خوبصورت ملک ہے۔"],
            )

    def test_tokenizer_fails_when_jsonl_text_field_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            path.write_text(json.dumps({"text": "wrong field"}) + "\n", encoding="utf-8")

            with self.assertRaises(KeyError):
                list(iter_training_texts([str(path)], text_key="normalized_text"))

    def test_strip_xml_text_uses_body_without_tags(self):
        xml = """
        <document>
          <meta><title>عنوان</title></meta>
          <body><p>اصل متن</p><p>دوسرا جملہ</p></body>
        </document>
        """
        text = strip_xml_text(xml)
        self.assertIn("اصل متن", text)
        self.assertIn("دوسرا جملہ", text)
        self.assertNotIn("<body>", text)
        self.assertNotIn("عنوان", text)


if __name__ == "__main__":
    unittest.main()
