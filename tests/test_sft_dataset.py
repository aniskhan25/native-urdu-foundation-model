import json
import tempfile
import unittest
from pathlib import Path

from training.sft_dataset import IGNORE_INDEX
from training.sft_dataset import SftBatchSampler
from training.sft_dataset import SftDataset
from training.sft_dataset import encode_sft_record
from training.sft_dataset import load_sft_records


class FakeTokenizer:
    def encode(self, text, out_type=int):
        return [ord(char) for char in text]

    def eos_id(self):
        return 2


class SftDatasetTests(unittest.TestCase):
    def test_masks_prompt_and_trains_on_response(self):
        example = encode_sft_record({"prompt": "سوال", "response": "جواب"}, FakeTokenizer(), 64)

        label_values = [int(value) for value in example.labels if int(value) != IGNORE_INDEX]
        self.assertEqual(label_values, [ord("ج"), ord("و"), ord("ا"), ord("ب"), 2])

    def test_pads_labels_with_ignore_index(self):
        example = encode_sft_record({"prompt": "الف", "response": "ب"}, FakeTokenizer(), 32)

        self.assertEqual(example.input_ids.shape, (32,))
        self.assertEqual(example.labels.shape, (32,))
        self.assertEqual(example.labels[-1], IGNORE_INDEX)

    def test_truncates_from_left_to_keep_response_tail(self):
        example = encode_sft_record({"prompt": "الف" * 50, "response": "جواب"}, FakeTokenizer(), 8)

        label_values = [int(value) for value in example.labels if int(value) != IGNORE_INDEX]
        self.assertEqual(label_values[-5:], [ord("ج"), ord("و"), ord("ا"), ord("ب"), 2])

    def test_loads_prompt_response_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            path.write_text(json.dumps({"prompt": "سوال", "response": "جواب"}, ensure_ascii=False), encoding="utf-8")

            records = load_sft_records(path)

            self.assertEqual(records, [{"prompt": "سوال", "response": "جواب"}])

    def test_rejects_empty_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            path.write_text(json.dumps({"prompt": "سوال", "response": ""}, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires non-empty prompt and response"):
                load_sft_records(path)

    def test_batches_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "سوال 1", "response": "جواب 1"}, ensure_ascii=False),
                        json.dumps({"prompt": "سوال 2", "response": "جواب 2"}, ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )
            dataset = SftDataset(path, FakeTokenizer(), 32)

            input_ids, labels = dataset.batch([0, 1])

            self.assertEqual(input_ids.shape, (2, 32))
            self.assertEqual(labels.shape, (2, 32))

    def test_batch_sampler_wraps_epoch(self):
        sampler = SftBatchSampler(dataset_size=3, batch_size=2, seed=1)

        self.assertEqual(len(sampler.batch_indices()), 2)
        self.assertEqual(len(sampler.batch_indices()), 2)


if __name__ == "__main__":
    unittest.main()
