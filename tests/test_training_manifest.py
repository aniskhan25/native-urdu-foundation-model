import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from data_pipeline.build_training_manifest import build_manifest
from data_pipeline.build_training_manifest import bucket_for_source


class TrainingManifestTests(unittest.TestCase):
    def test_build_manifest_from_tokenized_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "fineweb2_urd_arab-00000.bin"
            shard.write_bytes(b"\x00\x01")
            metadata = root / "fineweb2_urd_arab.json"
            metadata.write_text(
                json.dumps(
                    {
                        "tokenizer_model": "/tmp/tokenizer.model",
                        "vocab_size": 32000,
                        "dtype": "uint16",
                        "total_docs": 3,
                        "shards": [{"path": str(shard), "tokens": 2}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "manifest.json"

            with redirect_stdout(StringIO()):
                build_manifest([metadata], output)

            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_tokens"], 2)
            self.assertEqual(manifest["total_docs"], 3)
            self.assertEqual(manifest["sources"][0]["source"], "fineweb2_urd_arab")
            self.assertEqual(manifest["sources"][0]["bucket"], "urdu_web")

    def test_bucket_mapping_for_expansion_sources(self):
        self.assertEqual(bucket_for_source("fineweb2_urd_arab_extra"), "urdu_web")
        self.assertEqual(bucket_for_source("mc4_ur"), "urdu_web")
        self.assertEqual(bucket_for_source("wikimedia_wikisource_ur"), "urdu_literature")


if __name__ == "__main__":
    unittest.main()
