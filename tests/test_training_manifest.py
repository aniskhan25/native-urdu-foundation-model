import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from data_pipeline.build_training_manifest import build_manifest


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


if __name__ == "__main__":
    unittest.main()
