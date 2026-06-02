import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.dataset import RandomTokenSampler


class RandomTokenSamplerTests(unittest.TestCase):
    def test_samples_fixed_length_windows_from_weighted_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "source-00000.bin"
            np.arange(32, dtype=np.uint16).tofile(shard)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source": "source",
                                "bucket": "urdu_web",
                                "dtype": "uint16",
                                "shards": [{"path": str(shard), "tokens": 32}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            sampler = RandomTokenSampler(manifest, {"source": 1.0}, seq_len=8, seed=123)
            batch, counts = sampler.batch(4)

            self.assertEqual(batch.shape, (4, 9))
            self.assertEqual(batch.dtype, np.int64)
            self.assertEqual(counts, {"source": 4})
            self.assertEqual(sampler.source_buckets(), {"source": "urdu_web"})

    def test_ignores_zero_weight_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.bin"
            inactive = root / "inactive.bin"
            np.arange(32, dtype=np.uint16).tofile(active)
            np.arange(32, dtype=np.uint16).tofile(inactive)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source": "active",
                                "dtype": "uint16",
                                "shards": [{"path": str(active), "tokens": 32}],
                            },
                            {
                                "source": "inactive",
                                "dtype": "uint16",
                                "shards": [{"path": str(inactive), "tokens": 32}],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            sampler = RandomTokenSampler(manifest, {"active": 1.0, "inactive": 0.0}, seq_len=8, seed=123)
            _, counts = sampler.batch(3)

            self.assertEqual(counts, {"active": 3})


if __name__ == "__main__":
    unittest.main()
