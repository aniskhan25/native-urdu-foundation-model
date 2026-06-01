# Data Card Draft

## Sources

Default training mixture:

| Bucket | Weight |
|---|---:|
| Urdu web | 0.72 |
| Urdu literature / curated long-form | 0.20 |
| English replay | 0.08 |

Tokenizer training mixture:

| Bucket | Weight |
|---|---:|
| Urdu web | 0.70 |
| Urdu literature / curated long-form | 0.25 |
| English replay | 0.05 |

## Required Metadata

Each document should preserve:

```json
{
  "doc_id": "...",
  "source": "web|literature|english_replay",
  "url": "...",
  "license": "...",
  "timestamp": "...",
  "raw_text": "...",
  "normalized_text": "...",
  "normalization_version": "custom_urdu_nfkc_then_nfc_v1",
  "quality_score": 0.0,
  "language_score_ur": 0.0,
  "dedup_hash": "..."
}
```

## Filtering

Initial drop thresholds:

| Metric | Drop if |
|---|---:|
| Document length | `< 200 chars` |
| Urdu script ratio | `< 0.65` |
| Repeated character ratio | `> 0.20` |
| Symbol ratio | `> 0.15` |
| Average line length | `< 20 chars` |

## Deduplication

Use exact normalized-text SHA256 dedup first. Add paragraph-level dedup and MinHash/SimHash near-dedup before freezing the full corpus.

