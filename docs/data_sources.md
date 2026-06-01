# Open Data Source Inventory

Last checked: 2026-05-21

This file tracks candidate data sources for Urdu pretraining and post-training.

## Important Finding: Qalb

There are two similarly named resources:

1. **Qalb Urdu LLM**: an Urdu model/paper based on continued pretraining of Llama 3.1 8B. The paper reports a 1.97B-token corpus: 1.84B Urdu tokens plus 140M English Wikipedia tokens. I found the model and paper, but not a separately released pretraining corpus.
2. **QALB / Qatar Arabic Language Bank**: an Arabic grammatical error correction corpus, not an Urdu pretraining corpus.

Use Qalb as a reference recipe, not as an assumed downloadable Urdu corpus unless the authors release the data.

Links:

- Qalb paper: https://arxiv.org/abs/2601.08141
- Qalb model: https://huggingface.co/enstazao/Qalb-1.0-8B-Instruct
- Arabic QALB shared task: https://camel.abudhabi.nyu.edu/qalb-shared-task-2015/

## Best Pretraining Candidates

| Source | Type | Approx. Urdu Scale | License / Terms | Recommended Bucket | Notes |
|---|---|---:|---|---|---|
| FineWeb2 `urd_Arab` | Cleaned CommonCrawl web text | 4.8M docs, 19.93GB raw listed | ODC-By 1.0 + CommonCrawl terms | `urdu_web` | Strong first web source. Still run Urdu normalization, LID, dedup, and safety filters. |
| HPLT `ur_cleaned` / `ur_deduplicated` | Web crawl from CommonCrawl + Internet Archive | large | Packaging CC0; source text not owned by HPLT | `urdu_web` | Good second web source. Use cleaned/dedup configs first. |
| mC4 / C4 `ur` | CommonCrawl web text | large | ODC-By + CommonCrawl terms | `urdu_web` | Older than FineWeb2; useful for diversity but more quality filtering needed. |
| CC100 Urdu | CommonCrawl/CC-Net corpus | large | varies by mirror; check selected distribution | `urdu_web` | Useful for diversity, but de-duplicate aggressively against FineWeb2/HPLT/mC4. |
| Urdu Wikipedia dump | Encyclopedia | modest | CC BY-SA / GFDL terms | `urdu_web` or held-out eval | High signal, but likely already present in web corpora. Keep a clean held-out split. |
| Urdu Wikisource | Public-domain/CC texts | modest | Wikimedia project licensing | `urdu_literature` | Good curated long-form source when Urdu subset is available. |
| Maḵẖzan Urdu corpus | Curated Urdu text | 6.26M words listed by HF mirror | Free use stated; raw republication restricted | `urdu_literature` | High-quality curated text, but honor redistribution restrictions. |
| Kaleem Art Press Urdu Literature Corpus | Urdu literary corpus | small/medium | verify dataset card before use | `urdu_literature` | Useful for literary register if license is compatible. |
| OPUS Urdu pairs | Parallel translation corpora | varies by subcorpus | per-subcorpus licenses | `english_replay` / SFT | Better for translation/SFT than base LM pretraining. |
| Alif Urdu-Instruct | Synthetic instruction data | SFT scale | public per paper/GitHub | post-training | Do not mix into base pretraining unless intentionally training instruction behavior. |

## Initial Mixture Recommendation

For the first 5B-10B token corpus:

- `urdu_web`: FineWeb2 + HPLT + mC4/C4 + CC100, sampled after cross-source dedup.
- `urdu_literature`: Maḵẖzan + Wikisource + compatible literature datasets.
- `english_replay`: English Wikipedia or a small permissive high-quality English mix.

Keep the roadmap default:

```yaml
urdu_web: 0.72
urdu_literature: 0.20
english_replay: 0.08
```

## Selected Foundation-Model Corpus v1

Use **multiple sources**. No single open Urdu dataset is enough for a from-scratch foundation model, and relying on one web crawl will overfit its crawl/filtering biases.

The selected v1 corpus is encoded in `configs/data_sources.yaml`:

| Bucket | Weight | Selected sources |
|---|---:|---|
| Urdu web | 0.72 | FineWeb2 `urd_Arab`, HPLT Urdu cleaned/dedup, mC4/C4 Urdu, Urdu Wikipedia |
| Urdu literature | 0.20 | Maḵẖzan Urdu, Urdu/Wikimedia Wikisource, Kaleem Art Press if license review passes |
| English replay | 0.08 | English Wikipedia |

Within `urdu_web`, FineWeb2 is the primary backbone, HPLT is the secondary independent crawl, mC4/C4 is diversity filler, and Urdu Wikipedia should be partially held out for evaluation before any training portion is used.

Within `urdu_literature`, Maḵẖzan is the first curated long-form source, Wikisource is the second, and Kaleem Art Press is conditional until licensing/provenance is verified.

## Current Compile Status

Local pilot compilation has been validated for:

| Source | Status | Notes |
|---|---|---|
| FineWeb2 `urd_Arab` | working | 100-document pilot compiled to `data/compiled/fineweb2_urd_arab.jsonl` |
| Maḵẖzan Urdu | working | 100-document pilot compiled to `data/compiled/makhzan_urdu.jsonl`; loader uses longest-string fallback |
| HPLT Urdu | deferred | HF dataset script currently fails in streaming mode by resolving a remote map URL as a local path |

Current pilot summary:

| Metric | Value |
|---|---:|
| Files | 2 |
| Documents | 200 |
| Normalized characters | 1,596,831 |
| Whitespace words | 327,031 |
| Sources | FineWeb2 Urdu, Maḵẖzan Urdu |

## HPC Storage Note

The local `data/` directory is for pilot artifacts only. The full corpus should be compiled on LUMI scratch or project storage, for example:

```text
/scratch/project_462000131/anisrahm/native-urdu-foundation-data
```

Recommended subdirectories:

```text
data/
  hf_cache/
  compiled/
  tokenizer/
  tokenized/
  manifests/
```

Use `slurm/compile_corpus.sh` for the full source compile, `slurm/train_tokenizer.sh` for tokenizer training, and `slurm/pretokenize.sh` for token shard creation.

## Do Not Use Blindly

- Random Hugging Face Urdu datasets with unclear provenance.
- Twitter/X-derived datasets unless the redistribution terms are reviewed.
- News archives without explicit scraping and reuse rights.
- QALB Arabic error correction data for Urdu pretraining.
- Qalb model outputs as corpus text unless synthetic-data policy explicitly allows it.

## Minimum Intake Requirements

For every imported source, record:

- source name and URL
- license and terms snapshot
- acquisition date
- source bucket
- raw file checksums
- normalization version
- token count before and after filtering
- dedup rate
