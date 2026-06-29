# Roadmap

## Completed: Base Pipeline

- Urdu normalization, language filtering, exact deduplication, and quality filtering
- Frozen 32k SentencePiece BPE tokenizer
- Binary token shards and weighted source manifests
- LLaMA-style decoder, BF16/FSDP training, checkpoint resume, and validation
- LUMI container launch and multi-node scaling
- Fixed-prompt generation, repetition controls, and sample scoring

## Completed: 700M Base Model

- 685M-token dress rehearsal
- 2.43B-token expanded pretraining run
- 453M-token strict-clean continuation run
- Final clean base checkpoint:
  `/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_clean_continue_v1/step_000432.pt`

The clean continuation removed URL/UI artifacts. The recommended decoding preset removes most repeated n-gram loops, but the base model still lacks reliable instruction following, arithmetic, and code-switching.

## Completed: SFT Engineering Validation

- Response-only SFT labels using the Urdu `ہدایت:` / `جواب:` template
- BF16/FSDP SFT training, validation, checkpointing, and resume
- SFT-aware generation using the same prompt template
- One-node `dev-g` smoke configuration
- Exact-overlap-protected held-out prompt suite

The 30-example seed run validated the engineering path but overfit after 20 steps. Training loss reached `0.0051` while validation loss remained `3.8874`. Held-out generation failed to generalize, so `runs/700m_sft_v1/step_000020.pt` is not a release candidate and must not be used as the base for further SFT.

## In Progress: SFT Corpus V1

Goal: 5k–20k clean, diverse prompt/response examples with a separate held-out validation split.

Initial selected sources:

- `CohereLabs/aya_dataset`: human-annotated Urdu subset, Apache-2.0
- `large-traversaal/urdu-instruct`: Urdu instruction data, CC-BY-SA-4.0
- local curated seed records for targeted math, safety, style, and code-switching coverage

Compiler requirements:

- source-specific field mapping and language filters
- Unicode normalization
- minimum prompt/response quality checks
- exact prompt/pair deduplication
- exclusion of held-out evaluation prompts
- deterministic source/category-aware train/validation split
- source, category, filtering, and license statistics

Exit criteria:

- at least 5,000 accepted examples
- no exact overlap with `eval/prompts_sft_heldout.txt`
- manual review of at least 100 random examples
- balanced coverage of QA, explanation, summarization, translation, math, correction, safety, code-switching, and creative writing

## Next: SFT V1 Training

Start again from the clean base checkpoint, not the overfit smoke checkpoint.

- Run a one-node `dev-g` data/checkpoint smoke test
- Train for 1–2 epochs with validation every 100 steps
- Compare base and SFT checkpoints on the held-out suite
- Reject runs with falling training loss and worsening held-out loss
- Keep the base-model decoding preset fixed during comparisons

## Later

- Expand base pretraining toward the original 5B–10B token target
- Add native Urdu literary-knowledge and multi-step math instruction data
- Build a larger human-reviewed SFT set
- Add preference tuning only after SFT quality is demonstrated on held-out prompts
