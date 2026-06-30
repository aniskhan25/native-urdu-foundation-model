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

## Rejected: SFT Corpus V1

The reviewed v1 corpus contained 5,412 examples: 4,984 synthetic Traversaal records, 398 Aya records, and 30 curated records. The full SFT run completed 161 steps and reached its best validation loss of `2.0531` at step 150.

Held-out sampled and greedy generation both failed effectively 12 of 12 prompts. The model learned surface templates such as `حل:` and generic safety language but failed arithmetic, translation, correction, summarization, code-switching, and story generation. Checkpoints in `runs/700m_sft_corpus_v1` are rejected and must not be used as release or continuation checkpoints.

## In Progress: Balanced SFT Corpus V2

Goal: a smaller diagnostic corpus with explicit task balance and auditable references before any further SFT training.

V2 policy:

- Exclude Aya because manual review found stale news, SEO-style prompts, and unreliable factual answers.
- Exclude Traversaal open-domain QA because oversampled review found a material factual-error rate.
- Keep all 30 project-curated records.
- Cap reasoning and translation at 240 examples each.
- Cap generation and classification at 200 examples each.
- Cap sentiment at 160 examples and ethics at 80 examples.
- Reject responses over 500 characters and prompts over 300 characters.
- Emit a deterministic 120-record review sample spanning every source/category group.
- Fail compilation if any configured category quota cannot be filled.

Projected output is 1,150 examples: 1,120 category-balanced Traversaal records and 30 curated records.

Exit criteria before creating a v2 training config:

- all category quotas filled
- no prompt, response, or held-out evaluation overlap
- manual acceptance of the 120-record stratified review sample
- no malformed arithmetic, mistranslation, task mismatch, boilerplate refusal, or repeated text in that sample

Only after these gates pass should a short controlled v2 training configuration be added. It must start from `runs/700m_clean_continue_v1/step_000432.pt`, not any v1 SFT checkpoint.

## Later

- Expand base pretraining toward the original 5B–10B token target
- Add native Urdu literary-knowledge and multi-step math instruction data
- Build a larger human-reviewed SFT set
- Add preference tuning only after SFT quality is demonstrated on held-out prompts
