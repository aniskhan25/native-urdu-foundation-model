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

## Completed: Task-First SFT Corpus V2

Goal: a smaller diagnostic corpus with explicit task balance and auditable references before any further SFT training.

The first balanced candidate compiled in job `19619224` with 1,150 records, but its 120-record review sample failed. It still contained malformed synthetic arithmetic, awkward translations, task mismatches, and only two or three examples for several held-out capabilities. That candidate must not be trained.

Replacement v2 policy:

- Use no external or benchmark records.
- Keep all 30 project-curated records.
- Generate 240 arithmetic examples with answers calculated directly by code.
- Generate 80 bidirectional Urdu-English translations from audited sentence templates.
- Generate 60 examples each for summarization, Urdu correction, code-switching, and constrained stories.
- Generate 40 evidence-aware uncertainty examples.
- Reject responses over 500 characters and prompts over 600 characters.
- Emit a deterministic 120-record review sample spanning every source/category group.
- Fail compilation if any configured category quota cannot be filled.

Job `19620960` compiled the expected 630 examples: 600 deterministic task records and 30 curated records, split into 570 training and 60 validation examples. Independent verification found no duplicate prompts or responses, no train/validation overlap, no held-out overlap, and an exact match with the local generator contract.

Exit criteria before creating a v2 training config:

- all category quotas filled
- no prompt, response, or held-out evaluation overlap
- manual acceptance of the 120-record stratified review sample
- no malformed arithmetic, mistranslation, task mismatch, boilerplate refusal, or repeated text in that sample

All corpus gates passed. The controlled v2 training config starts from `runs/700m_clean_continue_v1/step_000432.pt`, not any v1 SFT checkpoint.

## Controlled SFT V2 Diagnostic: Rejected

- Four `dev-g` nodes with 32 logical GPUs; the one-node attempt exhausted memory with only eight FSDP shards
- Global batch 32 for 36 optimizer steps over two epochs
- Peak learning rate `1e-5`
- Validation every 5 steps
- Isolated output directory `runs/700m_sft_balanced_v2`
- Job `19643795` completed all 36 steps; validation loss reached `0.5144` at step 35 and ended at `0.5235`
- Held-out sampled decoding avoided repetition but failed both arithmetic prompts, translation, grammar correction, summarization, code-switching, and constrained story generation
- Greedy decoding copied prompts or repeated phrases and performed worse
- The only clear held-out success was the evidence-uncertainty response
- The train/validation split mixed examples from the same generated template families, so validation loss measured template interpolation rather than task generalization
- Do not promote or continue training from `runs/700m_sft_balanced_v2`

## Next: SFT V3 Corpus

- Keep the clean base checkpoint as the training start
- Define validation and test sets by held-out task/template families before compiling training data
- Replace repeated synthetic templates with a larger human-authored or individually reviewed Urdu task set
- Require exact task checks for arithmetic and constraints in addition to repetition and script-ratio metrics
- Run a short diagnostic only after the corpus passes family-overlap and manual-review gates

## Later

- Expand base pretraining toward the original 5B–10B token target
- Add native Urdu literary-knowledge and multi-step math instruction data
- Build a larger human-reviewed SFT set
- Add preference tuning only after SFT quality is demonstrated on held-out prompts
