# Roadmap

## Stage A: Corpus Pilot

Goal: 50M-100M clean tokens.

Deliverables:

- Normalizer
- Script-level LID
- Exact dedup
- Heuristic quality filter
- Tokenizer v0
- Tiny 50M-150M parameter training run

Success criteria:

- No tokenizer `<unk>` issues
- Acceptable fertility
- Loss decreases normally
- Small-scale Urdu generations are coherent

## Stage B: Tokenizer Freeze

Train the final 32k BPE tokenizer with a 70/25/5 tokenizer mixture:

- 70% Urdu web
- 25% Urdu literature
- 5% English replay

Freeze:

- `urdu_bpe_32k.model`
- `urdu_bpe_32k.vocab`
- normalization version
- special token IDs
- tokenizer hash

## Stage C: 1B-Token Dress Rehearsal

Validate:

- Throughput
- Checkpoint restart
- Loss curves
- Data sampling proportions
- Token distribution
- Evaluation harness
- ROCm kernel stability
- Multi-node scaling

## Stage D: Full Training

Run the 5B-10B token training job.

Checkpoint every 100M-250M tokens and keep:

- latest checkpoint
- best validation checkpoint
- every 1B-token milestone

## Stage E: Post-Training

After base pretraining:

- Continue pretraining on 200M-500M high-quality curated Urdu tokens
- Supervised fine-tune on 50k-500k Urdu instruction examples
- Preference tune if quality preference data exists

