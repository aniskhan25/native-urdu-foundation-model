# Roadmap

## Program Objective

The project produces two distinct checkpoints:

1. An Urdu foundation model evaluated as a next-token model.
2. An instruction model adapted from an accepted foundation checkpoint.

Supervised fine-tuning changes model behavior. It is not used to compensate for missing language, knowledge, or reasoning capability in the foundation checkpoint. Each phase below has an explicit gate; later GPU work does not start until that gate passes.

## Current State

The data pipeline, frozen 32k tokenizer, binary training shards, LLaMA-style model, BF16/FSDP training, checkpoint resume, validation, LUMI container launch, and generation tooling are implemented.

The accepted baseline is:

```text
/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_clean_continue_v1/step_000432.pt
```

The program history includes a 350M-model dress rehearsal. The accepted checkpoint lineage is the 700M-model, 685M-token pilot followed by the 2.43B-token expanded run and 453M-token strict-clean continuation. The base model is fluent enough for continued development but lacks reliable instruction following, arithmetic, translation, and code-switching.

All existing SFT checkpoints are rejected:

- The 30-example engineering run overfit: training loss `0.0051`, validation loss `3.8874`.
- SFT v1 used 5,412 mostly synthetic records and failed effectively 12 of 12 held-out prompts.
- SFT v2 used 630 cleaner records but repeated seven generated template families. Job `19643795` reached validation loss `0.5144`, yet held-out generation failed arithmetic, translation, correction, summarization, code-switching, and story constraints. The random record split leaked template families across train and validation.

Rejected SFT checkpoints are retained only as negative baselines. They must not be released or used as continuation checkpoints.

## Phase 0: Freeze And Reproduce

Status: in progress.

Work:

- Define the accepted clean base, tokenizer, normalization policy, training lineage, and rejected checkpoints in one machine-readable specification.
- Generate a lockfile on LUMI containing artifact sizes and SHA-256 hashes plus the exact Git commit.
- Maintain a run registry with hypothesis, inputs, data, job IDs, metrics, and decision.
- Document baseline generation and lock verification commands.

Deliverables:

- `configs/baseline_v1.yaml`
- `scripts/freeze_baseline.py`
- `slurm/freeze_baseline.sh`
- `docs/run_registry.md`
- `${DATA_ROOT}/manifests/baseline_v1.lock.json` generated on LUMI

Gate:

- The repository worktree is clean when the lock is generated.
- Every required artifact exists and has a recorded SHA-256 hash.
- A second verification run succeeds without changing the lock.
- The clean base can be loaded and used for fixed-prompt generation from the recorded commit.

## Phase 1: Evaluation V1

Status: pending Phase 0.

Build 300-500 independently authored Urdu items covering arithmetic, translation, grammar correction, reading comprehension, summarization, evidence handling, instruction constraints, code-switching, and open generation. Keep base-model completion probes separate from instruction-formatted prompts.

Each record includes `id`, `task`, `family_id`, `prompt`, expected result or rubric, scorer, split, and provenance. Deterministic scorers handle numeric answers, accepted corrections, extraction, and constraints. Human rubrics cover translation, summarization, code-switching, and open generation.

Deliverables:

- `eval/benchmark_v1.jsonl`
- `eval/evaluate_tasks.py`
- Scorer unit tests
- A human-review artifact for subjective tasks

Gate:

- All deterministic scorers pass reviewed fixtures.
- No benchmark prompt, answer, or close task family appears in training data.
- Test answers remain excluded from SFT compilation.

## Phase 2: Baseline Characterization

Status: pending Phase 1.

Evaluate the clean base and rejected SFT v1/v2 checkpoints with identical prompts and decoding settings. Record per-task accuracy, constraint compliance, repetition, prompt copying, Urdu fluency, and per-domain language-model loss.

Deliverable: `docs/baseline_report.md` with fixed acceptance thresholds for future runs.

Gate: baseline metrics are reproducible and every future experiment has a declared comparison checkpoint and threshold.

## Phase 3: Pretraining Corpus Audit

Status: pending Phase 0; may run in parallel with Phases 1-2.

Reconcile the corpus manifests with tokens actually consumed by every base checkpoint. Report unique and replayed tokens, source/domain proportions, literature and English exposure, exact and near-duplicate rates, licenses, quality distributions, tokenizer fertility, and evaluation contamination.

Deliverables:

- `data_pipeline/audit_training_corpus.py`
- `corpus_audit.json`
- `docs/corpus_audit.md`

Gate: checkpoint exposure totals reconcile with manifests and the report distinguishes unique corpus size from repeated training exposure.

## Phase 4: Corpus Expansion

Status: pending Phase 3.

Acquire and compile a source-audited corpus targeting the following new-token mixture:

| Bucket | Target |
| --- | ---: |
| Diverse Urdu web | 65% |
| Curated Urdu educational and literary text | 20% |
| High-quality English replay | 10% |
| Structured Urdu math, science, and reasoning text | 5% |

Ratios are not met by heavily replaying scarce documents. New data is globally deduplicated against previous training data. Entire sources or domains are held out before tokenization. Synthetic structured text remains below 5% unless individually reviewed.

Gate: every source has approved provenance, license, volume, filtering, deduplication, and contamination records; the compiled pilot matches the approved mixture.

## Phase 5: Continued-Pretraining Pilot

Status: pending Phases 2 and 4.

Continue the clean base on a representative 250M-500M-token pilot. Keep architecture and tokenizer fixed, use the proven distributed topology, save checkpoints every 100M tokens, and evaluate each checkpoint against the frozen baseline.

Gate:

- Most Urdu validation domains improve.
- No important domain regresses beyond the threshold fixed in Phase 2.
- English replay improves without material Urdu degradation.
- Repetition and prompt copying do not increase.
- Capability probes improve or remain statistically stable.

If the pilot fails, revise the corpus rather than decoding parameters.

## Phase 6: Complete Foundation Training

Status: pending Phase 5.

Continue the accepted mixture toward 8B-10B total model exposure. Save checkpoints every 250M tokens, evaluate every 500M tokens, and stop if held-out loss plateaus or important domains regress. Preserve model-only checkpoints separately from optimizer checkpoints.

Gate: stable Urdu generation, improved held-out Urdu loss, improved reading and factual completion, no tokenizer/repetition pathology, and a reviewed model card describing the limitations of the 729M model.

## Phase 7: SFT V3 Corpus

Status: pending Phase 6.

Build 2,000-5,000 gold human-authored or individually reviewed examples, then expand to 10,000-20,000 total examples only through controlled augmentation. Split source, topic, and template families before augmentation. Keep gold development and test sets isolated.

Target distribution:

| Category | Share |
| --- | ---: |
| General explanation and QA | 20% |
| Reading comprehension | 15% |
| Summarization | 10% |
| Translation | 10% |
| Grammar and rewriting | 10% |
| Math and structured reasoning | 10% |
| Extraction and constraints | 10% |
| Code-switching | 5% |
| Evidence and uncertainty | 5% |
| Safety | 5% |

Gate: no family overlap across splits, no benchmark contamination, complete provenance, and manual acceptance of the gold set plus an agreed defect threshold for augmented data.

## Phase 8: SFT Diagnostic And Full Run

Status: pending Phase 7.

Start from the accepted foundation checkpoint. Run a short diagnostic on the proven four-node topology, save multiple checkpoints within the first epoch, and compare them with the foundation baseline on Evaluation V1. Continue for at most one or two epochs only after the diagnostic passes.

Initial acceptance targets:

- Simple arithmetic at least 80%.
- Grammar correction at least 80%.
- Instruction-constraint compliance at least 90%.
- Material translation and comprehension improvement.
- No severe greedy-decoding loops.
- No material foundation-language regression.

## Phase 9: Release

Status: pending Phase 8.

Publish separate base and instruction checkpoints with model-only weights, tokenizer, normalization policy, model card, data card, evaluation report, decoding preset, provenance, and known limitations.

## Execution Order

Phases 1 and 3 may run in parallel after Phase 0. Corpus sourcing may begin while Evaluation V1 is reviewed, but no continued-pretraining job starts before Phases 2 and 4 pass. No SFT run starts before the foundation checkpoint and SFT v3 corpus independently pass their gates.
