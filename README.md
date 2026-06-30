# Native Urdu Foundation Model

This repository scaffolds a native Urdu decoder-only language model project for PyTorch + LUMI-G.

The initial target is a 350M-1.3B parameter model trained from scratch on 5B-10B tokens with:

- Urdu web text
- Urdu literature and curated long-form text
- A small high-quality English replay buffer

The highest-risk parts are data quality, Unicode normalization, deduplication, and tokenizer fertility. The code in this repository starts there.

## Repository Layout

```text
configs/                 Training and tokenizer configuration
data_pipeline/           Normalization, filtering, deduplication, sharding utilities
docs/                    Data/model/normalization documentation
eval/                    Evaluation placeholders and contamination checks
slurm/                   LUMI Slurm job templates
tokenizer/               SentencePiece BPE tokenizer training
training/                Minimal decoder model, shard sampler, and FSDP training entrypoint
tests/                   Unit tests for core pipeline behavior
```

## Quick Start

Install local dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Normalize plain text:

```bash
python data_pipeline/normalize_urdu.py \
  --input raw.txt \
  --output normalized.txt
```

Normalize JSONL while preserving raw text:

```bash
python data_pipeline/normalize_urdu.py \
  --input raw.jsonl \
  --output normalized.jsonl \
  --jsonl \
  --text-key raw_text \
  --normalized-key normalized_text
```

Compile a pilot corpus shard from the selected source plan:

```bash
python -m data_pipeline.compile_corpus \
  --source fineweb2_urd_arab \
  --max-docs-per-source 1000 \
  --output-dir data/compiled \
  --force-exit
```

Compile multiple selected sources:

```bash
python -m data_pipeline.compile_corpus \
  --source fineweb2_urd_arab \
  --source makhzan_urdu \
  --max-docs-per-source 10000 \
  --max-scanned-per-source 20000 \
  --output-dir data/compiled \
  --force-exit
```

Summarize compiled shards:

```bash
python -m data_pipeline.summarize_corpus data/compiled/*.jsonl
```

Train a 32k Urdu BPE tokenizer:

```bash
python tokenizer/train_urdu_bpe_tokenizer.py \
  --input /path/to/clean_urdu_web /path/to/clean_urdu_literature \
  --model-prefix tokenizer/urdu_bpe_32k \
  --vocab-size 32000 \
  --max-lines 2000000
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Default Target

The default project target is encoded in [configs/urdu_1_3b.yaml](configs/urdu_1_3b.yaml):

- 1.3B parameter LLaMA-style decoder-only transformer
- 32k SentencePiece BPE tokenizer
- 4096 token context
- BF16 AdamW training
- FSDP full-shard on LUMI-G
- 72% Urdu web, 20% Urdu literature, 8% English replay

## LUMI Scratch Storage

The local `data/` directory is for pilots only. Full corpus compilation should write to LUMI scratch or project storage, for example:

```bash
/scratch/project_462000131/anisrahm/native-urdu-foundation-data
```

Suggested LUMI flow:

```bash
export HF_TOKEN="hf_..."
sbatch slurm/check_container_requirements.sh
sbatch slurm/check_gpu_container.sh
sbatch --export=ALL slurm/compile_corpus.sh
sbatch slurm/train_tokenizer.sh
sbatch slurm/pretokenize.sh
sbatch slurm/build_manifest.sh
sbatch slurm/preflight_dress_rehearsal.sh
```

Do not commit the Hugging Face token or add it to the Slurm scripts. For repeated use, keep it in a private file and source it before submitting the corpus job:

```bash
mkdir -p ~/.config/huggingface
chmod 700 ~/.config/huggingface
printf '%s\n' 'export HF_TOKEN="hf_..."' > ~/.config/huggingface/token.env
chmod 600 ~/.config/huggingface/token.env

source ~/.config/huggingface/token.env
sbatch --export=ALL slurm/compile_corpus.sh
```

For a stricter cleanup compile, lower the artifact thresholds:

```bash
export CORPUS_SOURCES="fineweb2_urd_arab_extra"
export OUTPUT_DIR=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/compiled_clean_v1
export SKIP_DOCS_PER_SOURCE=4000000
export MAX_DOCS_PER_SOURCE=1000000
export MAX_SCANNED_PER_SOURCE=3000000
export MAX_REPEATED_4GRAM_RATIO=0.04
export MAX_REPEATED_6GRAM_RATIO=0.02
export MAX_LONGEST_REPEATED_NGRAM=6
export MAX_URL_HITS=1
export MAX_BOILERPLATE_HITS=0
sbatch --export=ALL slurm/compile_corpus.sh
```

Current Slurm defaults are:

```text
REPO_DIR=/scratch/project_462000131/anisrahm/native-urdu-foundation-model
DATA_ROOT=/scratch/project_462000131/anisrahm/native-urdu-foundation-data
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
```

The LUMI Slurm jobs follow the official LUMI AI Guide multi-GPU pattern: `module purge`, load `lumi-aif-singularity-bindings`, run the LAIF SIF with `singularity run`, set a Slurm-job-derived `MASTER_PORT`, and launch distributed training with `python -m torch.distributed.run --numa-binding=exclusive`. Run `slurm/check_container_requirements.sh` first for Python packages and `slurm/check_gpu_container.sh` before training to verify MI250X visibility.

Prepare the current 685M-token dress rehearsal manifest:

```bash
sbatch slurm/build_manifest.sh
sbatch slurm/preflight_dress_rehearsal.sh
```

The dress rehearsal config is `configs/urdu_dress_rehearsal.yaml`. It points at the current tokenized FineWeb2 and Maḵẖzan shards and uses a smaller 350M-parameter model target for pipeline validation.

For a minimal GPU smoke test before a full rehearsal:

```bash
MAX_STEPS=2 sbatch slurm/train_dress_rehearsal.sh
```

Resume from the most recent checkpoint:

```bash
RESUME=latest sbatch slurm/train_dress_rehearsal.sh
```

If the smoke test completes, submit the configured rehearsal:

```bash
sbatch slurm/train_dress_rehearsal.sh
```

Training checkpoints are written under the config `infrastructure.output_dir`, with `latest.json` pointing at the most recent checkpoint. Checkpoints include model and optimizer state, and old `step_*.pt` files are pruned according to `training.keep_last_checkpoints`.

Run the 700M pilot on the current 685M-token corpus:

```bash
CONFIG=configs/urdu_700m_pilot.yaml sbatch slurm/preflight_dress_rehearsal.sh
CONFIG=configs/urdu_700m_pilot.yaml MAX_STEPS=20 sbatch slurm/train_dress_rehearsal.sh
CONFIG=configs/urdu_700m_pilot.yaml sbatch slurm/train_dress_rehearsal.sh
```

## Corpus Expansion

Compile the next FineWeb2 Urdu continuation shard while training runs. This skips the first 1M kept documents already used in the rehearsal and writes the next kept documents to a separate compiled directory:

```bash
export CORPUS_SOURCES="fineweb2_urd_arab_extra"
export OUTPUT_DIR=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/compiled_expansion_v1
export SKIP_DOCS_PER_SOURCE=1000000
export MAX_DOCS_PER_SOURCE=3000000
export MAX_SCANNED_PER_SOURCE=6000000
sbatch --export=ALL slurm/compile_corpus.sh
```

Tokenize the expansion shard with the frozen tokenizer:

```bash
export CORPUS_SOURCES="fineweb2_urd_arab_extra"
export COMPILED_DIR=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/compiled_expansion_v1
sbatch --export=ALL slurm/pretokenize.sh
```

Build an expanded manifest that combines the original rehearsal shards and the new continuation shard:

```bash
export CORPUS_SOURCES="fineweb2_urd_arab makhzan_urdu fineweb2_urd_arab_extra"
export MANIFEST_NAME=expanded_v1_manifest.json
sbatch --export=ALL slurm/build_manifest.sh
```

Train the 700M model on expanded v1:

```bash
CONFIG=configs/urdu_700m_expanded_v1.yaml sbatch slurm/preflight_dress_rehearsal.sh
CONFIG=configs/urdu_700m_expanded_v1.yaml MAX_STEPS=20 sbatch slurm/train_dress_rehearsal.sh
CONFIG=configs/urdu_700m_expanded_v1.yaml sbatch slurm/train_dress_rehearsal.sh
```

Resume the expanded run:

```bash
CONFIG=configs/urdu_700m_expanded_v1.yaml RESUME=latest sbatch slurm/train_dress_rehearsal.sh
```

Generate qualitative Urdu samples from the latest checkpoint:

```bash
CONFIG=configs/urdu_700m_expanded_v1.yaml sbatch slurm/generate_samples.sh
```

The default prompt set is `eval/prompts_urdu.txt`; samples are written to `samples.jsonl` in the run output directory.

Recommended decoding preset for base-model qualitative checks:

```bash
export CONFIG=configs/urdu_700m_clean_continue_v1.yaml
export CHECKPOINT=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_clean_continue_v1/step_000432.pt
export OUTPUT=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_clean_continue_v1/samples_t07_p085_r112_ng6_96.jsonl
export MAX_NEW_TOKENS=96
export TEMPERATURE=0.7
export TOP_P=0.85
export TOP_K=40
export REPETITION_PENALTY=1.12
export NO_REPEAT_NGRAM_SIZE=6

sbatch --export=ALL slurm/generate_samples.sh
```

This preset was selected after the clean-continuation checkpoint removed URL/UI artifacts but still repeated under plain top-p sampling. The repetition penalty and 6-gram block are decoding controls for qualitative evaluation; they do not change the trained checkpoint.

Score generated samples for repetition, web artifacts, boilerplate, Urdu ratio, prompt copying, and simple math:

```bash
RUN_DIR=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_expanded_v1 sbatch slurm/score_samples.sh
```

Score an explicitly named sample file:

```bash
export INPUT=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_clean_continue_v1/samples_t07_p085_r112_ng6_96.jsonl
export OUTPUT=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_clean_continue_v1/samples_t07_p085_r112_ng6_96.scores.jsonl
export SUMMARY_OUTPUT=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_clean_continue_v1/samples_t07_p085_r112_ng6_96.summary.json

sbatch --export=ALL slurm/score_samples.sh
```

This writes `samples.scores.jsonl` and `samples.summary.json` next to `samples.jsonl`.

## Supervised Fine-Tuning

The first SFT path expects one prompt/response pair per JSONL line:

```json
{"prompt": "سوال یا ہدایت...", "response": "مثالی جواب...", "source": "optional"}
```

Only `prompt` and `response` are required. The trainer formats each record as:

```text
ہدایت:
{prompt}

جواب:
{response}
```

Loss is applied only to the response tokens plus EOS; prompt/template and padding labels are masked with `-100`.

Place the initial SFT split on LUMI scratch:

```bash
/scratch/project_462000131/anisrahm/native-urdu-foundation-data/sft/sft_train.jsonl
/scratch/project_462000131/anisrahm/native-urdu-foundation-data/sft/sft_val.jsonl
```

Create the curated seed split:

```bash
python -m sft.prepare_seed_sft \
  --output-dir /scratch/project_462000131/anisrahm/native-urdu-foundation-data/sft
```

This seed set is only a starter for checking SFT behavior. It covers the observed base-model weak spots: repetition-prone Urdu prompts, simple math, summaries, grammar/style correction, code-switching, safety/uncertainty, and short creative writing.

Compile SFT corpus v1 from the human-annotated Urdu subset of [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), [Traversaal Urdu Instruct](https://huggingface.co/datasets/large-traversaal/urdu-instruct), and the local curated seed records. Source mappings, filters, caps, provenance, and licenses are defined in `configs/sft_sources_v1.yaml`.

Run a small compiler pilot first:

```bash
export MAX_RECORDS_PER_SOURCE=100
export MIN_TOTAL_RECORDS=100
export OUTPUT_DIR=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/sft_pilot
sbatch --export=ALL slurm/compile_sft.sh
```

Compile the full v1 target after inspecting the pilot:

```bash
unset MAX_RECORDS_PER_SOURCE
unset MIN_TOTAL_RECORDS
export OUTPUT_DIR=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/sft
sbatch --export=ALL slurm/compile_sft.sh
```

Build the smaller balanced v2 candidate corpus after the v1 held-out failure:

```bash
unset MAX_RECORDS_PER_SOURCE SFT_SOURCES
export SFT_SOURCE_CONFIG=configs/sft_sources_v2.yaml
export OUTPUT_DIR=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/sft_balanced_v2
export MIN_TOTAL_RECORDS=600

sbatch --export=ALL slurm/compile_sft.sh
```

V2 contains 600 deterministic project-generated records plus the 30 project-curated seed records. It targets arithmetic, translation, summarization, Urdu correction, code-switching, constrained stories, and evidence-aware answers. It uses no external or benchmark records. Compilation fails if any task quota cannot be filled. Inspect `sft_review_sample.jsonl`, which contains 120 deterministic examples covering every retained source/category group, before creating or running a v2 training config.

Use `SFT_SOURCE_CONFIG` to override the compiler source config. The generic `CONFIG` variable is reserved for training and generation jobs and is intentionally ignored by `compile_sft.sh`.

The compiler writes:

```text
sft_train.jsonl
sft_val.jsonl
sft_summary.json
sft_review_sample.jsonl  # when review_sample_size is configured
```

It normalizes text, applies source language filters, rejects low-quality records, removes duplicate prompts, responses, and pairs, excludes exact held-out prompt overlap, creates a deterministic source/category-aware validation split, and reports accepted counts by source, category, license, and provenance. Review `sft_summary.json` and at least 100 random training examples before starting a full SFT run.

Run the task-first v2 preflight:

```bash
export CONFIG=configs/urdu_700m_sft_balanced_v2.yaml
sbatch --export=ALL slurm/preflight_sft.sh
```

Run the controlled v2 diagnostic on four `dev-g` nodes:

```bash
export CONFIG=configs/urdu_700m_sft_balanced_v2.yaml
unset RESUME MAX_STEPS
sbatch --partition=dev-g --nodes=4 --gpus-per-node=8 --time=00:30:00 --export=ALL slurm/train_sft.sh
```

The v2 config starts from the clean base checkpoint, uses the proven 32-rank FSDP topology and a global batch of 32 for 36 optimizer steps over two epochs, and writes to `runs/700m_sft_balanced_v2`. It does not resume from either rejected SFT run.

Resume v2 only after confirming the intended checkpoint:

```bash
export CONFIG=configs/urdu_700m_sft_balanced_v2.yaml
export RESUME=latest
sbatch --partition=dev-g --nodes=4 --gpus-per-node=8 --time=00:30:00 --export=ALL slurm/train_sft.sh
```

Generate from the latest SFT checkpoint with the recommended decoding preset:

```bash
export CONFIG=configs/urdu_700m_sft_balanced_v2.yaml
export CHECKPOINT=latest
export PROMPTS=eval/prompts_sft_heldout.txt
export OUTPUT=/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_sft_balanced_v2/samples_sft_heldout.jsonl
export MAX_NEW_TOKENS=96
export TEMPERATURE=0.7
export TOP_P=0.85
export TOP_K=40
export REPETITION_PENALTY=1.12
export NO_REPEAT_NGRAM_SIZE=6

sbatch --export=ALL slurm/generate_samples.sh
```

SFT configs set `generation.prompt_template: urdu_sft`, so generation automatically applies the same `ہدایت:` / `جواب:` template used during training. Set `PROMPT_TEMPLATE=raw` only when evaluating an unformatted base checkpoint with an SFT config.

Use `eval/prompts_sft_heldout.txt` for SFT quality checks. The original prompts in `eval/prompts_urdu.txt` are included in the curated seed training records and therefore measure memorization rather than generalization after SFT.
