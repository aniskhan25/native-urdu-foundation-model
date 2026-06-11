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
