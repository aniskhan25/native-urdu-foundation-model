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
