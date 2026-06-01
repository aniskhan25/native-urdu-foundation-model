#!/bin/bash
#SBATCH --job-name=urdu-tokenizer
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=project_462000131

set -euo pipefail

PROJECT_ID="project_462000131"
USER_NAME="${USER:-anisrahm}"
REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
COMPILED_DIR="${COMPILED_DIR:-${DATA_ROOT}/compiled}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-${DATA_ROOT}/tokenizer}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${TOKENIZER_ROOT}"
cd "${REPO_DIR}"

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
fi

python tokenizer/train_urdu_bpe_tokenizer.py \
  --input "${COMPILED_DIR}" \
  --model-prefix "${TOKENIZER_ROOT}/urdu_bpe_32k" \
  --vocab-size 32000 \
  --max-lines "${TOKENIZER_MAX_LINES:-2000000}" \
  --min-chars 50
