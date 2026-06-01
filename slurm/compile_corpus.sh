#!/bin/bash
#SBATCH --job-name=urdu-compile
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=project_462000131

set -euo pipefail

PROJECT_ID="project_462000131"
USER_NAME="${USER:-anisrahm}"
REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/compiled}"

export HF_HOME
export HF_DATASETS_CACHE
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${OUTPUT_DIR}" "${HF_DATASETS_CACHE}"
cd "${REPO_DIR}"

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
fi

python -m data_pipeline.compile_corpus \
  --source fineweb2_urd_arab \
  --source makhzan_urdu \
  --max-docs-per-source "${MAX_DOCS_PER_SOURCE:-1000000}" \
  --max-scanned-per-source "${MAX_SCANNED_PER_SOURCE:-1500000}" \
  --output-dir "${OUTPUT_DIR}" \
  --force-exit

python -m data_pipeline.summarize_corpus "${OUTPUT_DIR}"/*.jsonl > "${OUTPUT_DIR}/summary.json"
