#!/bin/bash
#SBATCH --job-name=urdu-tokenize
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --account=project_462000131

set -euo pipefail

module use /appl/local/csc/modulefiles/

PROJECT_ID="project_462000131"
USER_NAME="${USER:-anisrahm}"
REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
COMPILED_DIR="${COMPILED_DIR:-${DATA_ROOT}/compiled}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${DATA_ROOT}/tokenizer/urdu_bpe_32k.model}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${DATA_ROOT}/tokenized}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
CONTAINER_BINDS="${CONTAINER_BINDS:-/pfs,/users,/projappl,/scratch,/project,/flash}"

mkdir -p "${TOKENIZED_ROOT}"
cd "${REPO_DIR}"

if [ ! -f "${CONTAINER_IMAGE}" ]; then
  echo "Missing container image: ${CONTAINER_IMAGE}" >&2
  exit 1
fi

singularity exec \
  --bind="${CONTAINER_BINDS}" \
  "${CONTAINER_IMAGE}" \
  python data_pipeline/tokenize_shards.py \
  --input "${COMPILED_DIR}/fineweb2_urd_arab.jsonl" \
  --output-prefix "${TOKENIZED_ROOT}/fineweb2_urd_arab" \
  --model "${TOKENIZER_MODEL}" \
  --text-key normalized_text \
  --shard-tokens 134217728

singularity exec \
  --bind="${CONTAINER_BINDS}" \
  "${CONTAINER_IMAGE}" \
  python data_pipeline/tokenize_shards.py \
  --input "${COMPILED_DIR}/makhzan_urdu.jsonl" \
  --output-prefix "${TOKENIZED_ROOT}/makhzan_urdu" \
  --model "${TOKENIZER_MODEL}" \
  --text-key normalized_text \
  --shard-tokens 134217728
