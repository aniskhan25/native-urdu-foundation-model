#!/bin/bash
#SBATCH --job-name=urdu-tokenize
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --account=project_462000131

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

PROJECT_ID="project_462000131"
USER_NAME="${USER:-anisrahm}"
REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
COMPILED_DIR="${COMPILED_DIR:-${DATA_ROOT}/compiled}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${DATA_ROOT}/tokenizer/urdu_bpe_32k.model}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${DATA_ROOT}/tokenized}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
CORPUS_SOURCES="${CORPUS_SOURCES:-fineweb2_urd_arab makhzan_urdu}"

mkdir -p "${TOKENIZED_ROOT}"
cd "${REPO_DIR}"

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

read -r -a SOURCES <<< "${CORPUS_SOURCES}"
for SOURCE in "${SOURCES[@]}"; do
  singularity run \
    "${SIF}" \
    python data_pipeline/tokenize_shards.py \
    --input "${COMPILED_DIR}/${SOURCE}.jsonl" \
    --output-prefix "${TOKENIZED_ROOT}/${SOURCE}" \
    --model "${TOKENIZER_MODEL}" \
    --text-key normalized_text \
    --shard-tokens 134217728
done
