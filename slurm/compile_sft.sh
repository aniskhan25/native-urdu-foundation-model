#!/bin/bash
#SBATCH --job-name=urdu-sft-data
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:00:00
#SBATCH --account=project_462000131

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
SFT_SOURCE_CONFIG="${SFT_SOURCE_CONFIG:-configs/sft_sources_v1.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/sft}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
MIN_TOTAL_RECORDS="${MIN_TOTAL_RECORDS:-5000}"

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home}"
mkdir -p "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [ -n "${MAX_RECORDS_PER_SOURCE:-}" ]; then
  EXTRA_ARGS+=(--max-records-per-source "${MAX_RECORDS_PER_SOURCE}")
fi
for source in ${SFT_SOURCES:-}; do
  EXTRA_ARGS+=(--source "${source}")
done

cd "${REPO_DIR}"

singularity run \
  "${SIF}" \
  python -m sft.compile_sft_corpus \
  --config "${SFT_SOURCE_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --min-total-records "${MIN_TOTAL_RECORDS}" \
  "${EXTRA_ARGS[@]}"
