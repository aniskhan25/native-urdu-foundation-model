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

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

PROJECT_ID="project_462000131"
USER_NAME="${USER:-anisrahm}"
REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/compiled}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
CORPUS_SOURCES="${CORPUS_SOURCES:-fineweb2_urd_arab makhzan_urdu}"

export HF_HOME
export HF_DATASETS_CACHE
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${OUTPUT_DIR}" "${HF_DATASETS_CACHE}"
cd "${REPO_DIR}"

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

read -r -a SOURCES <<< "${CORPUS_SOURCES}"
SOURCE_ARGS=()
for SOURCE in "${SOURCES[@]}"; do
  SOURCE_ARGS+=(--source "${SOURCE}")
done

singularity run \
  "${SIF}" \
  python -m data_pipeline.compile_corpus \
  "${SOURCE_ARGS[@]}" \
  --max-docs-per-source "${MAX_DOCS_PER_SOURCE:-1000000}" \
  --skip-docs-per-source "${SKIP_DOCS_PER_SOURCE:-0}" \
  --max-scanned-per-source "${MAX_SCANNED_PER_SOURCE:-1500000}" \
  --max-repeated-4gram-ratio "${MAX_REPEATED_4GRAM_RATIO:-0.08}" \
  --max-repeated-6gram-ratio "${MAX_REPEATED_6GRAM_RATIO:-0.04}" \
  --max-longest-repeated-ngram "${MAX_LONGEST_REPEATED_NGRAM:-8}" \
  --max-url-hits "${MAX_URL_HITS:-3}" \
  --max-boilerplate-hits "${MAX_BOILERPLATE_HITS:-2}" \
  --output-dir "${OUTPUT_DIR}" \
  --force-exit

singularity run \
  "${SIF}" \
  python -m data_pipeline.summarize_corpus "${OUTPUT_DIR}"/*.jsonl > "${OUTPUT_DIR}/summary.json"
