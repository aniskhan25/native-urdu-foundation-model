#!/bin/bash
#SBATCH --job-name=urdu-generate
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=project_462000131

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
CONFIG="${CONFIG:-configs/urdu_700m_expanded_v1.yaml}"
CHECKPOINT="${CHECKPOINT:-latest}"
PROMPTS="${PROMPTS:-eval/prompts_urdu.txt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-0}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-}"
OUTPUT="${OUTPUT:-}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

cd "${REPO_DIR}"

EXTRA_ARGS=()
if [ -n "${OUTPUT}" ]; then
  EXTRA_ARGS=(--output "${OUTPUT}")
fi
if [ -n "${PROMPT_TEMPLATE}" ]; then
  EXTRA_ARGS+=(--prompt-template "${PROMPT_TEMPLATE}")
fi

singularity run \
  "${SIF}" \
  python -m eval.generate_samples \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompts "${PROMPTS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --repetition-penalty "${REPETITION_PENALTY}" \
  --no-repeat-ngram-size "${NO_REPEAT_NGRAM_SIZE}" \
  "${EXTRA_ARGS[@]}"
