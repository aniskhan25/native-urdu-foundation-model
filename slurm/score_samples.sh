#!/bin/bash
#SBATCH --job-name=urdu-score-samples
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --account=project_462000131

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
RUN_DIR="${RUN_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data/runs/700m_expanded_v1}"
INPUT="${INPUT:-${SAMPLES:-${RUN_DIR}/samples.jsonl}}"
OUTPUT="${OUTPUT:-}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

cd "${REPO_DIR}"

EXTRA_ARGS=()
if [ -n "${OUTPUT}" ]; then
  EXTRA_ARGS+=(--scores-output "${OUTPUT}")
fi
if [ -n "${SUMMARY_OUTPUT}" ]; then
  EXTRA_ARGS+=(--summary-output "${SUMMARY_OUTPUT}")
fi

singularity run \
  "${SIF}" \
  python -m eval.score_samples \
  --input "${INPUT}" \
  "${EXTRA_ARGS[@]}"
