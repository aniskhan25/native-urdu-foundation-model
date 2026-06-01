#!/bin/bash
#SBATCH --job-name=urdu-manifest
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=project_462000131

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${DATA_ROOT}/tokenized}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

mkdir -p "${MANIFEST_ROOT}"
cd "${REPO_DIR}"

singularity run \
  "${SIF}" \
  python -m data_pipeline.build_training_manifest \
  --output "${MANIFEST_ROOT}/dress_rehearsal_manifest.json" \
  "${TOKENIZED_ROOT}/fineweb2_urd_arab.json" \
  "${TOKENIZED_ROOT}/makhzan_urdu.json"
