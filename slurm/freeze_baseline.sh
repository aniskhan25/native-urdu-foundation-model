#!/bin/bash
#SBATCH --job-name=urdu-freeze-base
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=project_462000131

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
SPEC="${SPEC:-configs/baseline_v1.yaml}"
LOCK="${LOCK:-${DATA_ROOT}/manifests/baseline_v1.lock.json}"
MODE="${MODE:-freeze}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

cd "${REPO_DIR}"

case "${MODE}" in
  freeze)
    ARGS=(--output "${LOCK}")
    ;;
  verify)
    ARGS=(--verify "${LOCK}")
    ;;
  *)
    echo "MODE must be freeze or verify" >&2
    exit 2
    ;;
esac

singularity run \
  "${SIF}" \
  python scripts/freeze_baseline.py \
  --spec "${SPEC}" \
  --repo-root "${REPO_DIR}" \
  "${ARGS[@]}"
