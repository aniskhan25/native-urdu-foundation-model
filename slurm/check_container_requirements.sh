#!/bin/bash
#SBATCH --job-name=urdu-container-check
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --account=project_462000131

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
CONTAINER_BINDS="${CONTAINER_BINDS:-/pfs,/users,/projappl,/scratch,/project,/flash}"

if [ ! -f "${CONTAINER_IMAGE}" ]; then
  echo "Missing container image: ${CONTAINER_IMAGE}" >&2
  exit 1
fi

cd "${REPO_DIR}"

singularity exec \
  --bind="${CONTAINER_BINDS}" \
  "${CONTAINER_IMAGE}" \
  python scripts/check_container_requirements.py
