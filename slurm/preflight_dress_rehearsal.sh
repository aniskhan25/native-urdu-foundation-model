#!/bin/bash
#SBATCH --job-name=urdu-preflight
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --account=project_462000131

set -euo pipefail

module use /appl/local/csc/modulefiles/

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
CONFIG="${CONFIG:-configs/urdu_dress_rehearsal.yaml}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
CONTAINER_BINDS="${CONTAINER_BINDS:-/pfs,/users,/projappl,/scratch,/project,/flash}"

cd "${REPO_DIR}"

singularity exec \
  --bind="${CONTAINER_BINDS}" \
  "${CONTAINER_IMAGE}" \
  python -m training.train --config "${CONFIG}" --preflight
