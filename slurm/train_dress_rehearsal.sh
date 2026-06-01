#!/bin/bash
#SBATCH --job-name=urdu-dress
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --time=06:00:00
#SBATCH --account=project_462000131

set -euo pipefail

module use /appl/local/csc/modulefiles/

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
CONFIG="${CONFIG:-configs/urdu_dress_rehearsal.yaml}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
CONTAINER_BINDS="${CONTAINER_BINDS:-/pfs,/users,/projappl,/scratch,/project,/flash}"
EXTRA_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then
  EXTRA_ARGS=(--max-steps "${MAX_STEPS}")
fi

if [ ! -f "${CONTAINER_IMAGE}" ]; then
  echo "Missing container image: ${CONTAINER_IMAGE}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/${USER}-miopen-cache
mkdir -p "${MIOPEN_USER_DB_PATH}"

cd "${REPO_DIR}"

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

srun singularity exec \
  --rocm \
  --bind="${CONTAINER_BINDS}" \
  "${CONTAINER_IMAGE}" \
  python -m torch.distributed.run \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node=8 \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  -m training.train \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"
