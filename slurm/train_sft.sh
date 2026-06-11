#!/bin/bash
#SBATCH --job-name=urdu-sft
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH --account=project_462000131

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
CONFIG="${CONFIG:-configs/urdu_700m_sft_v1.yaml}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
export CONFIG
EXTRA_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then
  EXTRA_ARGS=(--max-steps "${MAX_STEPS}")
fi
if [ -n "${RESUME:-}" ]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_DIR}/cache"
export MIOPEN_USER_DB="${MIOPEN_DIR}/config"
export TORCH_HOME="${TORCH_HOME:-/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home}"
mkdir -p "${TORCH_HOME}"

cd "${REPO_DIR}"

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT="${MASTER_PORT:-1${SLURM_JOB_ID:0-4}}"
export MASTER_ADDR MASTER_PORT

srun singularity run \
  "${SIF}" \
  bash -c 'python -m torch.distributed.run \
  --numa-binding=exclusive \
  --nnodes="${SLURM_JOB_NUM_NODES}" \
  --nproc_per_node=8 \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  -m training.train_sft \
  --config "${CONFIG}" \
  "$@"' bash "${EXTRA_ARGS[@]}"
