#!/bin/bash
#SBATCH --job-name=urdu-llm
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=24:00:00
#SBATCH --account=project_462000131

set -euo pipefail

module use /appl/local/csc/modulefiles/
module load pytorch

PROJECT_ID="project_462000131"
REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/${USER}-miopen-cache
mkdir -p "${MIOPEN_USER_DB_PATH}"
cd "${REPO_DIR}"

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

srun python -m torch.distributed.run \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  training/train.py \
  --config configs/urdu_1_3b.yaml
