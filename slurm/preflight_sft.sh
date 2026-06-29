#!/bin/bash
#SBATCH --job-name=urdu-sft-preflight
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:15:00
#SBATCH --account=project_462000131

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
CONFIG="${CONFIG:-configs/urdu_700m_sft_v1.yaml}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

if [ ! -f "${SIF}" ]; then
  echo "Missing container image: ${SIF}" >&2
  exit 1
fi

MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_DIR}/cache"
export MIOPEN_USER_DB="${MIOPEN_DIR}/config"
export TORCH_HOME="${TORCH_HOME:-/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home}"
mkdir -p "${TORCH_HOME}"

cd "${REPO_DIR}"

singularity run \
  "${SIF}" \
  python -m training.train_sft --config "${CONFIG}" --preflight
