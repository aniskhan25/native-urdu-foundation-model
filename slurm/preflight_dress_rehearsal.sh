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
module load cray-python
module load pytorch

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
VENV_DIR="${VENV_DIR:-${DATA_ROOT}/venv}"
CONFIG="${CONFIG:-configs/urdu_dress_rehearsal.yaml}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/install_venv.sh before submitting jobs." >&2
  exit 1
fi

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

python training/train.py --config "${CONFIG}" --preflight

