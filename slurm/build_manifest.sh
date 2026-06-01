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

module use /appl/local/csc/modulefiles/
module load cray-python
module load pytorch

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
VENV_DIR="${VENV_DIR:-${DATA_ROOT}/venv}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${DATA_ROOT}/tokenized}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/install_venv.sh before submitting jobs." >&2
  exit 1
fi

mkdir -p "${MANIFEST_ROOT}"
cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

python -m data_pipeline.build_training_manifest \
  --output "${MANIFEST_ROOT}/dress_rehearsal_manifest.json" \
  "${TOKENIZED_ROOT}/fineweb2_urd_arab.json" \
  "${TOKENIZED_ROOT}/makhzan_urdu.json"

