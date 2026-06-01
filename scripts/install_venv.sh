#!/bin/bash

set -euo pipefail

module use /appl/local/csc/modulefiles/
module load cray-python
module load pytorch

REPO_DIR="${REPO_DIR:-/scratch/project_462000131/anisrahm/native-urdu-foundation-model}"
DATA_ROOT="${DATA_ROOT:-/scratch/project_462000131/anisrahm/native-urdu-foundation-data}"
VENV_DIR="${VENV_DIR:-${DATA_ROOT}/venv}"

mkdir -p "$(dirname "${VENV_DIR}")"
cd "${REPO_DIR}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import datasets
import sentencepiece
import tqdm
import yaml
print("venv ready")
PY
