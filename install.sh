#!/usr/bin/env bash
#
# Create the conda environment for VAIC.
#
#   ./install.sh                 # create env (default name: VAIC)
#   ./install.sh --name myenv    # use a different env name
#
# Env overrides: ENV_NAME, PYTHON_VERSION, TORCH_VERSION, TORCH_INDEX_URL,
#                RECREATE=1 (remove existing env first).
#
# NOTE: the scripts in data_preprocessing/ default to VAIC_ENV=VAIC. If you create the
# env under another name, run them with VAIC_ENV=<name>.
#
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-VAIC}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8.10}"
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

usage() { sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)      ENV_NAME="$2"; shift 2 ;;
        --name=*)    ENV_NAME="${1#*=}"; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found on PATH." >&2; exit 1; }
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "${RECREATE:-0}" = 1 ]; then
        echo ">>> Removing existing env '$ENV_NAME' (RECREATE=1)"
        conda env remove -y -n "$ENV_NAME"
    else
        echo "ERROR: conda env '$ENV_NAME' already exists. Use RECREATE=1 to replace it," >&2
        echo "       or pass --name <other>." >&2
        exit 1
    fi
fi

echo ">>> Creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip wheel setuptools

echo ">>> Installing PyTorch $TORCH_VERSION ($TORCH_INDEX_URL)"
pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

echo ">>> Installing numpy + cython first (chumpy builds against them)"
pip install numpy==1.23.1 cython==0.29.30

echo ">>> Installing chumpy (needs numpy present; SMPL .pkl loading depends on it)"
pip install --no-build-isolation chumpy==0.70

echo ">>> Installing the rest of requirements.txt (includes the eth-siplab/smplx fork)"
pip install -r requirements.txt

echo ">>> Import sanity check"
python - <<'PY'
import torch, numpy, scipy, h5py, cv2, joblib, tqdm, chumpy, smplx, gdown
import inspect
assert "compute_rotations" in inspect.signature(smplx.SMPLX.forward).parameters, \
    "smplx is not the eth-siplab fork (no compute_rotations)"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
print("core imports OK")
PY

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
