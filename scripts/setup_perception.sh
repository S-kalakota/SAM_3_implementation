#!/usr/bin/env bash
# Prepare the Python/SAM source environment without downloading gated weights.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SAM_SOURCE="${PROJECT_ROOT}/third_party/sam3"
SAM_REVISION="5dd401d1c5c1d5c3eedff06d41b77af824517619"

die() {
    echo "error: $*" >&2
    exit 1
}

command -v "${PYTHON}" >/dev/null 2>&1 \
    || die "Python interpreter not found: ${PYTHON}"
command -v git >/dev/null 2>&1 || die "git is required"

if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    "${PYTHON}" -m venv --system-site-packages "${PROJECT_ROOT}/.venv"
fi
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if ! "${VENV_PYTHON}" -c 'import torch' >/dev/null 2>&1; then
    die "PyTorch is unavailable. Install the CUDA/Jetson build for this host, then rerun."
fi

if [[ ! -d "${SAM_SOURCE}/.git" ]]; then
    [[ ! -e "${SAM_SOURCE}" ]] \
        || die "${SAM_SOURCE} exists but is not a Git checkout"
    git clone https://github.com/facebookresearch/sam3.git "${SAM_SOURCE}"
fi

git -C "${SAM_SOURCE}" fetch origin "${SAM_REVISION}"
git -C "${SAM_SOURCE}" checkout --detach "${SAM_REVISION}"

"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install \
    -r "${PROJECT_ROOT}/requirements.perception.txt" \
    -e "${SAM_SOURCE}" \
    -e "${PROJECT_ROOT}/VLA_project"

"${VENV_PYTHON}" - <<'PY'
import torch
print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
PY

cat <<'EOF'

Python and SAM source setup is complete. Model weights are intentionally not
downloaded by this script. Follow README.md's "Download model assets" section,
then run ./sam3-dino start.
EOF
