#!/usr/bin/env bash
# Run in a Python 3.10 environment with CUDA 12.4 and a C++ build toolchain.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
export MAX_JOBS=${MAX_JOBS:-4}

"${PYTHON}" -m pip install 'pip>=25.1' setuptools wheel packaging ninja
"${PYTHON}" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"${PYTHON}" -m pip install -r "${ROOT}/requirements.txt"
"${PYTHON}" -m pip install --no-build-isolation 'transformer-engine[pytorch]==2.2.0'
"${PYTHON}" -m pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' \
  'https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post2/flashinfer_python-0.2.1.post2+cu124torch2.6-cp38-abi3-linux_x86_64.whl'

APEX_BUILD_DIR=$(mktemp -d)
trap 'rm -rf -- "${APEX_BUILD_DIR}"' EXIT
git clone --branch 25.04 --depth 1 https://github.com/NVIDIA/apex.git "${APEX_BUILD_DIR}/apex"
"${PYTHON}" -m pip install --no-build-isolation --no-deps \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel ${MAX_JOBS}" \
  "${APEX_BUILD_DIR}/apex"
