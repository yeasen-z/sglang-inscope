#!/usr/bin/env bash
# P0: build the InScope development environment on an NVIDIA Jetson (aarch64) box.
#
# Same job as inscope/scripts/setup_env.sh, adapted for a Thor-class device where
# the x86 assumptions do not hold:
#
# - No sudo and no /opt prefix: the conda env lives under the user's own conda.
# - pip stalls on this network, so `uv` does the resolving/downloading.
# - CUDA_HOME must be a *self-consistent* toolchain directory. On Thor the
#   pieces are commonly split: nvcc/ptxas come from one place and the device
#   headers (curand_kernel.h, ...) from another (e.g. a pip `nvidia-*` package).
#   CUDA_HOME must resolve BOTH <CUDA_HOME>/bin/nvcc and
#   <CUDA_HOME>/include/curand_kernel.h, or DeepGEMM's build-time and the
#   runtime JIT will fail at different points. If your headers live elsewhere,
#   either assemble a directory that has both or add the header dir to CPATH.
#
# Every machine-specific value is an env var with a discoverable default; nothing
# here is hard-coded to one user or one checkout.
#
# Usage:
#   bash inscope/scripts/setup_env_thor.sh
#   REINSTALL=1 bash inscope/scripts/setup_env_thor.sh
#   CUDA_HOME=/path/to/cuda bash inscope/scripts/setup_env_thor.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY_VERSION="${PY_VERSION:-3.12}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

# Locate conda without assuming an install prefix: honour CONDA_BIN / CONDA_EXE,
# else look at the usual per-user and system locations.
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-}}"
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  for cand in \
    "$(command -v conda 2>/dev/null || true)" \
    "${HOME:-}/miniforge3/bin/conda" \
    "${HOME:-}/mambaforge/bin/conda" \
    "${HOME:-}/miniconda3/bin/conda" \
    "${HOME:-}/anaconda3/bin/conda" \
    /opt/miniconda3/bin/conda \
    /opt/conda/bin/conda; do
    if [[ -n "$cand" && -x "$cand" ]]; then CONDA_BIN="$cand"; break; fi
  done
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "error: cannot find 'conda'. Set CONDA_BIN=/path/to/conda." >&2
  exit 1
fi
# envs live beside the conda install, so derive the prefix from CONDA_BIN.
CONDA_ROOT="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$CONDA_ROOT/envs/inscope}"

echo "== repo:  $REPO_ROOT"
echo "== conda: $CONDA_BIN"
echo "== env:   $ENV_PREFIX (python $PY_VERSION)"
echo "== index: $PIP_INDEX"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" "python=$PY_VERSION"
else
  echo "== env already exists, reusing"
fi
PY="$ENV_PREFIX/bin/python"

# uv: Rust-based resolver/downloader; pip stalls on this network.
if [[ ! -x "$ENV_PREFIX/bin/uv" ]]; then
  "$PY" -m pip install --no-cache-dir uv
fi
UV="$ENV_PREFIX/bin/uv"

# 1) torch first, so the editable build can see it without a second download.
echo "== [1/3] torch==2.14.1"
"$UV" pip install --python "$PY" -i "$PIP_INDEX" "torch==2.14.1"

# 2) build backend, so --no-build-isolation has what pyproject requires.
echo "== [2/3] build backend"
"$UV" pip install --python "$PY" -i "$PIP_INDEX" \
  "setuptools>=61.0" "setuptools-rust>=1.11" "setuptools-scm>=8.0" wheel

# 3) sglang itself, editable; remaining deps resolve here.
export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
echo "== [3/3] pip install -e python (rust extensions: $SGLANG_BUILD_RUST_EXTS)"
if [[ "${REINSTALL:-0}" == "1" ]]; then
  "$UV" pip install --python "$PY" -i "$PIP_INDEX" \
    --force-reinstall --no-deps -e "$REPO_ROOT/python"
else
  "$UV" pip install --python "$PY" -i "$PIP_INDEX" \
    --no-build-isolation -e "$REPO_ROOT/python"
fi

echo "== verify"
"$PY" - <<'PYEOF'
import importlib.metadata as md
import torch

print("torch              ", torch.__version__, "cuda", torch.version.cuda)
print("cuda devices       ", torch.cuda.device_count())
for pkg in ("sglang", "flashinfer-python", "sgl-kernel", "transformers", "triton"):
    try:
        print(f"{pkg:19s}", md.version(pkg))
    except md.PackageNotFoundError:
        print(f"{pkg:19s}", "MISSING")

import sglang
print("sglang.__file__    ", sglang.__file__)
PYEOF

cat <<EOF

== environment ready

   python    $PY

Before running anything, export:
   CUDA_HOME=<dir with bin/nvcc AND include/curand_kernel.h>
   CUDACXX=\$CUDA_HOME/bin/nvcc

Feature capture (this fork's feature):
   export SGLANG_ENABLE_FEATURE_CAPTURE=1

DeepGEMM FP8 GEMM on SM110 needs the DeepGEMM-SM110 build (patched _C.so that
routes arch_major == 11 through the SM100 kernels) plus the two SGLang-side
edits carried on this branch. With those in place:
   export SGLANG_ENABLE_JIT_DEEPGEMM=1
Leave it at 0 to run the FP8 path on the Triton backend instead.
EOF