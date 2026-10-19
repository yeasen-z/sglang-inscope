#!/usr/bin/env bash
# P0: build the InScope development environment for THIS SGLang checkout.
#
# Why a script: the pinned versions (torch==2.14.1, flashinfer_python[cu13]==0.7.2)
# come from python/pyproject.toml and must not drift per operator. Everything
# machine-specific is an env var with a default.
#
# Notes:
# - No `cargo`/`nvcc` needed: the CUDA pyproject declares no
#   [tool.sglang] rust-extensions, so setup.py never calls cargo metadata.
# - We install torch FIRST and then use --no-build-isolation, otherwise pip's
#   PEP 517 build env downloads a second full copy of torch (build-system
#   requires torch==2.14.1).
#
# Usage:
#   bash inscope/scripts/setup_env.sh            # create env + install
#   REINSTALL=1 bash inscope/scripts/setup_env.sh  # force reinstall sglang

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/opt/miniconda3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/opt/miniconda3/envs/inscope}"
PY_VERSION="${PY_VERSION:-3.12}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_HOST="${PIP_HOST:-pypi.tuna.tsinghua.edu.cn}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "== repo:  $REPO_ROOT"
echo "== env:   $ENV_PREFIX (python $PY_VERSION)"
echo "== index: $PIP_INDEX"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" "python=$PY_VERSION"
else
  echo "== env already exists, reusing"
fi

PY="$ENV_PREFIX/bin/python"
PIP=("$PY" -m pip install --no-cache-dir -i "$PIP_INDEX" --trusted-host "$PIP_HOST")

# 1) torch first, so the editable build can see it without a second download.
echo "== [1/3] torch==2.14.1"
"${PIP[@]}" "torch==2.14.1"

# 2) build backend, so --no-build-isolation has what pyproject requires.
echo "== [2/3] build backend"
"${PIP[@]}" "setuptools>=61.0" "setuptools-rust>=1.11" "setuptools-scm>=8.0" wheel

# 3) sglang itself, editable; remaining deps (incl. flashinfer 0.7.2) resolve here.
echo "== [3/3] pip install -e python"
if [[ "${REINSTALL:-0}" == "1" ]]; then
  "${PIP[@]}" --force-reinstall --no-deps -e "$REPO_ROOT/python"
else
  "${PIP[@]}" --no-build-isolation -e "$REPO_ROOT/python"
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

from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.srt.layers.layer_boundary.residual.batch import final_norm
print("baseline plumbing  OK (aux_hidden_states, final_norm)")
PYEOF

echo "== P0 done"
