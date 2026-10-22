#!/usr/bin/env bash
# P0: install THIS SGLang checkout (the InScope fork) into an existing Python
# environment, in editable mode.
#
# One entry point for every platform. It does NOT create environments and does
# NOT look for conda: it installs into the interpreter you have already
# activated, which is the same contract upstream SGLang documents ("use Python
# 3.11+ in an activated Python environment"). Create the environment however you
# like -- uv, venv, conda -- then activate it and run this.
#
# What the script actually buys you over `pip install -e python`:
#
#   1. torch is installed FIRST and the editable build runs with
#      --no-build-isolation. python/pyproject.toml lists torch==2.14.1 in
#      build-system.requires, so an isolated build would download a second full
#      copy of torch (hundreds of MB, and slow on aarch64).
#   2. SGLANG_BUILD_RUST_EXTS=none, because python/setup.py builds every Rust
#      extension by default ("unset or 'all' builds everything") and that needs
#      cargo, which is not part of a stock install.
#   3. A post-install self-check that the fork's plumbing is importable, so a
#      half-installed tree is caught here instead of at first request.
#
# On aarch64 (Jetson / Thor) it additionally checks CUDA_HOME, where the pieces
# are commonly split: nvcc from one place and the device headers
# (curand_kernel.h, ...) from another (e.g. a pip `nvidia-*` package).
# CUDA_HOME must resolve BOTH <CUDA_HOME>/bin/nvcc and
# <CUDA_HOME>/include/curand_kernel.h, or DeepGEMM's build-time and its runtime
# JIT fail at different points.
#
# Usage:
#   source /path/to/env/bin/activate
#   bash inscope/scripts/setup_env.sh
#
#   PYTHON=/path/to/env/bin/python bash inscope/scripts/setup_env.sh
#   REINSTALL=1 bash inscope/scripts/setup_env.sh   # force reinstall sglang
#
# Overridable: PYTHON, UV, PIP, PIP_INDEX, REPO_ROOT, REINSTALL,
#              SGLANG_BUILD_RUST_EXTS, CUDA_HOME.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PIP_INDEX="${PIP_INDEX:-}"        # empty => whatever index the installer uses
MIN_PY_MAJOR=3
MIN_PY_MINOR=11

# --- Resolve the interpreter -------------------------------------------------
# Mirrors how an environment is normally activated; an explicit PYTHON wins.
# $VIRTUAL_ENV is set by venv/uv, $CONDA_PREFIX by `conda activate`, so neither
# branch needs a conda binary. Nothing here guesses install prefixes.
_resolve_python() {
  local c
  for c in "${PYTHON:-}" \
           "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}" \
           "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}" \
           "$(command -v python3 2>/dev/null || true)" \
           "$(command -v python 2>/dev/null || true)"; do
    if [[ -n "$c" && -x "$c" ]]; then printf '%s\n' "$c"; return 0; fi
  done
  return 1
}

PY="$(_resolve_python || true)"
if [[ -z "$PY" ]]; then
  cat >&2 <<'MSG'
error: no Python interpreter found.

Activate the environment SGLang should be installed into, then re-run:
    source /path/to/env/bin/activate
    bash inscope/scripts/setup_env.sh

Or name the interpreter explicitly:
    PYTHON=/path/to/env/bin/python bash inscope/scripts/setup_env.sh

No environment yet? Create one the way upstream does:
    uv venv --python 3.12 && source .venv/bin/activate
MSG
  exit 1
fi

if ! PY_VERSION="$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"; then
  echo "error: $PY does not run. Point PYTHON at a working interpreter." >&2
  exit 1
fi
if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (${MIN_PY_MAJOR}, ${MIN_PY_MINOR}) else 1)"; then
  echo "error: Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ required, $PY is ${PY_VERSION}." >&2
  exit 1
fi

# The interpreter *is* the environment; ask it rather than deriving the path
# from an installer layout.
ENV_PREFIX="$("$PY" -c 'import sys; print(sys.prefix)')"
if [[ ! -w "$ENV_PREFIX" ]]; then
  echo "error: $ENV_PREFIX is not writable; cannot install into it." >&2
  exit 1
fi

MACHINE="$(uname -m)"
echo "== repo:   $REPO_ROOT"
echo "== python: $PY ($PY_VERSION)"
echo "== env:    $ENV_PREFIX"
echo "== arch:   $MACHINE"
echo "== index:  ${PIP_INDEX:-<default>}"

# --- aarch64 / Jetson: CUDA_HOME must be self-consistent ----------------------
if [[ "$MACHINE" == "aarch64" || "$MACHINE" == "arm64" ]]; then
  if [[ -z "${CUDA_HOME:-}" ]]; then
    echo "warning: CUDA_HOME is not set. On Jetson the toolkit is rarely at" >&2
    echo "         /usr/local/cuda, so a later build or the DeepGEMM JIT may fail." >&2
  else
    if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
      echo "warning: $CUDA_HOME/bin/nvcc is missing; CUDA_HOME must contain bin/nvcc." >&2
    fi
    if [[ ! -f "$CUDA_HOME/include/curand_kernel.h" ]]; then
      cat >&2 <<EOF
warning: $CUDA_HOME/include/curand_kernel.h is missing.
         On Thor, nvcc and the device headers often live in different places
         (one CUDA install + one pip nvidia-* package). Either assemble a
         CUDA_HOME that has both, or add the header directory to CPATH.
EOF
    fi
  fi
fi

# --- Resolve the installer ---------------------------------------------------
# uv is preferred when present (much faster, and it does not stall on slow
# networks); pip otherwise. Same code path on every architecture.
IDX_ARGS=()
if [[ -n "$PIP_INDEX" ]]; then IDX_ARGS=(-i "$PIP_INDEX"); fi

UV="${UV:-$(command -v uv 2>/dev/null || true)}"
if [[ -z "$UV" && -x "$ENV_PREFIX/bin/uv" ]]; then
  UV="$ENV_PREFIX/bin/uv"
fi

if [[ -n "$UV" && -x "$UV" ]]; then
  echo "== installer: uv ($UV)"
  _install() { "$UV" pip install --python "$PY" ${IDX_ARGS[@]+"${IDX_ARGS[@]}"} "$@"; }
else
  PIP="${PIP:-$PY -m pip}"
  echo "== installer: pip ($PIP)"
  _install() { $PIP install ${IDX_ARGS[@]+"${IDX_ARGS[@]}"} "$@"; }
fi

# 1) torch first, so the editable build can see it without a second download.
echo "== [1/3] torch==2.14.1"
_install "torch==2.14.1"

# 2) build backend, so --no-build-isolation has what pyproject requires.
echo "== [2/3] build backend"
_install "setuptools>=61.0" "setuptools-rust>=1.11" "setuptools-scm>=8.0" wheel

# 3) sglang itself, editable; remaining deps (incl. flashinfer) resolve here.
export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
echo "== [3/3] install -e python (rust extensions: $SGLANG_BUILD_RUST_EXTS)"
if [[ "${REINSTALL:-0}" == "1" ]]; then
  _install --force-reinstall --no-deps -e "$REPO_ROOT/python"
else
  _install --no-build-isolation -e "$REPO_ROOT/python"
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

cat <<EOF

== P0 done

Next, if you use this fork's feature:
   export SGLANG_ENABLE_FEATURE_CAPTURE=1

DeepGEMM FP8 GEMM on SM110 additionally needs the DeepGEMM-SM110 build (a
patched _C.so routing arch_major == 11 through the SM100 kernels) plus the two
SGLang-side edits on this branch. With those in place:
   export SGLANG_ENABLE_JIT_DEEPGEMM=1
Leave it at 0 to run the FP8 path on the Triton backend instead.
EOF
