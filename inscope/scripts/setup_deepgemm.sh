#!/usr/bin/env bash
# Install the DeepGEMM-SM110 build that SGLang's FP8 path needs on Jetson Thor.
#
# Why this exists: the SM110 support lives in a DeepGEMM fork, not on PyPI.
# Stock `sgl-deep-gemm==0.2.3` -- what SGLang's dependency resolves to -- has no
# arch_major == 11 routing, so this script exists to get the patched build in.
#
# Two ways to get one, tried in this order:
#
#   1. a prebuilt wheel from the fork's GitHub release (instant, no toolchain
#      needed) -- but _C.so is only valid for the (arch, torch ABI, CUDA major)
#      triple it was compiled against, so the wheel is only accepted after it
#      has been installed and has proven it can dispatch on this machine;
#   2. build from source, which always works and always matches this box.
#
# A wheel that fails the check is discarded and the source build takes over, so
# the fast path is an optimisation and never a new failure mode.
#
# What it installs and why it must be this build:
#   The fork routes arch_major == 11 through the SM100 tcgen05 kernels. Stock
#   upstream DeepGEMM aborts with "Unsupported architecture" on SM110, and the
#   SGLang side detects that and falls back to Triton -- correct, but slower
#   (measured 1.47x on prefill). So installing this is what enables the fast path.
#
# The SGLang side probes the installed build (capability query first, then the
# "+sm110" version marker), so a wrong or half-installed DeepGEMM degrades to
# Triton with a warning instead of failing at request time.
#
# Usage:
#   source /path/to/env/bin/activate
#   bash inscope/scripts/setup_deepgemm.sh
#
#   DEEPGEMM_SRC=/path/to/already/cloned/fork bash inscope/scripts/setup_deepgemm.sh
#   PYTHON=/path/to/env/bin/python bash inscope/scripts/setup_deepgemm.sh
#   FORCE=1 bash inscope/scripts/setup_deepgemm.sh    # rebuild+reinstall even if present
#   FROM_SOURCE=1 bash inscope/scripts/setup_deepgemm.sh   # skip the prebuilt wheel
#   WHEEL_URL=<url> bash inscope/scripts/setup_deepgemm.sh  # use a specific wheel
#   ALLOW_ANY_SM=1 bash inscope/scripts/setup_deepgemm.sh  # cross-build off Thor
#
# Overridable: PYTHON, DEEPGEMM_SRC, DEEPGEMM_REPO, DEEPGEMM_REF, WORK_DIR,
#              CUDA_HOME, FORCE, KEEP_SRC, FROM_SOURCE, WHEEL_URL, ALLOW_ANY_SM.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/yeasen-z/DeepGEMM-thor-sm110.git}"
DEEPGEMM_REF="${DEEPGEMM_REF:-main}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/deepgemm-sm110-build}"
# The published wheel for the tested triple. Override WHEEL_URL to point at a
# different release, or set FROM_SOURCE=1 to ignore it entirely.
DEEPGEMM_WHEEL_URL="${WHEEL_URL:-https://github.com/yeasen-z/DeepGEMM-thor-sm110/releases/download/v0.2.3-sm110/sgl_deep_gemm-0.2.3+sm110-py3-none-linux_aarch64.whl}"
MIN_PY_MAJOR=3
MIN_PY_MINOR=11

# --- Resolve the interpreter (same contract as setup_env.sh) -----------------
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

Activate the environment SGLang is installed into, then re-run:
    source /path/to/env/bin/activate
    bash inscope/scripts/setup_deepgemm.sh

Or name it explicitly:
    PYTHON=/path/to/env/bin/python bash inscope/scripts/setup_deepgemm.sh
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

# The build links _C.so against THIS interpreter's torch, so it must be the
# environment SGLang runs in -- not some other python on PATH.
if ! TORCH_INFO="$("$PY" -c 'import torch,sys; print(torch.__version__, torch.version.cuda or "cpu")' 2>/dev/null)"; then
  cat >&2 <<EOF
error: $PY cannot import torch.

_C.so is linked against a specific torch ABI, so it must be built by the very
interpreter that will import it. Install the environment first:
    bash inscope/scripts/setup_env.sh
EOF
  exit 1
fi
echo "== python: $PY ($PY_VERSION)"
echo "== torch:  $TORCH_INFO"

# Interpreter checks below must not run from inside a DeepGEMM checkout: the
# source tree contains a `deep_gemm/` directory, so `import deep_gemm` there
# picks up the source instead of the installed wheel and dies with
# "ImportError: DeepGEMM extension is missing". Run them from a neutral dir.
NEUTRAL_DIR="$(mktemp -d)"
trap 'rm -rf "$NEUTRAL_DIR"' EXIT

# --- CUDA_HOME: the build needs a self-consistent toolkit --------------------
# Same trap as the SGLang install: on Thor nvcc and the device headers are often
# in different prefixes. The fork's build script checks this too; checking here
# means the user gets the message before a 150 MB clone.
#
# Only the source build needs this. A prebuilt wheel brings its own _C.so and
# still JITs at runtime, so a missing nvcc is a warning there, not a stop.
_need_build=1
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    echo "== CUDA_HOME: $CUDA_HOME (inferred from nvcc on PATH)"
  else
    _need_build=0
  fi
else
  echo "== CUDA_HOME: $CUDA_HOME"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME
  export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
fi

if [[ $_need_build -eq 1 && ! -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  echo "error: no executable nvcc at $CUDA_HOME/bin/nvcc." >&2
  exit 1
fi
if [[ $_need_build -eq 1 && ! -f "$CUDA_HOME/include/curand_kernel.h" ]]; then
  cat >&2 <<EOF
warning: $CUDA_HOME/include/curand_kernel.h is missing.
         The host layer will build, but the runtime JIT needs the CUDA device
         headers and will fail later. Either point CUDA_HOME at a prefix with
         BOTH bin/nvcc and include/, or add the header directory to CPATH.
EOF
fi

# --- Is this machine actually an SM110 box? ---------------------------------
# This build exists for one reason: stock DeepGEMM has no kernel routing for
# arch_major 11. On every other device the ordinary `sgl-deep-gemm==0.2.3` that
# SGLang depends on is the correct package, and installing this fork there would
# either do nothing useful or replace a working build with one compiled for a
# different architecture. So say so and stop rather than spend two minutes and
# 150 MB on it.
_actual_sm="$("$PY" - <<'PYEOF' 2>/dev/null || true
import torch
try:
    if not torch.cuda.is_available():
        print(0)
    else:
        major, minor = torch.cuda.get_device_capability()
        print(major * 10 + minor)
except Exception:
    print(0)
PYEOF
)"
_actual_sm="${_actual_sm:-0}"

if [[ "$_actual_sm" != "110" && "${ALLOW_ANY_SM:-0}" != "1" ]]; then
  cat <<EOF
== this machine reports sm_${_actual_sm}, not sm_110 (Jetson Thor).

The SM110 fork is only needed on Thor. Everywhere else the ordinary build that
SGLang already depends on -- sgl-deep-gemm==0.2.3 from PyPI -- is the right one,
and it is already installed as part of the normal setup. Nothing to do here.

If you are cross-building a wheel for a Thor to install elsewhere, or you know
this is wrong, re-run with:

    ALLOW_ANY_SM=1 bash inscope/scripts/setup_deepgemm.sh
EOF
  exit 0
fi

# --- Already installed? -----------------------------------------------------
if [[ "${FORCE:-0}" != "1" ]]; then
  if (cd "$NEUTRAL_DIR" && "$PY" - <<'PYEOF' >/dev/null 2>&1
import sys
import deep_gemm
version = str(getattr(deep_gemm, "__version__", ""))
query = getattr(deep_gemm, "supported_arch_majors", None)
if callable(query):
    sys.exit(0 if 11 in {int(v) for v in query()} else 1)
sys.exit(0 if "sm110" in version else 1)
PYEOF
  )
  then
    echo "== DeepGEMM with SM110 support is already installed; nothing to do."
    echo "   (re-run with FORCE=1 to rebuild and reinstall anyway)"
    exit 0
  fi
  echo "== no SM110-capable DeepGEMM found; building it"
fi

# --- Try the prebuilt wheel first -------------------------------------------
# _C.so is bound to the (arch, torch ABI, CUDA major) triple it was compiled
# against, and a wheel tag only encodes the CPU arch. So the wheel is treated as
# a candidate, not an answer: it is installed, asked whether it can dispatch on
# this machine, and rolled back if it cannot. Every failure here is a fall
# through to the source build rather than an error.
WHEEL_FILE=""
_install_ok=""

verify_installed() {
  # Prints nothing; exits 0 only if the installed build really dispatches SM110.
  # Runs outside the source tree so a checkout cannot shadow the installed wheel.
  (cd "$NEUTRAL_DIR" && "$PY" - <<'PYEOF' >/dev/null 2>&1
import sys
import deep_gemm
query = getattr(deep_gemm, "supported_arch_majors", None)
if callable(query):
    sys.exit(0 if 11 in {int(v) for v in query()} else 1)
sys.exit(0 if "sm110" in str(getattr(deep_gemm, "__version__", "")) else 1)
PYEOF
  )
}

try_prebuilt_wheel() {
  local url="$1" dir file

  # Only aarch64 builds exist, and only a machine that can run them should try.
  if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "== prebuilt wheel: skipped (published for aarch64, this is $(uname -m))"
    return 1
  fi

  case "$url" in
    *://*) ;;
    *) echo "== prebuilt wheel: $url looks like a local path; using it as-is"
       if [[ ! -f "$url" ]]; then
         echo "== prebuilt wheel: not found; falling back to source build"
         return 1
       fi
       WHEEL_FILE="$url"; return 0 ;;
  esac

  # Download to a scratch dir; never into the repo.
  dir="$(mktemp -d)"
  # pip refuses a wheel whose filename does not encode name/version/python/abi/
  # platform, so keep the URL's basename rather than inventing one.
  file="$dir/${url##*/}"

  echo "== prebuilt wheel: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --connect-timeout 15 -o "$file" "$url" || {
      echo "== prebuilt wheel: download failed; falling back to source build"
      rm -rf "$dir"; return 1; }
  elif command -v wget >/dev/null 2>&1; then
    wget -q --tries=3 --timeout=30 -O "$file" "$url" || {
      echo "== prebuilt wheel: download failed; falling back to source build"
      rm -rf "$dir"; return 1; }
  else
    echo "== prebuilt wheel: neither curl nor wget available; falling back"
    rm -rf "$dir"; return 1
  fi

  # A truncated download is the common failure on a flaky link, and pip's error
  # for it is opaque ("not a valid wheel filename" / "BadZipFile").
  if ! "$PY" - "$file" <<'PYEOF' >/dev/null 2>&1
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).testzip()
PYEOF
  then
    echo "== prebuilt wheel: download is truncated or corrupt; falling back"
    rm -rf "$dir"; return 1
  fi

  WHEEL_FILE="$file"
  return 0
}

if [[ "${FROM_SOURCE:-0}" != "1" && -z "${DEEPGEMM_SRC:-}" ]]; then
  if try_prebuilt_wheel "$DEEPGEMM_WHEEL_URL"; then
    echo "== installing prebuilt wheel"
    if "$PY" -m pip install --no-deps --force-reinstall "$WHEEL_FILE" >/dev/null 2>&1 \
       && verify_installed; then
      _install_ok="wheel"
    else
      echo "== prebuilt wheel: installed but cannot dispatch SM110 on this machine."
      echo "   (its _C.so was built for a different torch ABI or CUDA major)"
      echo "   falling back to a source build, which matches this box by construction."
      # Leave the environment as we found it; the build below reinstalls anyway.
    fi
  fi
fi

# --- Clone + build (only if the prebuilt wheel did not work out) -------------
SRC="${DEEPGEMM_SRC:-}"
if [[ -z "$_install_ok" ]]; then
if [[ -n "$SRC" ]]; then
  if [[ ! -f "$SRC/build_sgl_deep_gemm.sh" ]]; then
    echo "error: DEEPGEMM_SRC=$SRC does not look like the DeepGEMM fork" >&2
    echo "       (no build_sgl_deep_gemm.sh there)." >&2
    exit 1
  fi
  echo "== using existing checkout: $SRC"
else
  SRC="$WORK_DIR/DeepGEMM-thor-sm110"
  if [[ -d "$SRC/.git" ]]; then
    echo "== reusing clone at $SRC"
    git -C "$SRC" fetch --depth 1 origin "$DEEPGEMM_REF"
    git -C "$SRC" checkout -q FETCH_HEAD
  else
    echo "== cloning $DEEPGEMM_REPO ($DEEPGEMM_REF) -> $SRC"
    echo "   ~150 MB of CUTLASS submodule comes with it; this is the slow step."
    rm -rf "$SRC"
    mkdir -p "$(dirname "$SRC")"
    if ! git clone --depth 1 --branch "$DEEPGEMM_REF" "$DEEPGEMM_REPO" "$SRC"; then
      cat >&2 <<EOF

error: could not clone $DEEPGEMM_REPO

Both ways of getting the SM110 build have now failed: the prebuilt wheel was
not usable and the source repository could not be reached.

If this machine can reach the wheel but not git, or you have the release
already, point the script at a local copy:

    WHEEL_URL=/path/to/sgl_deep_gemm-0.2.3+sm110-py3-none-linux_aarch64.whl \
      bash inscope/scripts/setup_deepgemm.sh

    DEEPGEMM_SRC=/path/to/already/cloned/DeepGEMM-thor-sm110 \
      bash inscope/scripts/setup_deepgemm.sh

Retrying usually works too -- this clone is the largest download in the setup.
EOF
      exit 1
    fi
  fi
fi

# --- Build ------------------------------------------------------------------
# PYTHON_EXE decides the torch ABI of _C.so, so pass it explicitly rather than
# letting the fork's script fall back to `which python3`.
echo "== building (PYTHON_EXE=$PY)"
PYTHON_EXE="$PY" bash "$SRC/build_sgl_deep_gemm.sh"

WHEEL="$(ls -t "$SRC"/dist/sgl_deep_gemm-*.whl "$SRC"/dist/sgl-deep-gemm-*.whl 2>/dev/null | head -n1 || true)"
if [[ -z "$WHEEL" ]]; then
  echo "error: the build produced no wheel under $SRC/dist." >&2
  exit 1
fi
echo "== wheel: $WHEEL"

# --- Install ----------------------------------------------------------------
# --no-deps: the wheel declares no runtime dependencies (SGLang pins the
# distribution separately), and --force-reinstall is required when the same
# version is already present, or pip reports "already satisfied" and silently
# keeps the old binary.
echo "== installing"
"$PY" -m pip install --no-deps --force-reinstall "$WHEEL"
_install_ok="source"
fi   # end of the source-build branch

# --- Verify -----------------------------------------------------------------
echo "== verify"
# `set -e` is suspended for this one command: a failing verification is an
# expected outcome we want to explain, not a reason to abort silently.
set +e
(cd "$NEUTRAL_DIR" && "$PY" - <<'PYEOF'
import sys

problems = []

import deep_gemm
version = str(getattr(deep_gemm, "__version__", "<unknown>"))
print(f"deep_gemm version    : {version}")

# Preferred signal: the compiled binary reports which arch majors it accepts.
query = getattr(deep_gemm, "supported_arch_majors", None)
if callable(query):
    try:
        majors = tuple(query())
        print(f"supported arch majors: {majors}")
        if 11 in majors:
            print("SM110 dispatch       : [ok] reported by _C.so")
        else:
            problems.append(f"_C.so reports arch majors {majors}, which excludes 11 (SM110)")
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        problems.append(f"supported_arch_majors() failed: {type(exc).__name__}: {exc}")
else:
    # Older wheel: fall back to the version marker SGLang also understands.
    print("supported arch majors: <not exposed by this build>")
    if "sm110" in version:
        print("SM110 dispatch       : [ok] via version marker (older build)")
    else:
        problems.append('no capability query and version lacks "sm110"')

try:
    import torch
    if not torch.cuda.is_available():
        problems.append("torch.cuda is not available, cannot exercise a real GEMM")
    else:
        major, minor = torch.cuda.get_device_capability()
        print(f"device capability    : sm_{major}{minor}")
        if (major, minor) != (11, 0):
            print("                      (not SM110; the dispatch check below is informational)")
        # A tiny real GEMM: proves the kernels dispatch instead of trusting metadata.
        a = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        d = torch.empty(64, 128, device="cuda", dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(a, b, d)
        torch.cuda.synchronize()
        print("dispatch smoke test  : [ok] bf16_gemm_nt ran")
except Exception as exc:  # noqa: BLE001 - diagnostic path
    problems.append(f"dispatch smoke test failed: {type(exc).__name__}: {exc}")

if problems:
    print()
    for p in problems:
        print(f"[!!] {p}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
  cat >&2 <<'EOF'

DeepGEMM installed, but the verification above reported problems.
SGLang will detect this and fall back to the Triton FP8 backend (correct, but
slower). Useful next steps:

    python -m deep_gemm.doctor            # fuller health report, if the build ships it
    export CUDA_HOME=<dir with bin/nvcc AND include/curand_kernel.h>

See inscope/suggestion_and_require.md for the requirements this satisfies.
EOF
  exit $STATUS
fi

# --- Report where things landed ---------------------------------------------
if [[ "$_install_ok" == "source" && "${KEEP_SRC:-1}" == "1" ]]; then
  echo "== source kept at: ${SRC:-<none>}"
  echo "   (next run reuses it; set KEEP_SRC=0 to leave only the clone cache)"
fi

if [[ "$_install_ok" == "wheel" ]]; then
  _origin="prebuilt wheel ($DEEPGEMM_WHEEL_URL)"
else
  _origin="source build (${WHEEL:-?})"
fi

cat <<EOF

== DeepGEMM-SM110 ready

   from     $_origin
   python   $PY

SGLang will now take the DeepGEMM FP8 path on SM110. Make sure it can use it:
   export SGLANG_ENABLE_JIT_DEEPGEMM=1

Note that kernel compilation still happens at runtime, and it still needs a CUDA
toolkit with device headers -- a prebuilt wheel only saves the host-layer build:

   export CUDA_HOME=<dir with bin/nvcc AND include/curand_kernel.h>

First startup compiles kernels for the M shapes actually used (several minutes,
cached afterwards). SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 cuts that by ~8x.
EOF