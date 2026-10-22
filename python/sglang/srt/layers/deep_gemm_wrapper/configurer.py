import logging
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _sm120_deep_gemm_apis_available() -> bool:
    try:
        import deep_gemm
    except (ImportError, OSError, RuntimeError):
        return False
    return all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "fp8_einsum",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "transform_sf_into_required_layout",
        )
    )


def _deep_gemm_declares_sm(deep_gemm, sm_version: int) -> Optional[bool]:
    """Ask the installed DeepGEMM build whether it supports `sm_version`.

    Returns None when the build exposes no capability query at all, so the
    caller can fall back to the version string. The query shapes are the ones
    requested in inscope/suggestion_and_require.md R4; builds are accepted to
    report the SM version (110, matching get_device_sm) or the arch major (11,
    which is what the SM110 kernel routing switches on).
    """
    arch_major = sm_version // 10

    # supported_arch_majors() reads the answer out of the compiled _C.so, which
    # is the most trustworthy source: a fresh Python side over a stale _C.so
    # would otherwise claim support it does not have. Such a mismatch surfaces
    # as AttributeError here, so that is caught alongside the value errors.
    query = getattr(deep_gemm, "supported_arch_majors", None)
    if callable(query):
        try:
            return arch_major in {int(v) for v in query()}
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning(
                "deep_gemm.supported_arch_majors() failed (%s); ignoring it.",
                exc,
            )

    declared = getattr(deep_gemm, "SUPPORTED_SM_VERSIONS", None)
    if declared is not None:
        try:
            return sm_version in {int(v) for v in declared}
        except (TypeError, ValueError):
            logger.warning(
                "deep_gemm.SUPPORTED_SM_VERSIONS is not a set of integers (%r); "
                "ignoring it.",
                declared,
            )

    declared = getattr(deep_gemm, "SUPPORTED_ARCH_MAJORS", None)
    if declared is not None:
        try:
            return arch_major in {int(v) for v in declared}
        except (TypeError, ValueError):
            logger.warning(
                "deep_gemm.SUPPORTED_ARCH_MAJORS is not a set of integers (%r); "
                "ignoring it.",
                declared,
            )

    supports_arch = getattr(deep_gemm, "supports_arch", None)
    if callable(supports_arch):
        # A bare False cannot be told apart from "this argument type was
        # rejected", so accept support if *any* documented form answers True;
        # if a form was accepted but answered False for all of them, that is a
        # real negative.
        answered = False
        for arg in (sm_version, arch_major, f"sm{sm_version}"):
            try:
                if supports_arch(arg):
                    return True
                answered = True
            except (TypeError, ValueError, AttributeError):
                continue
        if answered:
            return False
        logger.warning(
            "deep_gemm.supports_arch() rejected every argument form "
            "(sm version, arch major, 'smNNN'); ignoring it."
        )

    return None


def _deep_gemm_supports_sm(sm_version: int) -> bool:
    """Whether the *installed* DeepGEMM build can dispatch kernels for this SM.

    Preferred signal is the capability query (suggestion_and_require.md R4);
    builds predating it are recognised by the `+sm110` version local segment
    (R5). Stock upstream DeepGEMM exposes neither and has no SM110 kernel
    routing, so it reports unsupported. Must not create a CUDA context.
    """
    try:
        import deep_gemm
    except (ImportError, OSError, RuntimeError):
        return False

    declared = _deep_gemm_declares_sm(deep_gemm, sm_version)
    if declared is not None:
        return declared

    version = str(getattr(deep_gemm, "__version__", ""))
    return f"sm{sm_version}" in version


def _compute_enable_deep_gemm():
    if not (_is_cuda or _is_musa):
        return False
    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():
        return False

    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    # SM120/SM121 support (mma.sync block-scale, no TMEM) landed in DeepGEMM#324;
    # probe every API used by the SM120 DSV4 paths since installed builds may
    # expose fp8_einsum but still predate the SM120 kernels.
    if sm_version in (120, 121) and not _sm120_deep_gemm_apis_available():
        return False
    # SM110 (Jetson Thor) runs the SM100 tcgen05 kernels only on a build that
    # routes arch_major == 11 through them. Stock upstream DeepGEMM does not,
    # and would abort on the first GEMM with "Unsupported architecture"; probe
    # first so an unsupported build falls back to the Triton FP8 backend
    # instead of failing at request time.
    if sm_version == 110 and not _deep_gemm_supports_sm(110):
        logger.warning(
            "Detected SM110 (Jetson Thor) but the installed DeepGEMM build does "
            "not support it; falling back to the Triton FP8 backend. Build "
            "DeepGEMM from the SM110 fork to enable it (see "
            "inscope/suggestion_and_require.md)."
        )
        return False

    try:
        import deep_gemm  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False

    return True


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and get_platform().is_sm100
# SM110 (Jetson Thor) shares the SM100 tcgen05 kernel family, and the SM100
# block-FP8 kernel requires UE8M0 (power-of-two) scales: its SF packing kernel
# traps on any FP32 scale whose sign/mantissa bits are non-zero
# (deep_gemm/impls/smxx_layout.cuh). So SM110 must request UE8M0 scales and the
# weight requant, exactly like SM100.
DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() in (110, 120, 121)
)
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)


def _supports_paged_sparse_mqa_logits() -> bool:
    if not DEEPGEMM_BLACKWELL:
        return False
    import deep_gemm

    return all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
        )
    )


DEEPGEMM_PAGED_SPARSE_MQA_LOGITS = _supports_paged_sparse_mqa_logits()
