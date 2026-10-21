"""Request-scoped hidden-feature capture for SGLang generation."""

from sglang.srt.feature_capture.bridge import final_capture, layer_capture
from sglang.srt.feature_capture.capabilities import (
    FeatureCaptureCapabilities,
    FeatureCaptureCapabilityError,
    is_enabled,
    validate_capabilities,
)
from sglang.srt.feature_capture.spec import (
    SCHEMA_VERSION,
    FeatureCaptureError,
    FeatureCaptureSuffixNotFound,
    FeatureRequestSpec,
    FeatureTarget,
    validate,
)

__all__ = (
    "SCHEMA_VERSION",
    "FeatureCaptureCapabilities",
    "FeatureCaptureCapabilityError",
    "FeatureCaptureError",
    "FeatureCaptureSuffixNotFound",
    "FeatureRequestSpec",
    "FeatureTarget",
    "final_capture",
    "is_enabled",
    "layer_capture",
    "validate",
    "validate_capabilities",
)
