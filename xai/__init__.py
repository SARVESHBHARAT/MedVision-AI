"""
MedVision AI — Unified Explainable AI layer
===========================================

One entry point for the three XAI methods used in the project::

    from xai import explain_image

    result = explain_image(cnn_model, image, method="gradcam")   # or "lime" / "shap"

`cnn_model` is the live `medvision_inference.CNNModel`, so the explanation is
always produced with the same model, device and preprocessing that produced the
prediction.  Every method returns the same `XAIResult` container (visuals +
machine-readable attribution + the settings that were used).

Modules
-------
common          : shared helpers, modality/model configuration, `XAIResult`
gradcam         : Grad-CAM (`GradCAM`, `explain_gradcam`)
lime_explainer  : image LIME (`explain_lime`)
shap_explainer  : image SHAP (`explain_shap`, `explain_shap_gradient`)

Explanations describe model behaviour, not clinical causality — see
`XAI_DISCLAIMER`.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

from .common import (
    DIVERGING_CMAP,
    HEATMAP_CMAP,
    METHOD_DESCRIPTIONS,
    METHOD_GRADCAM,
    METHOD_LABELS,
    METHOD_LIME,
    METHOD_SHAP,
    METHODS,
    XAI_DISCLAIMER,
    XAI_MODEL_CONFIG,
    ModelSource,
    XAIResult,
    describe_model,
    display_size,
    get_profile,
    image_fingerprint,
    prep_xai_image,
    resolve_module,
    resolve_target_layer,
    settings_fingerprint,
    unwrap_model_source,
)
from .gradcam import GradCAM, explain_gradcam
from .lime_explainer import explain_lime
from .shap_explainer import explain_shap, explain_shap_gradient, shap_version

__version__ = "1.0.0"

EXPLAINERS = {
    METHOD_GRADCAM: explain_gradcam,
    METHOD_LIME: explain_lime,
    METHOD_SHAP: explain_shap,
}

#: Defaults used by the API/UI.  LIME/SHAP defaults stay deliberately modest.
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    METHOD_GRADCAM: {
        "make_panel": True,
    },
    METHOD_LIME: {
        "num_samples": 200,
        "batch_size": 16,
        "num_features": 8,
        "random_seed": 42,
        "segmentation": "quickshift",
        "include_negative": True,
        "make_panel": True,
    },
    METHOD_SHAP: {
        "max_evals": 128,
        "batch_size": 16,
        "mask_value": "blur(16,16)",
        "reduce": "mean",
        "random_seed": 42,
        "prefer": "partition",
        "make_panel": True,
    },
}

#: Which parameters may be tuned, with safety bounds for the UI.
TUNABLE_PARAMS: Dict[str, Dict[str, Any]] = {
    METHOD_GRADCAM: {
        "target_layer_path": {"type": "str", "default": None},
        "alpha": {"type": "float", "min": 0.1, "max": 0.9, "default": 0.45},
    },
    METHOD_LIME: {
        "num_samples": {"type": "int", "min": 50, "max": 1000, "step": 50, "default": 200},
        "num_features": {"type": "int", "min": 3, "max": 30, "step": 1, "default": 8},
        "batch_size": {"type": "int", "min": 4, "max": 64, "step": 4, "default": 16},
        "segmentation": {"type": "choice",
                         "options": ["quickshift", "slic", "grid"], "default": "quickshift"},
        "random_seed": {"type": "int", "min": 0, "max": 100000, "default": 42},
    },
    METHOD_SHAP: {
        "max_evals": {"type": "int", "min": 32, "max": 512, "step": 32, "default": 128},
        "batch_size": {"type": "int", "min": 4, "max": 64, "step": 4, "default": 16},
        "mask_value": {"type": "choice",
                       "options": ["blur(16,16)", "blur(8,8)", "inpaint_telea", "inpaint_ns"],
                       "default": "blur(16,16)"},
        "reduce": {"type": "choice", "options": ["mean", "sum", "max_abs"], "default": "mean"},
        "prefer": {"type": "choice", "options": ["partition", "gradient"], "default": "partition"},
        "random_seed": {"type": "int", "min": 0, "max": 100000, "default": 42},
    },
}


def available_methods(probe: bool = True) -> Dict[str, Any]:
    """Availability + wording of every method in this environment."""
    import importlib.util

    info: Dict[str, Any] = {}
    for method in METHODS:
        entry: Dict[str, Any] = {
            "method": method,
            "label": METHOD_LABELS[method],
            "description": METHOD_DESCRIPTIONS[method],
            "available": True,
            "reason": None,
        }
        if probe and method == METHOD_LIME and importlib.util.find_spec("lime") is None:
            entry.update(available=False, reason="the 'lime' package is not installed")
        if probe and method == METHOD_SHAP and importlib.util.find_spec("shap") is None:
            entry.update(available=False, reason="the 'shap' package is not installed")
        info[method] = entry
    return info


def explain_image(model_or_source: Any,
                  image: Any,
                  method: str = METHOD_GRADCAM,
                  target_class: Any = None,
                  **params: Any) -> XAIResult:
    """
    Run one explanation method and return an `XAIResult`.

    `params` are filtered against the target explainer's signature, so the API
    may forward a full settings payload without breaking other methods.
    """
    key = str(method or METHOD_GRADCAM).strip().lower()
    if key not in EXPLAINERS:
        raise ValueError(f"unknown XAI method '{method}' (use {sorted(EXPLAINERS)})")

    explainer = EXPLAINERS[key]
    settings = dict(DEFAULT_PARAMS.get(key, {}))
    if params:
        allowed = set(inspect.signature(explainer).parameters)
        settings.update({k: v for k, v in params.items() if k in allowed and v is not None})
    return explainer(model_or_source, image, target_class=target_class, **settings)


__all__ = [
    "DEFAULT_PARAMS",
    "DIVERGING_CMAP",
    "EXPLAINERS",
    "GradCAM",
    "HEATMAP_CMAP",
    "METHOD_DESCRIPTIONS",
    "METHOD_LABELS",
    "METHODS",
    "METHOD_GRADCAM",
    "METHOD_LIME",
    "METHOD_SHAP",
    "ModelSource",
    "TUNABLE_PARAMS",
    "XAI_DISCLAIMER",
    "XAI_MODEL_CONFIG",
    "XAIResult",
    "__version__",
    "available_methods",
    "describe_model",
    "display_size",
    "explain_gradcam",
    "explain_image",
    "explain_lime",
    "explain_shap",
    "explain_shap_gradient",
    "get_profile",
    "image_fingerprint",
    "prep_xai_image",
    "resolve_module",
    "resolve_target_layer",
    "settings_fingerprint",
    "shap_version",
    "unwrap_model_source",
]
