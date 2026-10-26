"""
MedVision AI — SHAP for images (unified XAI layer)
==================================================

Two SHAP routes, both explaining the SAME live prediction pipeline:

1. **Primary — `shap.PartitionExplainer` + `shap.maskers.Image`** (the
   image-compatible method shipped with the installed SHAP 0.52.0).
   The explainer perturbs image regions of the *model-ready* image and calls a
   numpy prediction function that pushes every perturbed sample through the
   live `medvision_inference` transform and the live model, so SHAP never
   bypasses preprocessing.  Values are signed: positive regions pushed the
   explained class up, negative regions pushed it down.

2. **Fallback — `shap.GradientExplainer`** (Expected Gradients on the torch
   model).  Used only when the masker path is unavailable in the environment
   (it needs numba + cv2).  It needs a background reference set; a synthetic
   background derived from the input scan is used by default, and the reason for
   the fallback is recorded in the result metadata.

Both routes honour `max_evals` / `nsamples` so the cost stays controllable —
PartitionExplainer is a perturbation method and its runtime grows with
`max_evals`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .common import (
    DIVERGING_CMAP,
    METHOD_SHAP,
    ModelSource,
    XAIResult,
    array_to_pil,
    build_base_result,
    build_numpy_predict_fn,
    display_size,
    figure_from_panels,
    normalize_map,
    overlay_heatmap,
    prep_xai_image,
    resolve_target_index,
    set_determinism,
    symmetric_scale,
    unwrap_model_source,
)

#: PartitionExplainer perturbation budget (2 baseline evaluations are reserved
#: internally by SHAP, hence the small floor).
DEFAULT_MAX_EVALS = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_MASK_VALUE = "blur(16,16)"      # blur | inpaint_telea | inpaint_ns, or a number
DEFAULT_REDUCE = "mean"
DEFAULT_RANDOM_SEED = 42
#: GradientExplainer fallback settings
DEFAULT_NSAMPLES = 32
DEFAULT_BACKGROUND = "blur"


def shap_version() -> str:
    try:
        import importlib.metadata as metadata
        return metadata.version("shap")
    except Exception:
        return "unknown"


def reduce_channels(array: np.ndarray, mode: str = DEFAULT_REDUCE) -> np.ndarray:
    """Reduce an (H, W, C) signed attribution to (H, W) without losing signs."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    mode = str(mode or DEFAULT_REDUCE).lower()
    if mode == "sum":
        return arr.sum(axis=-1).astype(np.float32)
    if mode in ("max_abs", "maxabs"):
        picked = np.take_along_axis(arr, np.abs(arr).argmax(axis=-1, keepdims=True), axis=-1)
        return picked[..., 0].astype(np.float32)
    return arr.mean(axis=-1).astype(np.float32)


def to_2d_attribution(values: Any, reduce: str = DEFAULT_REDUCE) -> np.ndarray:
    """
    Normalize any SHAP value layout to a signed 2-D map.

    Handles (1, H, W, C, 1), (1, H, W, C), (H, W, C), (1, C, H, W), (C, H, W)
    and (H, W) — PartitionExplainer and GradientExplainer differ here.
    """
    arr = np.asarray(values, dtype=np.float32)
    while arr.ndim > 2:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3:
            break
        else:
            arr = arr.mean(axis=0)

    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] <= 4:                       # channels last (H, W, C)
            return reduce_channels(arr, reduce)
        if arr.shape[0] <= 4:                        # channels first (C, H, W)
            return reduce_channels(np.moveaxis(arr, 0, -1), reduce)
        return arr.mean(axis=-1).astype(np.float32)
    raise ValueError(f"cannot reduce SHAP values of shape {np.asarray(values).shape} to a 2-D map")


def count_model_calls(predict_fn):
    """Wrap a numpy predict function so the real number of model evaluations is known."""
    counter = {"batches": 0, "images": 0}

    def counted(batch):
        array = np.asarray(batch)
        counter["batches"] += 1
        counter["images"] += int(array.shape[0]) if array.ndim > 3 else 1
        return predict_fn(batch)

    return counted, counter


def build_background(xai_image: np.ndarray, mode: str = DEFAULT_BACKGROUND,
                     transform=None, device=None) -> torch.Tensor:
    """Synthetic reference set for the GradientExplainer fallback (no dataset needed)."""
    mode = str(mode or DEFAULT_BACKGROUND).lower()
    frames: List[np.ndarray] = []

    if mode == "blur":
        try:
            import cv2
            blurred = cv2.GaussianBlur((xai_image * 255.0).astype(np.uint8), (31, 31), 0)
            frames.append(blurred.astype(np.float32) / 255.0)
        except Exception:
            frames.append(xai_image.mean(axis=(0, 1), keepdims=True) * np.ones_like(xai_image))
    elif mode == "mean":
        frames.append(np.ones_like(xai_image) * np.asarray([0.485, 0.456, 0.406], dtype=np.float32))
    elif mode == "black":
        frames.append(np.zeros_like(xai_image))

    frames.append(np.zeros_like(xai_image))                       # always include a neutral sample
    frames.append(np.ones_like(xai_image) * xai_image.mean(axis=(0, 1), keepdims=True))

    tensors = [transform(array_to_pil(frame)) for frame in frames]
    return torch.stack(tensors).to(device if device is not None else "cpu")


# ─────────────────────────────────────────────
#  HIGH-LEVEL ENTRY POINT (PartitionExplainer, with automatic fallback)
# ─────────────────────────────────────────────
def explain_shap(model_or_source: Any,
                 image: Image.Image,
                 target_class: Any = None,
                 max_evals: int = DEFAULT_MAX_EVALS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 mask_value: Any = DEFAULT_MASK_VALUE,
                 reduce: str = DEFAULT_REDUCE,
                 random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
                 prefer: str = "partition",
                 nsamples: int = DEFAULT_NSAMPLES,
                 background: str = DEFAULT_BACKGROUND,
                 make_panel: bool = True) -> XAIResult:
    """
    Produce a standardized SHAP explanation.

    `prefer="partition"` (default) uses PartitionExplainer + image masker and
    automatically falls back to GradientExplainer if that path is not usable in
    this environment.  `prefer="gradient"` selects the fallback directly.
    """
    started = time.perf_counter()
    source = model_or_source if isinstance(model_or_source, ModelSource) \
        else unwrap_model_source(model_or_source)

    size = display_size(source.transform)
    xai_image = prep_xai_image(image, size)
    predict_fn = build_numpy_predict_fn(source.model, source.transform, source.device)
    probabilities = predict_fn(xai_image[None, ...])[0]
    target_index = resolve_target_index(source, target_class, probabilities)
    predicted_index = int(np.argmax(probabilities))

    set_determinism(random_seed)
    fallback_reason = ""
    method_used = "gradient" if str(prefer).lower() == "gradient" else "partition"

    values = None
    base_value = None
    gradient_meta: Dict[str, Any] = {}
    calls = {"batches": 0, "images": 0}

    if method_used == "partition":
        try:
            import shap
            counted, calls = count_model_calls(predict_fn)
            masker = shap.maskers.Image(mask_value, xai_image.shape)
            explainer = shap.PartitionExplainer(counted, masker,
                                                output_names=list(source.class_names))
            explanation = explainer(
                xai_image[None, ...],
                max_evals=int(max_evals),
                batch_size=int(batch_size),
                outputs=[target_index],          # explain exactly the requested class
                silent=True,
            )
            values = np.asarray(explanation.values)
            base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
        except Exception as exc:                 # documented limitation -> fallback
            fallback_reason = f"PartitionExplainer unavailable/failed ({type(exc).__name__}: {exc})"
            method_used = "gradient"

    if values is None:
        values, base_value, calls, gradient_meta = _gradient_values(
            source, xai_image, target_index, nsamples=nsamples, background=background)

    signed_map = to_2d_attribution(values, reduce=reduce)
    heatmap = normalize_map(np.abs(signed_map))

    base_image = array_to_pil(xai_image)
    overlay = overlay_heatmap(xai_image, heatmap, alpha=0.45)
    signed_overlay = overlay_heatmap(xai_image, signed_map, cmap=DIVERGING_CMAP,
                                     alpha=0.55, signed=True)

    result = build_base_result(
        source=source,
        method=METHOD_SHAP,
        probabilities=probabilities,
        original_image=base_image,
        parameters={
            "target_class": target_class,
            "shap_explainer": ("PartitionExplainer + maskers.Image" if method_used == "partition"
                               else "GradientExplainer (fallback)"),
            "max_evals": int(max_evals) if method_used == "partition" else None,
            "batch_size": int(batch_size) if method_used == "partition" else None,
            "mask_value": mask_value if method_used == "partition" else None,
            "nsamples": int(nsamples) if method_used == "gradient" else None,
            "background": background if method_used == "gradient" else None,
            "reduce_channels": str(reduce),
            "random_seed": random_seed,
        },
        target_index=target_index,
        predicted_index=predicted_index,
    )

    result.explanation_image = signed_overlay
    result.attribution_map = heatmap
    result.signed_attribution = signed_map
    result.images = {
        "original": base_image,
        "heatmap": array_to_pil(np.stack([heatmap] * 3, axis=-1)),
        "overlay": overlay,
        "overlay_signed": signed_overlay,
    }
    result.metadata.update({
        "shap_version": shap_version(),
        "explainer_used": method_used,
        "fallback_reason": fallback_reason or None,
        "model_evaluations": int(calls["images"]),
        "prediction_calls": int(calls["batches"]),
        "base_value_target_class": base_value,
        "signed_attribution_range": [float(signed_map.min()), float(signed_map.max())],
        "device": str(source.device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    result.metadata.update(gradient_meta)
    if fallback_reason:
        result.warnings.append(fallback_reason)
    if make_panel:
        result.images["panel"] = figure_from_panels(
            [
                {"title": "Original scan", "image": np.asarray(base_image)},
                {"title": "SHAP |attribution|", "image": heatmap, "cmap": "jet",
                 "colorbar_label": "|contribution|"},
                {"title": "SHAP contribution (+ / -)", "image": np.asarray(signed_overlay)},
            ],
            suptitle=(f"{source.name.upper()} | SHAP ({method_used}) | predicted "
                      f"{result.predicted_class} ({result.confidence * 100:.1f}%) | "
                      f"explained class: {result.target_class}"),
        )
    return result


# ─────────────────────────────────────────────
#  GRADIENTEXPLAINER FALLBACK
# ─────────────────────────────────────────────
def explain_shap_gradient(model_or_source: Any,
                          image: Image.Image,
                          target_class: Any = None,
                          nsamples: int = DEFAULT_NSAMPLES,
                          background: str = DEFAULT_BACKGROUND,
                          reduce: str = DEFAULT_REDUCE,
                          random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
                          make_panel: bool = True) -> XAIResult:
    """SHAP via `GradientExplainer` (Expected Gradients) — used explicitly."""
    return explain_shap(model_or_source, image, target_class=target_class,
                        reduce=reduce, random_seed=random_seed, prefer="gradient",
                        nsamples=nsamples, background=background, make_panel=make_panel)


def _gradient_values(source: ModelSource,
                     xai_image: np.ndarray,
                     target_index: int,
                     nsamples: int = DEFAULT_NSAMPLES,
                     background: str = DEFAULT_BACKGROUND,
                     ) -> Tuple[np.ndarray, Optional[float], Dict[str, int], Dict[str, Any]]:
    """
    Expected-Gradients values for one input.

    Returns (values, base_value, call_counter, metadata).  The background is
    synthetic (derived from this single scan) so no dataset/label is involved.
    """
    import shap

    tensor = source.transform(array_to_pil(xai_image)).unsqueeze(0).to(source.device)
    background_tensor = build_background(xai_image, background, source.transform, source.device)

    explainer = shap.GradientExplainer(source.model, background_tensor)
    raw = explainer.shap_values(tensor, nsamples=int(nsamples))

    if isinstance(raw, (list, tuple)):
        selected = raw[target_index] if len(raw) > target_index else raw[0]
    else:
        array = np.asarray(raw)
        if array.ndim > 3 and array.shape[-1] > 1 and array.shape[-1] == len(source.class_names):
            selected = array[..., target_index]
        else:
            selected = array

    metadata = {
        "gradient_nsamples": int(nsamples),
        "gradient_background": str(background),
        "gradient_background_samples": int(background_tensor.shape[0]),
        "gradient_model_evaluations": int(background_tensor.shape[0]) * int(nsamples),
    }
    calls = {"batches": int(background_tensor.shape[0]) * int(nsamples),
             "images": int(background_tensor.shape[0]) * int(nsamples)}
    return np.asarray(selected, dtype=np.float32), None, calls, metadata
