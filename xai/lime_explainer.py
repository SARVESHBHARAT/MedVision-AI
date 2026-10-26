"""
MedVision AI — LIME for images (unified XAI layer)
==================================================

Image-based LIME (`lime.lime_image.LimeImageExplainer`) on top of the SAME
prediction pipeline the UI uses.

Pipeline parity
---------------
`build_numpy_predict_fn` (see `xai.common`) turns a batch of perturbed RGB
arrays into probabilities by running every sample through the live
`medvision_inference` transform and the live model.  LIME therefore never
bypasses the project preprocessing: what LIME perturbs is the exact input the
prediction was made from.

Reproducibility
---------------
`num_samples`, `batch_size`, `num_features`, the segmentation parameters and
the random seed are all explicit parameters, recorded on the result object and
in the saved JSON record, so the same figure can be regenerated later.

Cost
----
LIME needs `num_samples` model evaluations per explanation.  The default (200)
is deliberately modest for medical images; raise it explicitly if needed.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .common import (
    METHOD_LIME,
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
    to_rgb,
    unwrap_model_source,
)

#: Deliberately modest defaults — medical images make LIME expensive.
DEFAULT_NUM_SAMPLES = 200
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_FEATURES = 8
DEFAULT_RANDOM_SEED = 42
DEFAULT_SEGMENTATION = "quickshift"
#: kernel_size / max_dist / ratio for skimage's quickshift on a 224px scan
DEFAULT_QUICKSHIFT = {"kernel_size": 4, "max_dist": 20, "ratio": 0.2}
DEFAULT_GRID = 8
DEFAULT_SLIC = {"n_segments": 48, "compactness": 10.0, "sigma": 1.0}


def quickshift_segmenter(kernel_size: int = 4, max_dist: int = 20, ratio: float = 0.2):
    """LIME segmentation function using skimage quickshift (canonical LIME)."""
    from skimage.segmentation import quickshift

    def segment(image: np.ndarray) -> np.ndarray:
        return quickshift(np.asarray(image, dtype=np.float64), kernel_size=int(kernel_size),
                          max_dist=int(max_dist), ratio=float(ratio), channel_axis=-1)
    return segment


def slic_segmenter(n_segments: int = 48, compactness: float = 10.0, sigma: float = 1.0):
    """LIME segmentation function using skimage SLIC superpixels."""
    from skimage.segmentation import slic

    def segment(image: np.ndarray) -> np.ndarray:
        return slic(np.asarray(image, dtype=np.float64), n_segments=int(n_segments),
                    compactness=float(compactness), sigma=float(sigma), channel_axis=-1)
    return segment


def grid_segmenter(grid: int = DEFAULT_GRID):
    """
    Fully deterministic grid segmentation (no randomness, no skimage call).

    Useful for stability studies where the segmentation itself must not move
    between runs.
    """
    cells = max(2, int(grid))

    def segment(image: np.ndarray) -> np.ndarray:
        height, width = np.asarray(image).shape[:2]
        rows = np.linspace(0, height, cells + 1).astype(int)
        cols = np.linspace(0, width, cells + 1).astype(int)
        segments = np.zeros((height, width), dtype=np.int32)
        index = 0
        for r in range(cells):
            for c in range(cells):
                segments[rows[r]:rows[r + 1], cols[c]:cols[c + 1]] = index
                index += 1
        return segments
    return segment


def build_segmenter(name: str = DEFAULT_SEGMENTATION,
                    params: Optional[Dict[str, Any]] = None,
                    random_seed: Optional[int] = None):
    """Factory for the supported LIME segmentations: quickshift | slic | grid."""
    params = dict(params or {})
    key = str(name or DEFAULT_SEGMENTATION).lower()
    if key == "quickshift":
        merged = dict(DEFAULT_QUICKSHIFT)
        merged.update(params)
        return quickshift_segmenter(**merged), merged
    if key == "slic":
        merged = dict(DEFAULT_SLIC)
        merged.update(params)
        return slic_segmenter(**merged), merged
    if key == "grid":
        merged = {"grid": int(params.get("grid", DEFAULT_GRID))}
        return grid_segmenter(**merged), merged
    raise ValueError(f"unknown LIME segmentation '{name}' (use quickshift | slic | grid)")


def segments_to_map(segments: np.ndarray, weights: Dict[int, float]) -> np.ndarray:
    """Superpixel ids + per-superpixel weights -> dense signed attribution map."""
    segments = np.asarray(segments)
    lut = np.zeros(int(segments.max()) + 1, dtype=np.float32)
    for index, weight in weights.items():
        index = int(index)
        if 0 <= index < lut.size:
            lut[index] = float(weight)
    return lut[segments].astype(np.float32)


def _boundaries(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Draw superpixel boundary contours with skimage (falls back to raw image)."""
    try:
        from skimage.segmentation import mark_boundaries
        drawn = mark_boundaries(np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0),
                                np.asarray(mask))
        return array_to_pil(np.clip(drawn, 0.0, 1.0))
    except Exception:
        return array_to_pil(image)


# ─────────────────────────────────────────────
#  HIGH-LEVEL ENTRY POINT
# ─────────────────────────────────────────────
def explain_lime(model_or_source: Any,
                 image: Image.Image,
                 target_class: Any = None,
                 num_samples: int = DEFAULT_NUM_SAMPLES,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 num_features: int = DEFAULT_NUM_FEATURES,
                 random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
                 segmentation: str = DEFAULT_SEGMENTATION,
                 segmentation_params: Optional[Dict[str, Any]] = None,
                 positive_only: bool = True,
                 include_negative: bool = True,
                 make_panel: bool = True) -> XAIResult:
    """
    Produce a standardized LIME explanation for a single scan.

    `num_samples` is the real cost knob: the prediction function is called
    `num_samples` times (in `batch_size` chunks).
    """
    from lime.lime_image import LimeImageExplainer

    started = time.perf_counter()
    source = model_or_source if isinstance(model_or_source, ModelSource) \
        else unwrap_model_source(model_or_source)

    size = display_size(source.transform)
    xai_image = prep_xai_image(image, size)                     # (H, W, 3) float [0,1]

    predict_fn = build_numpy_predict_fn(source.model, source.transform, source.device)
    probabilities = predict_fn(xai_image[None, ...])[0]
    target_index = resolve_target_index(source, target_class, probabilities)
    predicted_index = int(np.argmax(probabilities))

    segment_fn, segment_params = build_segmenter(segmentation, segmentation_params)
    set_determinism(random_seed)

    explainer = LimeImageExplainer(random_state=random_seed, verbose=False)
    try:                                     # LIME's ridge surrogate is often
        from sklearn.exceptions import LinAlgWarning   # ill-conditioned on small
        warning_context = warnings.catch_warnings()    # sample counts; the numbers
    except Exception:                                  # themselves are unaffected.
        warning_context = warnings.catch_warnings()
        LinAlgWarning = Warning
    with warning_context:
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        explanation = explainer.explain_instance(
            xai_image,
            predict_fn,
            labels=(target_index,),      # explain exactly the requested class ...
            top_labels=None,             # ... never "the top K labels"
            hide_color=None,
            num_features=int(num_features),
            num_samples=int(num_samples),
            batch_size=int(batch_size),
            segmentation_fn=segment_fn,
            # FIX (degenerate LIME maps): LIME's image kernel is
            # sqrt(exp(-d^2 / 0.25)) and its canonical default metric for
            # images is 'cosine' (distances bounded in [0, 1]).  With
            # 'euclidean' on binary superpixel vectors (40+ dimensions) the
            # distances reach ~6, the proximity kernel underflows (~1e-18),
            # the ridge surrogate collapses to an intercept-only model and
            # every attribution becomes ~0 — the previously observed
            # "constant map" artifact.  'cosine' restores a meaningful
            # neighbourhood weighting; nothing else in the pipeline changes.
            distance_metric="cosine",
            model_regressor=None,
            random_seed=random_seed,
        )

    weights = {int(seg): float(weight)
               for seg, weight in explanation.local_exp.get(target_index, [])}
    signed_map = segments_to_map(explanation.segments, weights)
    positive_map = np.clip(signed_map, 0.0, None)
    heatmap = normalize_map(positive_map)
    if float(np.abs(signed_map).max()) <= 1e-12:
        heatmap = normalize_map(signed_map)

    # ── LIME's own visualizations ────────────────────────────────────
    images: Dict[str, Image.Image] = {}
    selected_positive, mask_positive = explanation.get_image_and_mask(
        target_index, positive_only=bool(positive_only), negative_only=False,
        hide_rest=False, num_features=int(num_features), min_weight=0.0)
    images["lime_positive"] = _boundaries(selected_positive, mask_positive)

    if include_negative:
        selected_signed, mask_signed = explanation.get_image_and_mask(
            target_index, positive_only=False, negative_only=False,
            hide_rest=False, num_features=int(num_features), min_weight=0.0)
        images["lime_positive_negative"] = _boundaries(selected_signed, mask_signed)

    base_image = array_to_pil(xai_image)
    overlay = overlay_heatmap(xai_image, heatmap, alpha=0.45)
    images["original"] = base_image
    images["heatmap"] = array_to_pil(np.stack([heatmap] * 3, axis=-1))
    images["overlay"] = overlay

    result = build_base_result(
        source=source,
        method=METHOD_LIME,
        probabilities=probabilities,
        original_image=base_image,
        parameters={
            "target_class": target_class,
            "num_samples": int(num_samples),
            "batch_size": int(batch_size),
            "num_features": int(num_features),
            "random_seed": random_seed,
            "segmentation": str(segmentation).lower(),
            "segmentation_params": segment_params,
            "positive_only": bool(positive_only),
            "include_negative": bool(include_negative),
            "distance_metric": "cosine",
            "hide_color": None,
        },
        target_index=target_index,
        predicted_index=predicted_index,
    )

    result.explanation_image = overlay
    result.attribution_map = heatmap
    result.signed_attribution = signed_map
    result.images = images

    local_prediction = explanation.local_pred[target_index] if isinstance(
        explanation.local_pred, dict) else explanation.local_pred
    result.metadata.update({
        "num_superpixels": int(np.unique(explanation.segments).size),
        "intercept": float(explanation.intercept.get(target_index, 0.0)),
        "local_ridge_score": float(explanation.score) if explanation.score is not None else None,
        "local_surrogate_prediction": (float(np.asarray(local_prediction).reshape(-1)[0])
                                       if local_prediction is not None else None),
        "model_probability_target_class": float(probabilities[target_index]),
        "top_weighted_superpixels": sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:int(num_features)],
        "device": str(source.device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "lime_version": _lime_version(),
    })

    if make_panel:
        panels = [
            {"title": "Original scan", "image": np.asarray(base_image)},
            {"title": "LIME importance map", "image": heatmap, "cmap": "jet",
             "colorbar_label": "attribution"},
            {"title": "LIME overlay", "image": np.asarray(overlay)},
        ]
        if include_negative and "lime_positive_negative" in images:
            panels.append({"title": "LIME superpixels (+ / -)",
                           "image": np.asarray(images["lime_positive_negative"])})
        result.images["panel"] = figure_from_panels(
            panels,
            suptitle=(f"{source.name.upper()} | LIME ({num_samples} samples) | predicted "
                      f"{result.predicted_class} ({result.confidence * 100:.1f}%) | "
                      f"explained class: {result.target_class}"),
        )
    return result


def _lime_version() -> str:
    try:
        import importlib.metadata as metadata
        return metadata.version("lime")
    except Exception:
        return "unknown"
