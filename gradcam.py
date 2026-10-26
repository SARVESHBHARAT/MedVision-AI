"""
MedVision AI — GRAD-CAM (compatibility shim)
============================================

The Grad-CAM implementation moved into the unified XAI layer:

    xai/gradcam.py      -> class GradCAM, explain_gradcam(...)

This module is kept so that existing callers keep working unchanged
(`main.py`-style imports, older notebooks, external scripts).  It contains no
second implementation — everything delegates to `xai.gradcam`, so there is only
one Grad-CAM code path in the project.

Legacy API (unchanged):
    overlay, heatmap, predicted_class = generate_gradcam_image(model, image)
    figure = create_gradcam_figure(original, overlay, heatmap, prediction,
                                   confidence, scan_type)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from xai.common import (
    HEATMAP_CMAP,
    array_to_pil,
    display_size,
    figure_from_panels,
    normalize_map,
    overlay_heatmap,
    prep_xai_image,
    unwrap_model_source,
)
from xai.gradcam import GradCAM, explain_gradcam

#: Kept for backwards compatibility (identical to medvision_inference values).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

__all__ = ["GradCAM", "MEAN", "STD", "create_gradcam_figure",
           "explain_gradcam", "generate_gradcam_image"]


def generate_gradcam_image(model, image: Image.Image, class_idx: Optional[int] = None
                           ) -> Tuple[Image.Image, np.ndarray, int]:
    """
    Legacy helper: PIL image -> (overlay, heatmap, predicted_class).

    `model` may be a `medvision_inference.CNNModel` (preferred: its live
    transform is used) or a bare `torch.nn.Module` (ResNet50 @224px ImageNet
    preprocessing is assumed, exactly like the original implementation).
    """
    result = explain_gradcam(model, image, target_class=class_idx, make_panel=False)
    overlay = result.explanation_image or array_to_pil(prep_xai_image(
        image, display_size(unwrap_model_source(model).transform)))
    heatmap = result.attribution_map
    return overlay, heatmap, int(result.predicted_class_index)


def create_gradcam_figure(original: Image.Image,
                          overlay: Image.Image,
                          heatmap: np.ndarray,
                          prediction: str,
                          confidence: float,
                          scan_type: str) -> Image.Image:
    """
    Side-by-side report figure: Original | GRAD-CAM Overlay | Attention Heatmap.

    Implemented on the shared panel renderer, which uses matplotlib's
    `buffer_rgba` API (the old `canvas.tostring_rgb()` call was removed in
    matplotlib >= 3.9 and silently failed).
    """
    heatmap = normalize_map(np.asarray(heatmap, dtype=np.float32))
    return figure_from_panels(
        [
            {"title": "Original Scan", "image": np.asarray(original.convert("RGB"))},
            {"title": "GRAD-CAM Overlay", "image": np.asarray(overlay.convert("RGB"))},
            {"title": "Attention Heatmap", "image": heatmap, "cmap": HEATMAP_CMAP,
             "colorbar_label": "Attention"},
        ],
        suptitle=f"{scan_type.upper()} | Prediction: {prediction} ({confidence * 100:.1f}%)",
    )


if __name__ == "__main__":
    print("[GRAD-CAM] Compatibility shim ready — implementation: xai/gradcam.py")
    print("Usage: from gradcam import generate_gradcam_image, create_gradcam_figure")
    print("New code: from xai import explain_image -> explain_image(cnn_model, image)")
