"""
MedVision AI — Grad-CAM (unified XAI layer)
===========================================

Gradient-weighted Class Activation Mapping for the active MedVision models.

Guarantees
----------
* Runs on the LIVE model object and the LIVE preprocessing transform handed
  over by `medvision_inference.CNNModel`, so the heatmap always belongs to the
  exact prediction the UI displays.
* The model is put in `eval()` mode, only the input tensor requires gradients,
  and the backward pass is scoped with `torch.enable_grad()` (so calling the
  explainer inside `torch.no_grad()` still works).
* Hooks are ALWAYS removed (`remove_hooks()` in a `finally` block, plus a
  context manager and a `__del__` safety net).  Nothing stays attached to the
  shared model.
* No parameter is updated: gradients only weight activations and the graph is
  dropped immediately (`zero_grad(set_to_none=True)`).

Target layer
------------
Resolved per modality through `xai.common.resolve_target_layer`, honouring (in
order) an explicit `target_layer_path`, the `XAI_MODEL_CONFIG` entry of the
modality, then auto-detection.  The active chest / brain / skin / eye
checkpoints are ResNet50 variants, so `layer4[-1]` is used; an EfficientNet
backbone resolves to `backbone.features[-1]` without code changes here.

This module is the project's single Grad-CAM implementation — the legacy
top-level `gradcam.py` is a thin compatibility shim over it.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import (
    HEATMAP_CMAP,
    METHOD_GRADCAM,
    ModelSource,
    XAIResult,
    array_to_pil,
    build_base_result,
    display_size,
    figure_from_panels,
    model_probabilities,
    normalize_map,
    overlay_heatmap,
    prep_xai_image,
    primary_logits,
    resolve_target_index,
    resolve_target_layer,
    resize_map,
    to_rgb,
    unwrap_model_source,
)


class GradCAM:
    """
    Grad-CAM engine bound to one model + one target layer::

        with GradCAM(model, modality="chest") as cam:
            raw_cam, class_idx, probabilities = cam.generate(tensor)

    or explicitly, with `remove_hooks()` in a `finally` block (what
    `explain_gradcam` does).
    """

    def __init__(self, model: torch.nn.Module,
                 target_layer: Optional[torch.nn.Module] = None,
                 target_layer_path: Optional[str] = None,
                 modality: Optional[str] = None):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks: list = []
        self.activation_shape: Optional[Tuple[int, ...]] = None

        if target_layer is None:
            self.target_layer_path, target_layer = resolve_target_layer(
                model, expression=target_layer_path, modality=modality)
        else:
            self.target_layer_path = target_layer_path or type(target_layer).__name__
        self.target_layer = target_layer

        self.hooks.append(target_layer.register_forward_hook(self._save_activation))
        # full backward hook: valid for the sequential ResNet50 blocks used here
        self.hooks.append(target_layer.register_full_backward_hook(self._save_gradient))

    # ── hook callbacks ──────────────────────────────────────────────
    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    # ── core algorithm ──────────────────────────────────────────────
    def generate(self, image_tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        image_tensor: (1, 3, H, W) already preprocessed for this model.

        Returns (raw_cam HxW float32, class_idx, probabilities for all classes).
        The raw cam is NOT min-max normalized: normalization is a display
        decision applied by `explain_gradcam`.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        tensor = image_tensor.detach().clone().to(device)

        with torch.enable_grad():
            tensor.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            output = primary_logits(self.model(tensor))
            probabilities = torch.softmax(output.detach().float(), dim=1)[0].cpu().numpy()

            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())
            class_idx = int(class_idx)

            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1.0
            output.backward(gradient=one_hot)

            gradients = self.gradients
            activations = self.activations
            self.gradients = None
            self.activations = None

        self.model.zero_grad(set_to_none=True)
        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM hooks captured nothing — check the target layer")

        self.activation_shape = tuple(activations.shape)
        cam = self._weighted_cam(gradients, activations)
        return cam, class_idx, probabilities.astype(np.float32)

    @staticmethod
    def _weighted_cam(gradients: torch.Tensor, activations: torch.Tensor) -> np.ndarray:
        """Global-average-pool the gradients -> channel weights -> ReLU(sum w*A)."""
        weights = gradients.mean(dim=(2, 3), keepdim=True)              # (1, C, 1, 1)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        return cam.squeeze().detach().cpu().numpy().astype(np.float32)

    # ── cleanup ─────────────────────────────────────────────────────
    def remove_hooks(self) -> None:
        """Detach every hook this instance registered (idempotent)."""
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove_hooks()

    def __del__(self):                    # last-resort safety net
        try:
            self.remove_hooks()
        except Exception:
            pass


# ─────────────────────────────────────────────
#  HIGH-LEVEL ENTRY POINT
# ─────────────────────────────────────────────
def explain_gradcam(model_or_source: Any,
                    image: Image.Image,
                    target_class: Any = None,
                    target_layer_path: Optional[str] = None,
                    target_layer: Optional[torch.nn.Module] = None,
                    modality: Optional[str] = None,
                    alpha: float = 0.45,
                    cmap: str = HEATMAP_CMAP,
                    make_panel: bool = True) -> XAIResult:
    """
    Produce a standardized Grad-CAM explanation.

    Parameters
    ----------
    model_or_source : medvision_inference.CNNModel | nn.Module | dict
        The live inference object (its `.model` and `.transform` are used).
    image : PIL.Image
        The (untouched) uploaded scan.
    target_class : str | int | None
        Class to explain; None explains the predicted class.
    target_layer_path : str | None
        Explicit module path, e.g. "layer4[-1]" or "backbone.features[-1]".
    make_panel : bool
        Also build the Original | Heatmap | Overlay report figure.
    """
    started = time.perf_counter()
    source = model_or_source if isinstance(model_or_source, ModelSource) \
        else unwrap_model_source(model_or_source)

    size = display_size(source.transform)
    input_array = prep_xai_image(image, size)                 # (H, W, 3) float [0,1]
    tensor = source.transform(array_to_pil(input_array)).unsqueeze(0)
    probabilities = model_probabilities(source.model, source.transform, image, source.device)
    target_index = resolve_target_index(source, target_class, probabilities)

    engine = GradCAM(source.model, target_layer=target_layer,
                     target_layer_path=target_layer_path,
                     modality=modality or source.name)
    try:
        raw_cam, class_idx, cam_probabilities = engine.generate(tensor, class_idx=target_index)
        layer_path = engine.target_layer_path
        activation_shape = engine.activation_shape
    finally:
        engine.remove_hooks()
        hooks_left = len(engine.hooks)

    heatmap = resize_map(normalize_map(raw_cam), size)
    overlay = overlay_heatmap(input_array, heatmap, cmap=cmap, alpha=alpha)
    base_image = array_to_pil(input_array)

    result = build_base_result(
        source=source,
        method=METHOD_GRADCAM,
        probabilities=probabilities,
        original_image=base_image,
        parameters={
            "target_class": target_class,
            "target_layer_path": layer_path,
            "alpha": float(alpha),
            "colormap": cmap,
            "normalization": "min-max",
            "variant": "gradcam",
        },
        target_index=target_index,
        predicted_index=int(np.argmax(probabilities)),
    )

    result.explanation_image = overlay
    result.attribution_map = heatmap
    # Keep both maps on the model-input grid: the raw CAM lives at feature-map
    # resolution (e.g. 7x7 for ResNet50 layer4) and is upsampled here, which is
    # what the UI shows and what later occlusion/fidelity experiments mask.
    result.signed_attribution = resize_map(raw_cam, size)
    result.images = {
        "original": base_image,
        "heatmap": array_to_pil(np.stack([heatmap] * 3, axis=-1)),
        "overlay": overlay,
    }
    result.metadata.update({
        "target_layer": layer_path,
        "activation_shape": list(activation_shape) if activation_shape else None,
        "cam_range": [float(raw_cam.min()), float(raw_cam.max())],
        "class_probabilities_from_cam_pass": {
            name: round(float(p), 6) for name, p in zip(source.class_names, cam_probabilities)},
        "hooks_remaining": int(hooks_left),
        "device": str(source.device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    if make_panel:
        result.images["panel"] = _gradcam_panel(source, result, base_image, heatmap, overlay, cmap)
    return result


def _gradcam_panel(source: ModelSource, result: XAIResult, base_image: Image.Image,
                   heatmap: np.ndarray, overlay: Image.Image, cmap: str) -> Image.Image:
    """Original | Heatmap | Overlay report figure."""
    return figure_from_panels(
        [
            {"title": "Original scan", "image": np.asarray(base_image)},
            {"title": "Grad-CAM heatmap", "image": heatmap, "cmap": cmap,
             "colorbar_label": "attribution"},
            {"title": "Overlay", "image": np.asarray(overlay)},
        ],
        suptitle=(f"{source.name.upper()} | Grad-CAM | predicted {result.predicted_class} "
                  f"({result.confidence * 100:.1f}%) | explained class: {result.target_class}"),
    )
