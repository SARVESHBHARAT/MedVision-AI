"""
MedVision AI — Shared Explainable-AI utilities
==============================================

Small, dependency-light helpers shared by the unified Grad-CAM / LIME / SHAP
layer (`xai/`).

Design rules (see XAI.md):
  * The XAI layer never loads, trains, converts or rewrites model weights.
  * Every explanation is produced with the SAME model object, SAME device and
    SAME preprocessing transform that produced the original prediction
    (both are taken from the live `medvision_inference.CNNModel`).
  * Every method returns the same machine-readable result object
    (`XAIResult`) so the API / Streamlit UI / research scripts can consume
    Grad-CAM, LIME and SHAP in exactly the same way.

Author: MedVision AI
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
METHOD_GRADCAM = "gradcam"
METHOD_LIME = "lime"
METHOD_SHAP = "shap"
METHODS: Tuple[str, ...] = (METHOD_GRADCAM, METHOD_LIME, METHOD_SHAP)

#: ImageNet normalization — identical to medvision_inference.img_transform
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEFAULT_INPUT_SIZE = (224, 224)      # fallback only; the live transform wins
HEATMAP_CMAP = "jet"                 # matches the original gradcam.py look
DIVERGING_CMAP = "coolwarm"          # signed (+ / -) attributions

METHOD_DESCRIPTIONS: Dict[str, str] = {
    METHOD_GRADCAM: (
        "Shows image regions that contributed strongly to the model's "
        "prediction (gradient-weighted class activation map)."
    ),
    METHOD_LIME: (
        "Shows local image regions that influenced this particular prediction "
        "(local surrogate model fitted around this image)."
    ),
    METHOD_SHAP: (
        "Shows estimated feature/region contributions toward or away from the "
        "prediction (Shapley-value attribution of the model's output)."
    ),
}

METHOD_LABELS: Dict[str, str] = {
    METHOD_GRADCAM: "Grad-CAM",
    METHOD_LIME: "LIME",
    METHOD_SHAP: "SHAP",
}

XAI_DISCLAIMER = (
    "Explanations describe the behaviour of the model on this input. They are "
    "attribution maps, not evidence of clinical causality, and must never be "
    "read as the confirmed location of a disease. All findings require "
    "confirmation by a licensed clinician."
)

# ─────────────────────────────────────────────
#  IMAGE HELPERS
# ─────────────────────────────────────────────
def to_rgb(image: Image.Image) -> Image.Image:
    """Return the image as RGB without modifying the original object."""
    return image.convert("RGB") if image.mode != "RGB" else image


def display_size(transform: Callable[[Image.Image], torch.Tensor],
                 default: Tuple[int, int] = DEFAULT_INPUT_SIZE) -> Tuple[int, int]:
    """
    Infer the (height, width) the model actually receives by running the LIVE
    preprocessing transform on a probe image.  Whatever the transform does to a
    real scan it will also do to the probe, so the explanation always works at
    the same resolution the prediction was made at.
    """
    try:
        probe = Image.new("RGB", (16, 16), (0, 0, 0))
        tensor = transform(probe)
        return (int(tensor.shape[-2]), int(tensor.shape[-1]))
    except Exception:
        return default


def prep_xai_image(image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    """
    PIL image -> float32 RGB array in [0, 1] at `size`.

    This is the *input space* seen by LIME/SHAP: the same resize the model's
    transform performs, before tensor conversion and ImageNet normalization.
    LIME/SHAP perturbations therefore travel through the real preprocessing
    pipeline (see `build_numpy_predict_fn`).

    Note: this is NOT the same as the raw uploaded file — if a modality ever
    needs an extra geometric preprocessing step, it must be applied here too so
    that explanations keep matching predictions.
    """
    height, width = int(size[0]), int(size[1])
    rgb = to_rgb(image)
    if rgb.size != (width, height):
        rgb = rgb.resize((width, height), Image.BILINEAR)
    return np.asarray(rgb).astype(np.float32) / 255.0


def array_to_pil(array: np.ndarray) -> Image.Image:
    """float [0,1] or uint8 array (H, W, 3) -> PIL RGB image."""
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")


def normalize_map(array: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] (all-zero / constant maps stay zero)."""
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def symmetric_scale(array: np.ndarray) -> np.ndarray:
    """Scale signed values to [-1, 1] by the largest absolute value."""
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.abs(arr).max())
    if peak <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / peak).astype(np.float32)


def resize_map(array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a 2-D attribution map to (height, width)."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    height, width = int(size[0]), int(size[1])
    if arr.shape == (height, width):
        return arr
    try:
        import cv2
        return cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception:  # cv2 is present in this project; PIL keeps us safe anyway
        pil = Image.fromarray(arr, mode="F").resize((width, height), Image.BILINEAR)
        return np.asarray(pil, dtype=np.float32)


def _colormap(name: str):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.cm as cm
    try:
        return matplotlib.colormaps[name]          # matplotlib >= 3.6
    except Exception:
        return getattr(cm, name)


def colorize(attribution: np.ndarray, cmap: str = HEATMAP_CMAP,
             signed: bool = False) -> np.ndarray:
    """
    Attribution map (H, W) -> RGB float array in [0, 1].

    `signed=True` maps [-1, 1] through a diverging colormap so positive and
    negative contributions stay visually distinguishable.
    """
    arr = np.asarray(attribution, dtype=np.float32)
    cfun = _colormap(cmap)
    if signed:
        scaled = (np.clip(symmetric_scale(arr), -1.0, 1.0) + 1.0) / 2.0
    else:
        scaled = np.clip(normalize_map(arr), 0.0, 1.0)
    return cfun(scaled)[..., :3].astype(np.float32)


def overlay_heatmap(base_rgb: np.ndarray, attribution: np.ndarray,
                    cmap: str = HEATMAP_CMAP, alpha: float = 0.45,
                    signed: bool = False) -> Image.Image:
    """
    Blend a colour-mapped attribution over the unmodified base image.

    The base image is never modified in place — the overlay is a new RGB array.
    """
    base = np.asarray(base_rgb, dtype=np.float32)
    if base.dtype == np.uint8 or base.max() > 1.0:
        base = base[..., :3] / 255.0
    base = np.clip(base[..., :3], 0.0, 1.0)

    attr = np.asarray(attribution, dtype=np.float32)
    if attr.ndim == 3:
        attr = attr.mean(axis=-1)
    if attr.shape != base.shape[:2]:
        attr = resize_map(attr, base.shape[:2])

    colored = colorize(attr, cmap=cmap, signed=signed)
    if signed:
        # Keep the scan fully readable where the attribution is ~0; the colour
        # itself carries the sign of the contribution.
        weight = np.abs(symmetric_scale(attr))[..., None] * float(alpha)
    else:
        weight = float(alpha)

    blended = np.clip(base * (1.0 - weight) + colored * weight, 0.0, 1.0)
    return array_to_pil(blended)


def encode_png_base64(image: Image.Image, data_url: bool = True) -> str:
    """PIL image -> base64 PNG string (optionally as a `data:` URL)."""
    buffer = io.BytesIO()
    to_rgb(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}" if data_url else encoded


def decode_png_base64(payload: str) -> Image.Image:
    """base64 PNG string (with or without a `data:` prefix) -> PIL image."""
    if isinstance(payload, str) and payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def image_fingerprint(image: Image.Image) -> str:
    """Stable short hash of an image's pixels (cache / settings keys)."""
    arr = np.asarray(to_rgb(image).resize((64, 64)))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def settings_fingerprint(settings: Dict[str, Any]) -> str:
    """Stable short hash of an explanation configuration."""
    blob = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def set_determinism(seed: Optional[int]) -> None:
    """Best-effort determinism for the perturbation-based explainers."""
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ─────────────────────────────────────────────
#  MODEL PLUMBING (no weights are ever modified)
# ─────────────────────────────────────────────
def primary_logits(output: Any) -> torch.Tensor:
    """
    Safely extract the primary class logits from a model output.

    Handles plain tensors, `(logits, aux)` tuples/lists and dict-style outputs
    (the Eye V5 model exposes an auxiliary ordinal head; the active ResNet50
    models return a plain tensor, but the guard keeps the XAI layer robust).
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "out", "output", "main"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("model output dict contains no tensor")
    if isinstance(output, (tuple, list)) and output:
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"unsupported model output type: {type(output)!r}")


def build_numpy_predict_fn(model: torch.nn.Module,
                           transform: Callable[[Image.Image], torch.Tensor],
                           device: Optional[torch.device] = None,
                           ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build the `classifier_fn` contract used by LIME and SHAP.

    input : numpy array (N, H, W, 3), float in [0, 1] (uint8 also accepted)
    output: numpy array (N, num_classes) of softmax probabilities

    Every perturbed image is pushed through EXACTLY the preprocessing transform
    the live prediction path uses, then through the live model in eval mode —
    the perturbation explainers therefore explain the real prediction pipeline
    instead of a re-implementation of it.
    """
    device = device or next(model.parameters()).device

    @torch.no_grad()
    def predict(images: np.ndarray) -> np.ndarray:
        array = np.asarray(images)
        if array.ndim == 3:
            array = array[None, ...]
        if array.dtype != np.uint8:
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        tensors = [transform(Image.fromarray(frame[..., :3].astype(np.uint8), mode="RGB"))
                   for frame in array]
        batch = torch.stack(tensors).to(device)

        was_training = model.training
        model.eval()
        try:
            logits = primary_logits(model(batch)).float()
        finally:
            if was_training:           # never leave the live model in another mode
                model.train()

        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

    return predict


def model_probabilities(model: torch.nn.Module,
                        transform: Callable[[Image.Image], torch.Tensor],
                        image: Image.Image,
                        device: Optional[torch.device] = None) -> np.ndarray:
    """Class probabilities for a single PIL image, via the live pipeline."""
    return build_numpy_predict_fn(model, transform, device=device)(
        prep_xai_image(image, display_size(transform))[None, ...])[0]


def class_probability_dict(class_names: Sequence[str], probabilities: np.ndarray) -> Dict[str, float]:
    """{class_name: rounded probability} for every class."""
    return {str(name): round(float(p), 6) for name, p in zip(class_names, probabilities)}


def resolve_target_class(class_names: Sequence[str], target_class: Any) -> Optional[int]:
    """Accept a class index, a class name or None (None = predicted class)."""
    if target_class is None:
        return None
    if isinstance(target_class, (int, np.integer)) and not isinstance(target_class, bool):
        index = int(target_class)
        if index < 0 or index >= len(class_names):
            raise ValueError(f"target class index {index} out of range 0..{len(class_names) - 1}")
        return index
    text = str(target_class).strip()
    if text.lstrip("+-").isdigit():
        return resolve_target_class(class_names, int(text))
    lookup = {str(name).strip().lower(): i for i, name in enumerate(class_names)}
    if text.lower() not in lookup:
        raise ValueError(f"unknown target class '{target_class}'. Known: {list(class_names)}")
    return lookup[text.lower()]


# ─────────────────────────────────────────────
#  GRAD-CAM TARGET-LAYER RESOLUTION
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class ModelXAIProfile:
    """Per-modality XAI defaults (target layer, notes) for the active models."""
    modality: str
    architecture: str = "auto"
    target_layer: Optional[str] = None      # None -> auto-detect from the module tree
    notes: str = ""


#: The active MedVision checkpoints are all ResNet50 variants (chest / brain /
#: skin @224px, eye / Eye V3 @384px — the Eye task stays five-class).  The target
#: layer is expressed as a module path so a different architecture can be
#: described here without touching the explainer code (`backbone.features[-1]`
#: for EfficientNet, `features[-1]` for torchvision EfficientNet, ...).
XAI_MODEL_CONFIG: Dict[str, ModelXAIProfile] = {
    "chest": ModelXAIProfile(
        modality="chest", architecture="resnet50", target_layer="layer4[-1]",
        notes="Chest X-ray ResNet50 @224px, ImageNet normalization."),
    "brain": ModelXAIProfile(
        modality="brain", architecture="resnet50", target_layer="layer4[-1]",
        notes="Brain MRI ResNet50 V2 @224px, ImageNet normalization."),
    "skin": ModelXAIProfile(
        modality="skin", architecture="resnet50", target_layer="layer4[-1]",
        notes="Skin ISIC ResNet50 @224px, ImageNet normalization."),
    "eye": ModelXAIProfile(
        modality="eye", architecture="resnet50", target_layer="layer4[-1]",
        notes="Eye DR five-class ResNet50 V3 @384px fundus cache, ImageNet "
              "normalization. Uses medvision_inference.MODALITY_TRANSFORMS['eye']. "
              "If the active Eye checkpoint is ever swapped (e.g. an EfficientNet "
              "variant), update medvision_inference.MODEL_PATHS / "
              "MODALITY_TRANSFORMS and this profile together."),
}


def get_profile(modality: Optional[str]) -> Optional[ModelXAIProfile]:
    """Profile for a modality key ('chest' | 'brain' | 'skin' | 'eye')."""
    return XAI_MODEL_CONFIG.get(str(modality).lower()) if modality else None


def resolve_module(model: torch.nn.Module, expression: str) -> torch.nn.Module:
    """
    Resolve a dotted module path with optional integer indexing, e.g.
    `layer4[-1]`, `backbone.features[-1]`, `features.7`, `[-1]`.
    """
    current: Any = model
    for token in str(expression).split("."):
        token = token.strip()
        if not token:
            continue
        match = re.match(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)?(?:\[(?P<idx>-?\d+)\])?$", token)
        if not match or (match.group("name") is None and match.group("idx") is None):
            raise ValueError(f"invalid module path segment '{token}' in '{expression}'")
        if match.group("name"):
            current = getattr(current, match.group("name"))
        if match.group("idx") is not None:
            current = current[int(match.group("idx"))]
    if not isinstance(current, torch.nn.Module):
        raise TypeError(f"'{expression}' resolved to {type(current)!r}, not an nn.Module")
    return current


def auto_target_layer(model: torch.nn.Module) -> str:
    """
    Detect a sensible Grad-CAM target layer for an unknown architecture:
    ResNet `layer4`, then common `features[-1]` stacks, then the last Conv2d.
    """
    for candidate in ("layer4[-1]", "backbone.features[-1]", "features[-1]",
                      "encoder.features[-1]", "stages[-1]"):
        try:
            resolve_module(model, candidate)
            return candidate
        except Exception:
            continue

    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
    if last_name is None:
        raise ValueError("could not auto-detect a Grad-CAM target layer for this model")
    return last_name


def resolve_target_layer(model: torch.nn.Module,
                         expression: Optional[str] = None,
                         modality: Optional[str] = None,
                         ) -> Tuple[str, torch.nn.Module]:
    """
    Pick the Grad-CAM target layer.

    Priority: explicit `expression` argument > per-modality XAI_MODEL_CONFIG
    entry > automatic detection.  Returns (path_string, module).
    """
    candidates: List[str] = []
    if expression:
        candidates.append(expression)
    profile = get_profile(modality)
    if profile and profile.target_layer:
        candidates.append(profile.target_layer)

    for candidate in candidates:
        try:
            module = resolve_module(model, candidate)
        except Exception:
            continue
        # A target layer must expose a spatial feature map — sanity check it.
        if not isinstance(module, torch.nn.Module):
            continue
        return candidate, module

    detected = auto_target_layer(model)
    return detected, resolve_module(model, detected)


def describe_model(model: torch.nn.Module) -> Dict[str, Any]:
    """Lightweight architecture description recorded in explanation metadata."""
    return {
        "class": type(model).__name__,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "training_mode": bool(model.training),
        "device": str(next(model.parameters()).device) if any(True for _ in model.parameters()) else "cpu",
    }


# ─────────────────────────────────────────────
#  FIGURE / PANEL HELPERS
# ─────────────────────────────────────────────
def figure_from_panels(panels: Sequence[Dict[str, Any]],
                       suptitle: Optional[str] = None,
                       panel_size: Tuple[float, float] = (4.6, 4.6),
                       facecolor: str = "#0a0f1e",
                       title_color: str = "#00d4ff",
                       label_color: str = "#8899aa") -> Image.Image:
    """
    Build a side-by-side PNG figure from labeled panels.

    `panels` items: {"title": str, "image": array, "cmap": str | None,
                     "colorbar_label": str | None, "vmin"/"vmax": float}
    The same dark look as the original gradcam.py figure is kept, but rendering
    uses matplotlib's buffer_rgba API (tostring_rgb was removed in matplotlib
    3.11 and silently broke the old helper).
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    count = max(1, len(panels))
    fig, axes = plt.subplots(1, count, figsize=(panel_size[0] * count, panel_size[1]),
                             squeeze=False)
    fig.patch.set_facecolor(facecolor)

    for ax, panel in zip(axes[0], panels):
        ax.set_facecolor(facecolor)
        image = np.asarray(panel["image"])
        cmap = panel.get("cmap")
        if image.ndim == 2 and cmap is None:
            cmap = "gray"
        artist = ax.imshow(image, cmap=cmap,
                           vmin=panel.get("vmin", 0.0 if cmap else None),
                           vmax=panel.get("vmax", 1.0 if cmap else None))
        ax.set_title(str(panel.get("title", "")), color=title_color,
                     fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")
        if panel.get("colorbar_label"):
            bar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
            bar.set_label(str(panel["colorbar_label"]), color=label_color, fontsize=9)
            bar.ax.yaxis.set_tick_params(color=label_color)
            plt.setp(bar.ax.yaxis.get_ticklabels(), color=label_color)

    if suptitle:
        fig.suptitle(str(suptitle), color="white", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(frame, mode="RGB")


# ─────────────────────────────────────────────
#  UNIFIED RESULT OBJECT
# ─────────────────────────────────────────────
@dataclass
class XAIResult:
    """
    Standardized container returned by Grad-CAM, LIME and SHAP.

    It carries both human-readable visuals (PIL images) and machine-readable
    attribution data (float maps), so the UI, the API and future fidelity /
    stability experiments can all consume the same object.
    """
    method: str
    modality: str
    model_file: str = ""
    predicted_class: str = ""
    predicted_class_index: int = -1
    confidence: float = 0.0
    target_class: str = ""
    target_class_index: int = -1
    target_probability: float = 0.0
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    explanation_image: Optional[Image.Image] = None                 # primary visual
    original_image: Optional[Image.Image] = None                    # untouched input
    attribution_map: Optional[np.ndarray] = None                    # normalized magnitude 0..1
    signed_attribution: Optional[np.ndarray] = None                 # signed map (target class)
    images: Dict[str, Image.Image] = field(default_factory=dict)    # extra views
    parameters: Dict[str, Any] = field(default_factory=dict)        # settings used
    metadata: Dict[str, Any] = field(default_factory=dict)          # layers/versions/timings
    warnings: List[str] = field(default_factory=list)

    # ── convenience accessors ───────────────────────────────────────
    @property
    def label(self) -> str:
        return METHOD_LABELS.get(self.method, str(self.method))

    @property
    def description(self) -> str:
        return METHOD_DESCRIPTIONS.get(self.method, "")

    @property
    def attribution(self) -> Optional[np.ndarray]:
        """Research-friendly alias for the normalized attribution map."""
        return self.attribution_map

    def properties(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_label": self.label,
            "modality": self.modality,
            "model_file": self.model_file,
            "predicted_class": self.predicted_class,
            "predicted_class_index": int(self.predicted_class_index),
            "confidence": round(float(self.confidence), 6),
            "target_class": self.target_class,
            "target_class_index": int(self.target_class_index),
            "target_probability": round(float(self.target_probability), 6),
            "class_probabilities": dict(self.class_probabilities),
            "parameters": _jsonable(self.parameters),
            "metadata": _jsonable(self.metadata),
            "warnings": list(self.warnings),
            "settings_fingerprint": settings_fingerprint(
                {"method": self.method, "modality": self.modality,
                 **{k: self.parameters.get(k) for k in sorted(self.parameters)}}),
        }

    def to_payload(self, image_keys: Optional[Sequence[str]] = None,
                   data_url: bool = True) -> Dict[str, Any]:
        """
        JSON-safe dict for the FastAPI backend / Streamlit frontend.

        Images are returned as base64 PNG strings; raw attribution arrays are
        not embedded (they stay in this object and on disk via `save`).
        """
        payload = self.properties()
        keys = list(image_keys) if image_keys else list(self.images.keys())
        payload["images"] = {key: encode_png_base64(self.images[key], data_url=data_url)
                             for key in keys if self.images.get(key) is not None}
        if self.explanation_image is not None:
            payload["explanation_image"] = encode_png_base64(self.explanation_image, data_url=data_url)
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_payload(), indent=indent, default=str)

    def save(self, out_dir: Any, stem: Optional[str] = None) -> Dict[str, str]:
        """
        Persist the explanation for research use: PNG views, raw `.npy`
        attribution maps and a JSON record of the exact settings used.
        """
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if stem:
            name = str(stem)
        else:
            fingerprint = image_fingerprint(self.original_image) if self.original_image is not None else "image"
            name = f"{self.modality}_{self.method}_{fingerprint}"
        written: Dict[str, str] = {}

        if self.explanation_image is not None:
            path = directory / f"{name}.png"
            self.explanation_image.save(path)
            written["explanation_image"] = str(path)
        for key, image in self.images.items():
            if image is None:
                continue
            path = directory / f"{name}__{key}.png"
            image.save(path)
            written[key] = str(path)
        if self.attribution_map is not None:
            path = directory / f"{name}__attribution.npy"
            np.save(path, self.attribution_map)
            written["attribution"] = str(path)
        if self.signed_attribution is not None:
            path = directory / f"{name}__signed.npy"
            np.save(path, self.signed_attribution)
            written["signed_attribution"] = str(path)
        record = directory / f"{name}.json"
        record.write_text(self.to_json(), encoding="utf-8")
        written["record"] = str(record)
        return written

    def __str__(self) -> str:  # compact console/log summary
        return (f"[XAI {self.label}] {self.modality} -> {self.predicted_class} "
                f"({self.confidence * 100:.1f}%) | target={self.target_class} "
                f"({self.target_probability * 100:.1f}%)")


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy / PIL / path values into JSON-safe types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_base_result(source: ModelSource,
                      method: str,
                      probabilities: np.ndarray,
                      original_image: Image.Image,
                      parameters: Dict[str, Any],
                      target_index: Optional[int] = None,
                      predicted_index: Optional[int] = None) -> XAIResult:
    """Fill the prediction / probability part shared by all three methods."""
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if predicted_index is None:
        predicted_index = int(np.argmax(probabilities))
    if target_index is None:
        target_index = predicted_index
    predicted_index = int(predicted_index)
    target_index = int(target_index)

    return XAIResult(
        method=method,
        modality=source.name,
        model_file=source.model_file,
        predicted_class=source.class_names[predicted_index],
        predicted_class_index=predicted_index,
        confidence=float(probabilities[predicted_index]),
        target_class=source.class_names[target_index],
        target_class_index=target_index,
        target_probability=float(probabilities[target_index]),
        class_probabilities=class_probability_dict(source.class_names, probabilities),
        original_image=to_rgb(original_image),
        parameters=dict(parameters),
        metadata={
            "model": describe_model(source.model),
            "input_size": list(display_size(source.transform)),
            "class_names": list(source.class_names),
            "preprocessing": ("the live medvision_inference transform "
                              "(Resize -> ToTensor -> Normalize)"),
        },
        warnings=list(source.warnings),
    )


def resolve_target_index(source: ModelSource, target_class: Any,
                         probabilities: np.ndarray) -> int:
    """target_class (name / index / None) -> concrete class index."""
    index = resolve_target_class(source.class_names, target_class)
    if index is None:
        index = int(np.argmax(np.asarray(probabilities).reshape(-1)))
    return int(index)


# ─────────────────────────────────────────────
#  MODEL SOURCE ADAPTER
# ─────────────────────────────────────────────
@dataclass
class ModelSource:
    """
    Normalized view of the live inference object.

    `medvision_inference.CNNModel` is the canonical source; anything exposing
    `.model`, `.transform` and `.class_names` works, and a bare `nn.Module` is
    accepted too (with a default ImageNet preprocessing).
    """
    model: torch.nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    class_names: List[str]
    name: str
    model_file: str
    meta: Dict[str, Any]
    device: torch.device
    warnings: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def default_transform(size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
                      mean: Sequence[float] = MEAN,
                      std: Sequence[float] = STD) -> Callable[[Image.Image], torch.Tensor]:
    """ImageNet ResNet50 preprocessing (only used if no live transform exists)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(list(mean), list(std)),
    ])


def infer_class_names(model: torch.nn.Module, num_classes: Optional[int] = None) -> List[str]:
    """Best-effort class-name fallback (raw indices) for a bare nn.Module."""
    if num_classes is None:
        head = getattr(model, "fc", None)
        if isinstance(head, torch.nn.Linear):
            num_classes = head.out_features
        elif isinstance(head, torch.nn.Sequential):
            for layer in reversed(list(head)):
                if isinstance(layer, torch.nn.Linear):
                    num_classes = layer.out_features
                    break
        if num_classes is None:
            num_classes = getattr(model, "num_classes", None)
    if not num_classes:
        raise ValueError("cannot infer the number of classes — pass a model object with `class_names`")
    return [f"class_{i}" for i in range(int(num_classes))]


def unwrap_model_source(source: Any) -> ModelSource:
    """`CNNModel` / `nn.Module` / dict -> `ModelSource` used by all explainers."""
    warnings: List[str] = []
    container = None

    if isinstance(source, torch.nn.Module):
        model = source
    elif isinstance(source, dict):
        container = source
        model = source.get("model")
    else:
        container = source
        model = getattr(source, "model", None)

    if not isinstance(model, torch.nn.Module):
        raise TypeError("explainers expect a medvision_inference.CNNModel, an nn.Module, "
                        "or a mapping containing 'model'")

    if isinstance(container, dict):
        transform = container.get("transform")
        class_names = list(container.get("class_names") or [])
        name = container.get("name") or container.get("modality")
        model_file = str(container.get("model_path") or container.get("model_file") or "")
        meta = dict(container.get("meta") or {})
    else:
        transform = getattr(container, "transform", None) if container is not None else None
        class_names = list(getattr(container, "class_names", None) or [])
        name = getattr(container, "name", None) if container is not None else None
        model_path = getattr(container, "model_path", None) if container is not None else None
        model_file = str(model_path) if model_path else ""
        meta = dict(getattr(container, "meta", None) or {})

    if transform is None:
        transform = default_transform()
        warnings.append("no live preprocessing transform supplied — "
                        "fell back to Resize(224,224) + ImageNet normalization")

    if not class_names:
        class_names = infer_class_names(model)
        warnings.append("no class names supplied — using positional class labels")

    return ModelSource(
        model=model,
        transform=transform,
        class_names=[str(c) for c in class_names],
        name=str(name or type(model).__name__),
        model_file=model_file,
        meta=meta,
        device=next(model.parameters()).device,
        warnings=warnings,
    )

