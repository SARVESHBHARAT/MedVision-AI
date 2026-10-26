"""
MedVision AI — XAI smoke test (validation only)
==============================================

Runs Grad-CAM, LIME and SHAP on ONE existing sample image per modality and
verifies that the unified XAI layer is wired correctly.

What it checks
--------------
1. checkpoint hash + parameter checksum are unchanged before/after
   (proof that no weight was modified),
2. the model prediction is identical before/after the explanations,
3. Grad-CAM leaves no hooks behind on its target layer,
4. attribution maps / explanation images / payloads have the expected
   shapes and types and are JSON-serializable,
5. the research artefacts (`save`) are written.

It NEVER trains, retrains, converts or writes a model file, and it only READS a
sample image that already exists in the project dataset folders.

Usage
-----
    python xai_smoke_test.py --modality chest
    python xai_smoke_test.py --modality all --quick
    python xai_smoke_test.py --modality eye --method gradcam shap --image path\\to\\scan.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Read-only sample-image locations (first match wins).
SAMPLE_PATTERNS = {
    "chest": [
        "cnn/chest_xray/chest_xray/test/NORMAL/*.jpeg",
        "cnn/chest_xray/chest_xray/test/NORMAL/*.jpg",
        "cnn/chest_xray/chest_xray/test/PNEUMONIA/*.jpeg",
        "cnn/chest_xray/chest_xray/val/NORMAL/*.jpeg",
    ],
    "brain": [
        "cnn/brain/Testing/notumor/*.jpg",
        "cnn/brain/Testing/glioma/*.jpg",
        "cnn/brain/Testing/meningioma/*.jpg",
        "cnn/brain/Testing/pituitary/*.jpg",
    ],
    "skin": [
        "cnn/skin/Skin cancer ISIC The International Skin Imaging Collaboration/Test/nevus/*.jpg",
        "cnn/skin/Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/*.jpg",
    ],
    "eye": [
        "cnn/eye/test_images/test_images/*.png",
        "cnn/eye/fundus384/val/*/*.png",
        "cnn/eye/fundus384/test/*/*.png",
    ],
}

METHODS = ["gradcam", "lime", "shap"]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_checksum(model: torch.nn.Module) -> float:
    """Order-independent scalar checksum of all parameters (drift detector)."""
    total = 0.0
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            data = tensor.detach().float().cpu()
            total += float(data.sum()) * (1.0 + (len(name) % 7))
    return total


def hook_counts(module: torch.nn.Module) -> dict:
    """Number of hooks currently registered on a module."""
    return {
        "forward": len(module._forward_hooks),
        "forward_pre": len(module._forward_pre_hooks),
        "backward": len(module._backward_hooks),
        "full_backward": len(getattr(module, "_full_backward_hooks", {})),
    }


def find_sample(modality: str, override: str | None = None) -> Path:
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"sample image not found: {path}")
        return path
    for pattern in SAMPLE_PATTERNS.get(modality, []):
        for candidate in sorted(ROOT.glob(pattern)):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"no sample image found for modality '{modality}' — pass --image explicitly")


def _fail(errors: list, message: str) -> None:
    errors.append(message)
    print(f"    [FAIL] {message}")


def cnn_model_explain(cnn, image, method: str, params: dict):
    """Explain through the live CNNModel via the unified dispatcher."""
    from xai import explain_image
    return explain_image(cnn, image, method=method, **params)


def check_modality(modality: str, methods: list, args) -> list:
    """Run every requested method for one modality and return a list of errors."""
    from medvision_inference import CNNModel, MODEL_PATHS
    from xai.common import display_size, resolve_target_layer

    errors: list = []
    print(f"\n=== {modality.upper()} ===")

    sample = find_sample(modality, args.image)
    print(f"  sample image : {sample.relative_to(ROOT)}")
    model_path = MODEL_PATHS[modality]
    if not model_path.exists():
        _fail(errors, f"model not found: {model_path}")
        return errors

    before_md5 = md5(model_path)

    load_started = time.perf_counter()
    cnn = CNNModel(modality)
    print(f"  model        : {model_path.name} (loaded in {time.perf_counter() - load_started:.2f}s) "
          f"| classes={cnn.num_classes} | device={cnn.device}")

    image = Image.open(sample)
    baseline_prediction = cnn.predict(image)
    baseline_checksum = parameter_checksum(cnn.model)
    layer_path, layer_module = resolve_target_layer(cnn.model, modality=modality)
    baseline_hooks = hook_counts(layer_module)
    print(f"  target layer : {layer_path} | input {display_size(cnn.transform)}")
    print(f"  prediction   : {baseline_prediction['prediction']} "
          f"({baseline_prediction['confidence'] * 100:.1f}%)")

    for method in methods:
        params = {"make_panel": True}
        if method == "lime":
            params.update(num_samples=args.lime_samples, batch_size=16, num_features=6,
                          random_seed=42, include_negative=True)
        elif method == "shap":
            params.update(max_evals=args.shap_evals, batch_size=16, random_seed=42,
                          prefer="partition")

        started = time.perf_counter()
        try:
            result = cnn_model_explain(cnn, image, method, params)
        except Exception as exc:                       # noqa: BLE001 - report, keep testing
            _fail(errors, f"{method}: raised {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started

        errors.extend(_check_result(result, baseline_prediction, cnn, method, args))
        print(f"  [{method:7}] {result} | {elapsed:.2f}s "
              f"| attribution {tuple(result.attribution_map.shape)} "
              f"| detail={result.metadata.get('explainer_used') or result.metadata.get('target_layer')}")
        for warning in result.warnings:
            print(f"              note: {warning}")

    after_prediction = cnn.predict(image)
    after_checksum = parameter_checksum(cnn.model)
    after_hooks = hook_counts(layer_module)
    after_md5 = md5(model_path)

    if baseline_prediction != after_prediction:
        _fail(errors, "prediction changed after running explanations")
    if abs(baseline_checksum - after_checksum) > 0.0:
        _fail(errors, "parameter checksum changed — model weights were modified!")
    if after_hooks != baseline_hooks:
        _fail(errors, f"hooks leaked: {baseline_hooks} -> {after_hooks}")
    if before_md5 != after_md5:
        _fail(errors, f"checkpoint file changed: {before_md5} -> {after_md5}")

    print(f"  weights      : checksum {baseline_checksum:.3f} -> {after_checksum:.3f}, "
          f"checkpoint md5 {after_md5[:12]} (unchanged)")
    print(f"  hooks        : {baseline_hooks} -> {after_hooks} (unchanged)")
    print("  prediction   : unchanged after XAI")
    return errors


def _check_result(result, baseline_prediction, cnn, method: str, args) -> list:
    """Shape / type / payload / artefact checks for one explanation result."""
    from xai.common import display_size

    errors: list = []
    size = display_size(cnn.transform)
    attribution = result.attribution_map

    if result.predicted_class != baseline_prediction["prediction"]:
        _fail(errors, f"{method}: predicted class changed "
                      f"({baseline_prediction['prediction']} -> {result.predicted_class})")
    if abs(result.confidence - baseline_prediction["confidence"]) > 1e-4:
        _fail(errors, f"{method}: confidence changed "
                      f"({baseline_prediction['confidence']} -> {result.confidence})")
    if attribution is None or attribution.shape != size:
        _fail(errors, f"{method}: attribution map shape "
                      f"{None if attribution is None else attribution.shape} != {size}")
    elif attribution.dtype != np.float32:
        _fail(errors, f"{method}: attribution dtype {attribution.dtype} != float32")
    elif float(attribution.min()) < 0.0 or float(attribution.max()) > 1.0 + 1e-6:
        _fail(errors, f"{method}: attribution outside [0,1] "
                      f"({attribution.min():.4f}..{attribution.max():.4f})")
    if result.signed_attribution is None or result.signed_attribution.shape != size:
        _fail(errors, f"{method}: signed attribution missing or wrong shape")
    if result.explanation_image is None or result.explanation_image.mode != "RGB":
        _fail(errors, f"{method}: explanation image missing / not RGB")
    if len(result.images) < 3:
        _fail(errors, f"{method}: expected at least 3 views, got {sorted(result.images)}")

    try:
        payload = result.to_payload()
        json.dumps(payload)
        for key in ("method", "predicted_class", "confidence", "images", "parameters", "metadata"):
            if key not in payload:
                _fail(errors, f"{method}: payload missing '{key}'")
        if not payload.get("images"):
            _fail(errors, f"{method}: payload contains no images")
    except Exception as exc:                       # noqa: BLE001
        _fail(errors, f"{method}: payload not JSON-serializable: {exc}")

    try:
        saved = result.save(args.out, stem=f"smoke_{result.modality}_{method}")
        if not Path(saved["attribution"]).exists():
            _fail(errors, f"{method}: attribution .npy was not written")
    except Exception as exc:                       # noqa: BLE001
        _fail(errors, f"{method}: could not save artefacts: {exc}")
    return errors


def check_engine_path(modality: str, args) -> list:
    """Sanity-check the engine-level API the FastAPI endpoints use."""
    from medvision_inference import MedVisionEngine

    errors: list = []
    print(f"\n=== ENGINE PATH ({modality}) ===")
    engine = MedVisionEngine(load_models=[modality])
    methods = engine.xai_methods().get("methods", {})
    print(f"  available    : {[m for m, info in methods.items() if info.get('available')]}")
    if not methods:
        _fail(errors, "engine.xai_methods() reported no methods")
        return errors

    image = Image.open(find_sample(modality, args.image))
    payload = engine.explain_image(image, modality, method="gradcam", make_panel=False)
    if payload.get("error"):
        _fail(errors, f"engine.explain_image error: {payload['error']}")
        return errors
    print(f"  gradcam      : {payload['predicted_class']} "
          f"({payload['confidence'] * 100:.1f}%) | views={sorted(payload['images'])}")
    for key in ("method", "predicted_class", "confidence", "images", "parameters"):
        if key not in payload:
            _fail(errors, f"payload missing key '{key}'")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="MedVision AI XAI smoke test (no training).")
    parser.add_argument("--modality", default="chest",
                        help="chest | brain | skin | eye | all (default: chest)")
    parser.add_argument("--method", nargs="*", default=METHODS, choices=METHODS,
                        help="methods to run (default: all three)")
    parser.add_argument("--image", default=None, help="explicit sample image path")
    parser.add_argument("--quick", action="store_true", help="cheapest settings (fast sanity run)")
    parser.add_argument("--lime-samples", type=int, default=None)
    parser.add_argument("--shap-evals", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "xai_outputs" / "smoke"),
                        help="where explanation artefacts are written")
    parser.add_argument("--engine", action="store_true",
                        help="also exercise MedVisionEngine.explain_image")
    args = parser.parse_args()

    args.lime_samples = args.lime_samples or (50 if args.quick else 200)
    args.shap_evals = args.shap_evals or (32 if args.quick else 128)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    modalities = ["chest", "brain", "skin", "eye"] if args.modality == "all" else [args.modality]
    errors: list = []
    for modality in modalities:
        errors.extend(check_modality(modality, args.method, args))
    if args.engine:
        errors.extend(check_engine_path(modalities[0], args))

    print("\n" + "=" * 68)
    if errors:
        print(f"XAI SMOKE TEST FAILED — {len(errors)} problem(s):")
        for message in errors:
            print(f"  - {message}")
        return 1
    print("XAI SMOKE TEST PASSED — explanations work, weights and predictions unchanged.")
    print("NO MODEL TRAINING WAS PERFORMED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
