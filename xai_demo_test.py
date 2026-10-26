"""
MedVision AI — XAI demonstration test (LIME / SHAP validation, no training)
============================================================================

Runs ONE known-existing sample image per modality through the REAL application
code and verifies — with printed numbers, not just "no exception" — that:

  1. the normal prediction runs (`medvision_inference.CNNModel.predict`),
  2. LIME produces a genuine superpixel attribution from the live model,
  3. SHAP produces a genuine non-zero image attribution from the live model,
  4. the XAI pipeline sees the SAME probabilities as the normal prediction
     (identical preprocessing — the Eye V3 384x384 case included),
  5. a DIFFERENT image produces a DIFFERENT attribution (no stale cache),
  6. no checkpoint file changed (SHA-256 before vs after) and the prediction
     is identical before/after the explanations.

It NEVER trains, retrains, converts or writes a model file; it only READS
sample images that already exist in the project dataset folders.

Usage
-----
    python xai_demo_test.py                       # CHEST only (default, fastest)
    python xai_demo_test.py --all                 # all four modalities
    python xai_demo_test.py --modality eye
    python xai_demo_test.py --methods lime shap gradcam
    python xai_demo_test.py --quick               # cheaper settings (50 / 32)
    python xai_demo_test.py --image path\\to\\scan.png

Outputs
-------
    xai_outputs/demo/<modality>_<method>.png         primary overlay image
    xai_outputs/demo/<modality>_<method>_panel.png   multi-panel figure
    xai_outputs/demo/demo_summary.json               machine-readable summary

Exit code 0 = every check passed, 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Read-only sample-image locations (all matches are collected; first match is
#: the primary demo image, a different second match is used for the
#: "does the explanation change with the image?" check).
SAMPLE_PATTERNS = {
    "chest": [
        "cnn/chest_xray/chest_xray/test/NORMAL/*.jpeg",
        "cnn/chest_xray/chest_xray/test/NORMAL/*.jpg",
        "cnn/chest_xray/chest_xray/test/PNEUMONIA/*.jpeg",
        "cnn/chest_xray/chest_xray/test/PNEUMONIA/*.jpg",
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
        "cnn/skin/Skin cancer ISIC The International Skin Imaging Collaboration/Train/nevus/*.jpg",
    ],
    "eye": [
        "cnn/eye/test_images/test_images/*.png",
        "cnn/eye/fundus384/val/*/*.png",
        "cnn/eye/fundus384/test/*/*.png",
    ],
}

ALL_METHODS = ["lime", "shap", "gradcam"]
#: A normalized attribution map whose mean-absolute difference against another
#: image's map is below this is considered "suspiciously unchanged" (stale
#: cache / hardcoded output). Real LIME/SHAP maps for different scans differ
#: by far more than this.
CHANGE_CHECK_THRESHOLD = 0.005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_samples(modality: str, override: Optional[str] = None) -> List[Path]:
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"sample image not found: {path}")
        return [path]
    found: List[Path] = []
    for pattern in SAMPLE_PATTERNS.get(modality, []):
        for candidate in sorted(ROOT.glob(pattern)):
            if candidate.is_file() and candidate not in found:
                found.append(candidate)
        if found:
            break
    if not found:
        raise FileNotFoundError(
            f"no sample image found for modality '{modality}' — pass --image explicitly")
    return found


def attribution_stats(array: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(array, dtype=np.float64).reshape(-1)
    nonzero = float(np.count_nonzero(arr)) / max(1, arr.size) * 100.0
    return {"min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "nonzero_pct": nonzero}


def fmt_stats(stats: Dict[str, float], indent: str = "  ") -> str:
    return (f"{indent}attribution min      : {stats['min']:.6f}\n"
            f"{indent}attribution max      : {stats['max']:.6f}\n"
            f"{indent}attribution mean     : {stats['mean']:.6f}\n"
            f"{indent}attribution std      : {stats['std']:.6f}\n"
            f"{indent}non-zero attribution : {stats['nonzero_pct']:.1f}%")


def probs_from_prediction(prediction: dict, class_names: Sequence[str]) -> np.ndarray:
    return np.asarray([prediction["all_scores"][name] for name in class_names],
                      dtype=np.float64)


def save_views(result, out_dir: Path, stem: str) -> Dict[str, str]:
    """Save the primary overlay + the multi-panel figure; return saved paths."""
    saved: Dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = result.explanation_image or result.images.get("overlay")
    if primary is not None:
        path = out_dir / f"{stem}.png"
        primary.save(path, format="PNG")
        saved["overlay"] = str(path)
    panel = result.images.get("panel")
    if panel is not None:
        path = out_dir / f"{stem}_panel.png"
        panel.save(path, format="PNG")
        saved["panel"] = str(path)
    return saved


def check_integrity(before: Dict[str, str], after: Dict[str, str]) -> List[str]:
    problems = []
    for key, digest_before in before.items():
        digest_after = after.get(key)
        if digest_after != digest_before:
            problems.append(f"checkpoint changed: {key} "
                            f"{digest_before[:12]} -> {str(digest_after)[:12]}")
    return problems


def run_modality(modality: str, methods: List[str], args, summary: dict) -> List[str]:
    """Run prediction + the requested explainers for one modality."""
    from medvision_inference import CNNModel, MODEL_PATHS
    from xai import explain_image
    from xai.common import build_numpy_predict_fn, display_size, prep_xai_image

    errors: List[str] = []
    print("\n" + "=" * 72)
    print(f"MODALITY: {modality.upper()}")
    print("=" * 72)

    samples = find_samples(modality, args.image)
    sample = samples[0]
    print(f"  image path   : {sample.relative_to(ROOT)}")
    model_path = MODEL_PATHS[modality]
    if not model_path.exists():
        errors.append(f"{modality}: model not found: {model_path}")
        print(f"  [FAIL] {errors[-1]}")
        return errors

    hash_before = sha256_file(model_path)
    cnn = CNNModel(modality)
    image = Image.open(sample)

    baseline = cnn.predict(image)
    print(f"  model        : {model_path.name} | device={cnn.device} | "
          f"input size={display_size(cnn.transform)}")
    print(f"  prediction   : {baseline['prediction']} "
          f"({baseline['confidence'] * 100:.1f}%)")
    print(f"  all scores   : {baseline['all_scores']}")

    # ── TASK 6: SAME-IMAGE / SAME-PIPELINE CHECK ─────────────────────────
    size = display_size(cnn.transform)
    xai_image = prep_xai_image(image, size)
    predict_fn = build_numpy_predict_fn(cnn.model, cnn.transform, cnn.device)
    probs_xai = predict_fn(xai_image[None, ...])[0]
    probs_pred = probs_from_prediction(baseline, cnn.class_names)
    pipeline_diff = float(np.max(np.abs(probs_xai - probs_pred)))
    pipeline_ok = pipeline_diff <= 1e-4
    print(f"  same-pipeline check: max|predict() - XAI predict_fn| = "
          f"{pipeline_diff:.3e} -> {'PASS' if pipeline_ok else 'FAIL'}")
    if not pipeline_ok:
        errors.append(f"{modality}: XAI preprocessing diverges from prediction "
                      f"(max abs diff {pipeline_diff:.3e})")

    entry = {
        "modality": modality,
        "image": str(sample.relative_to(ROOT)),
        "model": model_path.name,
        "predicted_class": baseline["prediction"],
        "confidence": baseline["confidence"],
        "all_scores": baseline["all_scores"],
        "input_size": list(size),
        "same_pipeline_max_abs_diff": pipeline_diff,
        "same_pipeline_pass": pipeline_ok,
        "methods": {},
    }

    # ── PRIMARY RUN: explanation for the selected image ──────────────────
    primary_maps: Dict[str, np.ndarray] = {}
    for method in methods:
        params = {"make_panel": True}
        if method == "lime":
            params.update(num_samples=args.lime_samples, batch_size=16,
                          num_features=8, random_seed=42, include_negative=True)
        elif method == "shap":
            params.update(max_evals=args.shap_evals, batch_size=16,
                          random_seed=42, prefer="partition")

        started = time.perf_counter()
        try:
            result = explain_image(cnn, image, method=method, **params)
        except Exception as exc:                       # noqa: BLE001
            elapsed = time.perf_counter() - started
            errors.append(f"{modality}/{method}: raised {type(exc).__name__}: {exc}")
            print(f"\n{method.upper()} — EXECUTION FAILED after {elapsed:.2f}s: "
                  f"{type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started

        if result.predicted_class != baseline["prediction"]:
            errors.append(f"{modality}/{method}: explained prediction "
                          f"({result.predicted_class}) != displayed prediction "
                          f"({baseline['prediction']})")
        if abs(result.confidence - baseline["confidence"]) > 1e-3:
            errors.append(f"{modality}/{method}: confidence changed "
                          f"({baseline['confidence']} -> {result.confidence})")

        signed = np.asarray(result.signed_attribution, dtype=np.float32)
        norm = np.asarray(result.attribution_map, dtype=np.float32)
        signed_stats = attribution_stats(signed)
        norm_stats = attribution_stats(norm)
        saved = save_views(result, Path(args.out), f"{modality}_{method}")
        if not saved:
            errors.append(f"{modality}/{method}: no visualization was produced")

        method_entry = {
            "status": "OK",
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "target_class": result.target_class,
            "target_class_index": int(result.target_class_index),
            "signed_stats": signed_stats,
            "normalized_stats": norm_stats,
            "execution_seconds": round(elapsed, 3),
            "saved_files": saved,
            "warnings": list(result.warnings),
            "metadata": {k: result.metadata.get(k) for k in
                         ("num_superpixels", "intercept", "local_ridge_score",
                          "explainer_used", "model_evaluations", "input_size",
                          "elapsed_seconds") if k in result.metadata},
        }

        if method == "lime":
            method_errors = print_lime_report(result, signed_stats, norm_stats,
                                              elapsed, modality)
        elif method == "shap":
            method_errors = print_shap_report(result, signed_stats, norm_stats,
                                              elapsed, modality)
        else:
            method_errors = []
            print(f"\nGRADCAM ({modality})")
            print(f"  Prediction     : {result.predicted_class} "
                  f"({result.confidence * 100:.1f}%)")
            print(f"  Target layer   : {result.metadata.get('target_layer')}")
            print(fmt_stats(norm_stats))
            print(f"  Execution time : {elapsed:.2f}s")
            if norm_stats["nonzero_pct"] <= 0.5:
                method_errors.append(
                    f"{modality}/gradcam: attribution is degenerate "
                    f"({norm_stats['nonzero_pct']:.1f}% non-zero)")

        if method_errors:
            method_entry["status"] = "DEGENERATE"
        errors.extend(method_errors)
        primary_maps[method] = norm
        entry["methods"][method] = method_entry

    # ── CHANGE CHECK: a different image must yield a different map ──────
    if len(samples) > 1 and not args.no_change_check:
        other = samples[1]
        other_image = Image.open(other)
        other_pred = cnn.predict(other_image)
        print(f"\n  change check image: {other.relative_to(ROOT)} "
              f"-> {other_pred['prediction']} ({other_pred['confidence'] * 100:.1f}%)")
        entry["change_check"] = {"image": str(other.relative_to(ROOT)),
                                 "predicted_class": other_pred["prediction"],
                                 "methods": {}}
        for method in methods:
            if method not in primary_maps:
                continue
            params = {"make_panel": False}
            if method == "lime":
                params.update(num_samples=args.lime_samples, batch_size=16,
                              num_features=8, random_seed=42, include_negative=True)
            elif method == "shap":
                params.update(max_evals=args.shap_evals, batch_size=16,
                              random_seed=42, prefer="partition")
            try:
                other_result = explain_image(cnn, other_image, method=method, **params)
            except Exception as exc:                   # noqa: BLE001
                errors.append(f"{modality}/{method}: change check failed: {exc}")
                continue
            other_map = np.asarray(other_result.attribution_map, dtype=np.float32)
            mad = float(np.mean(np.abs(primary_maps[method] - other_map)))
            changed = mad > CHANGE_CHECK_THRESHOLD
            print(f"    {method:8} mean|d attribution| = {mad:.4f} -> "
                  f"{'CHANGED (PASS)' if changed else 'UNCHANGED (FAIL)'}")
            entry["change_check"]["methods"][method] = {
                "mean_abs_diff": mad, "changed": changed}
            if not changed:
                errors.append(f"{modality}/{method}: attribution did not change "
                              f"for a different image (mean|d|={mad:.6f}) — "
                              f"possible stale cache")
    elif not args.no_change_check:
        print("\n  change check: skipped (only one sample image available)")

    # ── TASK 7: prediction + weights unchanged after XAI ────────────────
    after = cnn.predict(image)
    if after != baseline:
        errors.append(f"{modality}: prediction changed after running explanations")
    hash_after = sha256_file(model_path)
    errors.extend([f"{modality}: {msg}" for msg in
                   check_integrity({model_path.name: hash_before},
                                   {model_path.name: hash_after})])
    print("\n  prediction after XAI : "
          f"{'unchanged (PASS)' if after == baseline else 'CHANGED (FAIL)'}")
    print(f"  checkpoint sha256    : {hash_after[:16]}... "
          f"{'unchanged (PASS)' if hash_before == hash_after else 'CHANGED (FAIL)'}")

    entry["checkpoint_sha256_before"] = hash_before
    entry["checkpoint_sha256_after"] = hash_after
    entry["prediction_unchanged"] = after == baseline
    summary["modalities"].append(entry)
    return errors


def print_lime_report(result, signed_stats, norm_stats, elapsed,
                      modality: str) -> List[str]:
    """TASK 3 — LIME validation report. Returns list of failures."""
    errors: List[str] = []
    weighted = list(result.metadata.get("top_weighted_superpixels", []))
    weights = {int(seg): float(w) for seg, w in weighted}
    n_superpixels = int(result.metadata.get("num_superpixels", 0))
    n_features = len(weighted)
    # Regions with strictly positive / negative weight (piecewise-constant map).
    signed = np.asarray(result.signed_attribution, dtype=np.float32)
    unique_values = np.unique(np.round(signed.astype(np.float64), 9))
    n_positive_regions = int(np.sum(unique_values > 0))
    n_negative_regions = int(np.sum(unique_values < 0))

    print(f"\nLIME ({modality})")
    print(f"  Prediction                   : {result.predicted_class} "
          f"({result.confidence * 100:.1f}%)")
    print(f"  Explained class              : {result.target_class} "
          f"(index {result.target_class_index})")
    print(f"  Number of superpixels        : {n_superpixels}")
    print(f"  Number of explained features : {n_features}")
    print(f"  Positive attribution features: "
          f"{sum(1 for w in weights.values() if w > 0)}")
    print(f"  Negative attribution features: "
          f"{sum(1 for w in weights.values() if w < 0)}")
    print(f"  Positive attribution regions : {n_positive_regions}")
    print(f"  Negative attribution regions : {n_negative_regions}")
    print(fmt_stats(signed_stats))
    print(f"  Non-zero in visual map       : {norm_stats['nonzero_pct']:.1f}%")
    print(f"  Execution time               : {elapsed:.2f}s")

    if n_superpixels < 2:
        errors.append(f"{modality}/lime: segmentation produced "
                      f"{n_superpixels} superpixel(s) — degenerate LIME input")
    if n_features < 1:
        errors.append(f"{modality}/lime: no features were explained")
    if signed_stats["std"] <= 1e-12:
        errors.append(f"{modality}/lime: attribution is constant "
                      f"(std={signed_stats['std']:.3e}) — DEGENERATE")
    if norm_stats["nonzero_pct"] <= 0.5:
        errors.append(f"{modality}/lime: only "
                      f"{norm_stats['nonzero_pct']:.1f}% of the visualization "
                      f"is non-zero — DEGENERATE")
    return errors


def print_shap_report(result, signed_stats, norm_stats, elapsed,
                      modality: str) -> List[str]:
    """TASK 4 — SHAP validation report. Returns list of failures."""
    errors: List[str] = []
    explainer_used = result.metadata.get("explainer_used")
    print(f"\nSHAP ({modality})")
    print(f"  Prediction           : {result.predicted_class} "
          f"({result.confidence * 100:.1f}%)")
    print(f"  Explained class      : {result.target_class} "
          f"(index {result.target_class_index})")
    print(f"  Explainer            : {explainer_used} | "
          f"model evaluations: {result.metadata.get('model_evaluations')}")
    if result.metadata.get("fallback_reason"):
        print(f"  FALLBACK             : {result.metadata['fallback_reason']}")
    print(fmt_stats(signed_stats))
    print(f"  Non-zero in visual map: {norm_stats['nonzero_pct']:.1f}%")
    print(f"  Execution time       : {elapsed:.2f}s")

    if explainer_used not in ("partition", "gradient"):
        errors.append(f"{modality}/shap: no SHAP explainer actually ran")
    if signed_stats["std"] <= 1e-12:
        errors.append(f"{modality}/shap: attribution is constant "
                      f"(std={signed_stats['std']:.3e}) — DEGENERATE")
    if norm_stats["nonzero_pct"] <= 0.5:
        errors.append(f"{modality}/shap: only "
                      f"{norm_stats['nonzero_pct']:.1f}% of the visualization "
                      f"is non-zero — DEGENERATE")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MedVision AI XAI demonstration test (LIME/SHAP/Grad-CAM, "
                    "no training, no checkpoint writes).")
    parser.add_argument("--modality", default="chest",
                        help="chest | brain | skin | eye | all (default: chest)")
    parser.add_argument("--all", action="store_true",
                        help="shortcut for --modality all (runs all 4 modalities)")
    parser.add_argument("--methods", nargs="*", default=["lime", "shap"],
                        choices=ALL_METHODS,
                        help="methods to run (default: lime shap)")
    parser.add_argument("--image", default=None,
                        help="explicit primary sample image path")
    parser.add_argument("--quick", action="store_true",
                        help="cheaper settings (LIME 50 samples / SHAP 32 evals)")
    parser.add_argument("--lime-samples", type=int, default=None)
    parser.add_argument("--shap-evals", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "xai_outputs" / "demo"),
                        help="directory for generated explanation images")
    parser.add_argument("--no-change-check", action="store_true",
                        help="skip the 'different image -> different map' check")
    args = parser.parse_args()

    args.lime_samples = args.lime_samples or (50 if args.quick else 200)
    args.shap_evals = args.shap_evals or (32 if args.quick else 128)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    modalities = (["chest", "brain", "skin", "eye"]
                  if (args.all or args.modality == "all") else [args.modality])

    from medvision_inference import MODEL_PATHS
    watched = {name: path for name, path in MODEL_PATHS.items() if path.exists()}
    hashes_before = {name: sha256_file(path) for name, path in watched.items()}

    summary = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {"lime_num_samples": args.lime_samples,
                     "shap_max_evals": args.shap_evals,
                     "methods": args.methods},
        "checkpoint_sha256_before": hashes_before,
        "modalities": [],
    }

    errors: List[str] = []
    for modality in modalities:
        errors.extend(run_modality(modality, list(args.methods), args, summary))

    hashes_after = {name: sha256_file(path) for name, path in watched.items()}
    integrity_errors = check_integrity(hashes_before, hashes_after)
    errors.extend(integrity_errors)
    summary["checkpoint_sha256_after"] = hashes_after
    summary["model_integrity_pass"] = not integrity_errors
    summary["errors"] = errors
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    summary_path = Path(args.out) / "demo_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("\n" + "=" * 72)
    print("MODEL INTEGRITY: " + ("PASS — no checkpoint file was modified"
                                 if not integrity_errors else "FAIL"))
    print(f"Summary written to: {summary_path}")
    print(f"Explanation images in: {Path(args.out).resolve()}")
    if errors:
        print(f"\nXAI DEMO TEST FAILED — {len(errors)} problem(s):")
        for message in errors:
            print(f"  - {message}")
        return 1
    print("\nXAI DEMO TEST PASSED — LIME/SHAP explanations are generated from "
          "the live model,\nare non-degenerate, match the displayed prediction, "
          "and change with the image.")
    print("NO MODEL TRAINING WAS PERFORMED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())








ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
