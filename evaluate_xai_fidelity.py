"""MedVision AI - XAI fidelity (deletion) evaluation, read-only.

Protocol (fixed, no tuning on test data):
 1. Load one frozen modality model (no training, no weight writes).
 2. For each test image: predict original class + target probability.
 3. Attribution for ORIGINAL predicted class via existing xai.explain_image.
 4. Min-max normalize map to [0,1], rank pixels descending.
 5. Mask top fractions [0,.1,.2,.3,.4,.5] with FIXED blur baseline,
    re-run SAME model + SAME transform.
 6. Record target probability, drop, deletion AUC.

Usage:
 .\\.venv\\Scripts\\python.exe evaluate_xai_fidelity.py --modality chest --method gradcam --max-samples 2 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "xai_outputs" / "fidelity"
FRACTIONS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
METHODS = ("gradcam", "lime", "shap")
MODALITIES = ("chest", "brain", "skin", "eye")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _is_valid_test_image(p: Path) -> bool:
    """Reject macOS metadata and non-image files (read-only check).

    - any path component named __MACOSX
    - filenames starting with ._ (AppleDouble resource forks)
    - files PIL cannot open/verify as a real image
    """
    if "__MACOSX" in p.parts:
        return False
    if p.name.startswith("._"):
        return False
    if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
        return False
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def find_test_images(modality: str, max_samples: int | None, seed: int):
    """Test-only images, deterministic (sorted + seeded shuffle).

    max_samples=None -> ALL valid test images (CLI default);
    an int N -> first N after the seeded shuffle (smoke tests).
    """
    candidates = []
    cnn_dir = ROOT / "cnn"
    for base in sorted(cnn_dir.glob(f"{modality}*")):
        if base.is_file():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS and "test" in p.parts:
                candidates.append(p)
    if not candidates:
        for p in sorted(cnn_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS and "test" in p.parts \
                    and modality in str(p).lower():
                candidates.append(p)
    # Filter once so both branches (all modalities) skip macOS metadata and
    # invalid image files instead of failing mid-run.
    candidates = sorted({p for p in candidates if _is_valid_test_image(p)})
    # Deduplicate by CONTENT identity (SHA-256), not filename. Order by path
    # depth first so the shallowest (canonical) copy of a duplicated test tree
    # wins, e.g. cnn/.../chest_xray/test/ over cnn/.../chest_xray/chest_xray
    # /test/. Same-depth files keep lexicographic order (seed-stable for
    # non-duplicated datasets). Dataset files are never touched.
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    seen_hashes: set = set()
    unique: list = []
    for p in candidates:
        digest = sha256_file(p)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique.append(p)
    candidates = unique
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_samples]
def blurred_baseline(arr01: np.ndarray) -> np.ndarray:
    """Deterministic neutral baseline: heavy blur of model-input image."""
    try:
        import cv2
        u8 = (np.clip(arr01, 0, 1) * 255).round().astype(np.uint8)
        k = max(3, (min(arr01.shape[:2]) // 4) * 2 + 1)
        blur = cv2.GaussianBlur(u8, (k, k), 0)
        return blur.astype(np.float32) / 255.0
    except Exception:
        mean = float(np.clip(arr01, 0, 1).mean())
        return np.full_like(arr01, mean, dtype=np.float32)


def normalize01(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m.mean(axis=-1)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(m.min()), float(m.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - lo) / (hi - lo)).astype(np.float32)


def save_plot(mdir, modality, method, agg, auc, n) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([a["mask_fraction"] * 100 for a in agg],
                [a["probability_mean"] for a in agg], marker="o")
        ax.set_xlabel("Masked top-attribution fraction (%)")
        ax.set_ylabel("Mean target-class probability")
        ax.set_title(f"Deletion curve: {modality}/{method} (n={n}, AUC={auc:.3f})")
        ax.set_xticks([f * 100 for f in FRACTIONS])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(mdir) / "deletion_curve.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[fidelity] plot skipped for {method}: {e}")

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def smooth_importance(m: np.ndarray) -> np.ndarray:
    """Fixed, deterministic spatial smoothing (same for every XAI method).

    Ranking raw per-pixel attributions produces salt-and-pepper masks; a mild
    Gaussian blur (kernel ~5% of the shorter side) groups nearby important
    pixels into contiguous regions.  Applied identically to Grad-CAM, LIME and
    SHAP so the masking operation stays consistent across methods.
    """
    try:
        import cv2
        k = max(3, (min(m.shape[:2]) // 20) * 2 + 1)
        return cv2.GaussianBlur(np.asarray(m, dtype=np.float32), (k, k), 0)
    except Exception:
        return np.asarray(m, dtype=np.float32)


def resize_to(m: np.ndarray, hw) -> np.ndarray:
    """Make an attribution map match the model-input grid (H, W)."""
    arr = np.asarray(m, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.shape == (int(hw[0]), int(hw[1])):
        return arr
    import cv2
    return cv2.resize(arr, (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_LINEAR)


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _trapezoid(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2 renamed it
    return float(fn(y, x))


def deletion_auc(fracs, probs) -> float:
    f = np.asarray(fracs, dtype=float)
    p = np.asarray(probs, dtype=float)
    span = f.max() - f.min()
    if span <= 0:
        return float(p.mean())
    return _trapezoid(p, f) / span


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "median": None, "count": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
            "median": float(np.median(arr)), "count": int(arr.size)}


def _env_info() -> dict:
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "torch", "cv2", "matplotlib", "lime", "shap"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[name] = None
    return info



# ─────────────────────────────────────────────
#  ONE METHOD x ONE MODALITY
# ─────────────────────────────────────────────
def evaluate(modality: str, method: str, max_samples: int | None, seed: int) -> dict:
    """Deletion experiment for one method on one modality.  Read-only."""
    from PIL import Image

    from medvision_inference import CNNModel, MODEL_PATHS
    from xai import DEFAULT_PARAMS, explain_image
    from xai.common import build_numpy_predict_fn, display_size, prep_xai_image

    set_seeds(seed)
    out_dir = OUT_ROOT / modality / method
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fidelity] {modality} / {method} — loading frozen model ...")
    cnn = CNNModel(modality)
    model_path = Path(MODEL_PATHS[modality])
    sha_before = sha256_file(model_path)

    size = display_size(cnn.transform)                 # (H, W) model-input grid
    prob_fn = build_numpy_predict_fn(cnn.model, cnn.transform)

    samples = find_test_images(modality, max_samples, seed)
    if not samples:
        raise SystemExit(f"[fidelity] no test images found for modality '{modality}'")

    # Fixed XAI settings (never tuned on test results; recorded in config).
    xai_params = {"make_panel": False}
    if method in ("lime", "shap"):
        xai_params.update({"random_seed": seed})

    rows, per_sample_json = [], []
    effective_params = None

    for index, path in enumerate(samples, 1):
        image = Image.open(path).convert("RGB")
        pred = cnn.predict(image)
        predicted_class = pred["prediction"]
        target_idx = cnn.class_names.index(predicted_class)
        orig_prob = float(pred["all_scores"][predicted_class])

        result = explain_image(cnn, image, method=method,
                               target_class=target_idx, **xai_params)
        if effective_params is None:
            effective_params = dict(result.parameters or {})

        base = prep_xai_image(image, size)             # float [0,1], model grid
        importance = normalize01(resize_to(result.attribution_map, size))
        importance = normalize01(smooth_importance(importance))
        baseline = blurred_baseline(base)

        h, w = importance.shape
        flat_order = np.argsort(-importance.reshape(-1), kind="stable")
        curve = []
        for frac in FRACTIONS:
            k = int(round(frac * h * w))
            mask = np.zeros(h * w, dtype=bool)
            if k > 0:
                mask[flat_order[:k]] = True
            mask = mask.reshape(h, w)
            masked = np.where(mask[..., None], baseline, base)
            prob = float(prob_fn(masked[None, ...])[0][target_idx])
            drop = orig_prob - prob
            curve.append({"mask_fraction": frac, "masked_probability": prob,
                          "probability_drop": drop})
            rows.append({
                "modality": modality, "method": method, "seed": seed,
                "sample_id": str(path.relative_to(ROOT)),
                "sample_index": index,
                "predicted_class": predicted_class,
                "predicted_class_index": target_idx,
                "original_probability": round(orig_prob, 6),
                "mask_fraction": frac,
                "mask_percent": int(round(frac * 100)),
                "masked_probability": round(prob, 6),
                "probability_drop": round(drop, 6),
            })
        auc = deletion_auc([c["mask_fraction"] for c in curve],
                           [c["masked_probability"] for c in curve])
        per_sample_json.append({
            "sample_id": str(path.relative_to(ROOT)),
            "predicted_class": predicted_class,
            "predicted_class_index": target_idx,
            "original_probability": round(orig_prob, 6),
            "deletion_auc": auc,
            "curve": curve,
        })
        print(f"  [{index}/{len(samples)}] {path.name} -> {predicted_class} "
              f"p={orig_prob:.3f} AUC={auc:.3f}")


    # ── aggregates ─────────────────────────────────────
    aggregates = []
    for frac in FRACTIONS:
        sel = [r for r in rows if r["mask_fraction"] == frac]
        probs = [r["masked_probability"] for r in sel]
        drops = [r["probability_drop"] for r in sel]
        agg = {"mask_fraction": frac, "mask_percent": int(round(frac * 100))}
        agg.update({f"probability_{k}": v for k, v in summarize(probs).items()})
        agg.update({f"drop_{k}": v for k, v in summarize(drops).items()})
        aggregates.append(agg)

    mean_curve = [a["probability_mean"] for a in aggregates]
    auc_mean = deletion_auc(FRACTIONS, mean_curve)
    auc_stats = summarize([s["deletion_auc"] for s in per_sample_json])

    # ── save per-method outputs ────────────────────────
    with open(out_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "per_sample.json", "w", encoding="utf-8") as f:
        json.dump({"modality": modality, "method": method, "seed": seed,
                   "samples": per_sample_json}, f, indent=2, default=str)

    with open(out_dir / "aggregate.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)

    aggregate_json = {
        "modality": modality, "method": method, "seed": seed,
        "sample_count": len(samples),
        "masking_fractions": FRACTIONS,
        "per_fraction": aggregates,
        "deletion_auc_mean_curve": auc_mean,
        "deletion_auc_per_sample": auc_stats,
        "interpretation": "Deletion AUC = trapezoidal area under the mean "
                          "target-probability curve over masked fraction "
                          "[0, 0.5]; lower AUC means removing highlighted "
                          "regions reduced the model probability more. "
                          "This measures model behaviour, not clinical causality.",
    }
    with open(out_dir / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_json, f, indent=2, default=str)

    sha_after = sha256_file(model_path)
    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "modality": modality,
        "method": method,
        "seed": seed,
        "max_samples": max_samples,
        "sample_count": len(samples),
        "samples": [str(p.relative_to(ROOT)) for p in samples],
        "model_file": str(model_path),
        "model_sha256_before": sha_before,
        "model_sha256_after": sha_after,
        "weights_unchanged": sha_before == sha_after,
        "class_names": list(cnn.class_names),
        "input_size": list(size),
        "masking": {
            "strategy": "replace top-attribution pixels with a deterministic "
                        "Gaussian-blur baseline of the same image",
            "fractions": [f"{int(f * 100)}%" for f in FRACTIONS],
            "importance_smoothing": "Gaussian blur, kernel ~5% of shorter side "
                                    "(identical for all methods)",
            "ranking": "magnitude of min-max normalized attribution map",
        },
        "xai_params_requested": xai_params,
        "xai_params_effective": effective_params,
        "xai_defaults": DEFAULT_PARAMS.get(method, {}),
        "environment": _env_info(),
        "note": "Test images are used for final reporting only; no XAI "
                "parameter was tuned on these results.",
    }
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    save_plot(out_dir, modality, method, aggregates, auc_mean, len(samples))

    if sha_before != sha_after:
        print("[fidelity] WARNING: model file hash changed during the run!")

    print(f"[fidelity] {modality}/{method}: n={len(samples)} "
          f"AUC(mean curve)={auc_mean:.4f} saved -> {out_dir}")
    return aggregate_json



# ─────────────────────────────────────────────
#  MODALITY SUMMARY
# ─────────────────────────────────────────────
def write_modality_summary(modality: str, methods: list) -> None:
    out_dir = OUT_ROOT / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method in methods:
        agg_path = out_dir / method / "aggregate.json"
        if not agg_path.exists():
            continue
        with open(agg_path, encoding="utf-8") as f:
            agg = json.load(f)
        row = {"modality": modality, "method": method,
               "samples": agg["sample_count"],
               "deletion_auc": agg["deletion_auc_mean_curve"]}
        for entry in agg["per_fraction"]:
            if entry["mask_fraction"] == 0.0:
                row["orig_prob_mean"] = entry["probability_mean"]
            else:
                row[f"prob_mean_{entry['mask_percent']}pct"] = entry["probability_mean"]
                row[f"drop_mean_{entry['mask_percent']}pct"] = entry["drop_mean"]
        summaries.append(row)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"modality": modality,
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "methods": summaries}, f, indent=2, default=str)
    if summaries:
        fields = []
        for row in summaries:                 # stable union of keys
            for key in row:
                if key not in fields:
                    fields.append(key)
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
    print(f"[fidelity] summary written -> {out_dir / 'summary.json'}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="MedVision AI — XAI fidelity (deletion) evaluation. "
                    "Read-only: no training, no weight/dataset changes.")
    ap.add_argument("--modality", required=True, choices=MODALITIES)
    ap.add_argument("--method", default="all", choices=[*METHODS, "all"])
    ap.add_argument("--max-samples", type=int, default=None,
                    help="test images per run (default: ALL valid test "
                         "images; use a small N e.g. 2 for smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    methods = list(METHODS) if args.method == "all" else [args.method]
    for method in methods:
        evaluate(args.modality, method, args.max_samples, args.seed)
    write_modality_summary(args.modality, methods)
    print("[fidelity] done. No training performed; model weights verified unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


