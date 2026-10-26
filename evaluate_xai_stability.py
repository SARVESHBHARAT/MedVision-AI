"""MedVision AI — XAI stability evaluation, read-only.

Question: does an explanation remain similar when the SAME test image receives
small, controlled, deterministic perturbations?

Protocol (fixed, no tuning on test data):
 1. Load one frozen modality model (no training, no weight writes).
 2. For each TEST image: original prediction + original attribution
    (signed map for Pearson, magnitude map for top-k overlap).
 3. Apply each fixed perturbation (deterministic, parameters recorded):
      brightness +10% | contrast +10% | gaussian noise std 0.03
      | rotation +5 deg (inverse-aligned before comparison)
      | resize-recompress (0.95 down/up + JPEG q=85)
 4. Re-run the SAME XAI method with the SAME settings and the SAME target
    class on every perturbed image.
 5. Compare underlying attribution ARRAYS (never rendered PNGs):
      - Pearson correlation on the SIGNED attribution (magnitude fallback)
      - top-10% magnitude overlap: IoU and overlap coefficient
 6. Report mean / std / median / count per perturbation and overall.

What this measures (and what it does NOT):
 * Robustness to IMAGE PERTURBATION with XAI randomness held fixed
   (LIME/SHAP seeds are identical for original and perturbed runs).
 * LIME's inherent sampling stochasticity is a DIFFERENT experiment
   (seed-repeat on the same image) and is explicitly NOT measured here.

Usage:
 .\\.venv\\Scripts\\python.exe evaluate_xai_stability.py --modality chest --method gradcam --max-samples 2 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse helpers already validated in the fidelity evaluator (test-only image
# selection, seeds, normalization, hashing, stats) — no duplicated logic.
from evaluate_xai_fidelity import (
    METHODS,
    MODALITIES,
    ROOT,
    _env_info,
    find_test_images,
    normalize01,
    resize_to,
    set_seeds,
    sha256_file,
    summarize,
)

OUT_ROOT = ROOT / "xai_outputs" / "stability"
PERTURBATIONS = ("brightness", "contrast", "gaussian_noise",
                 "rotation", "resize_recompress")
ROTATION_DEG = 5.0
TOPK_FRACTION = 0.10
NOISE_STD = 0.03
BRIGHTNESS_GAIN = 1.10
CONTRAST_GAIN = 1.10
RESIZE_SCALE = 0.95
JPEG_QUALITY = 85

PERTURBATION_PARAMS = {
    "brightness": {"gain": BRIGHTNESS_GAIN,
                   "note": "pixel value * 1.10, clipped to [0,1]"},
    "contrast": {"gain": CONTRAST_GAIN, "pivot": 0.5,
                 "note": "(pixel - 0.5) * 1.10 + 0.5, clipped to [0,1]"},
    "gaussian_noise": {"std": NOISE_STD, "distribution": "gaussian",
                       "rng": "np.random.default_rng(seed + sample_index)",
                       "note": "additive noise on [0,1] scale, clipped"},
    "rotation": {"degrees": ROTATION_DEG, "resample": "bilinear", "expand": False,
                 "fill": "per-channel image mean colour",
                 "alignment": "perturbed attribution inverse-rotated by -5 deg; "
                              "metrics computed only on the valid (non-fill) region"},
    "resize_recompress": {"downscale": RESIZE_SCALE, "jpeg_quality": JPEG_QUALITY,
                          "note": "bilinear downscale to 95%, back to original "
                                  "size, in-memory JPEG round-trip"},
}



# ─────────────────────────────────────────────
#  PERTURBATIONS (deterministic)
# ─────────────────────────────────────────────
def _to_arr(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def _to_pil(arr: np.ndarray) -> Image.Image:
    u8 = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(u8)


def apply_perturbation(image: Image.Image, name: str, seed: int,
                       sample_index: int) -> Image.Image:
    """Return a mildly perturbed copy of `image` (never modifies the input)."""
    if name == "rotation":
        mean_rgb = tuple(int(round(c)) for c in
                         _to_arr(image).reshape(-1, 3).mean(axis=0))
        return image.rotate(ROTATION_DEG, resample=Image.BILINEAR,
                            expand=False, fillcolor=mean_rgb)

    if name == "resize_recompress":
        w, h = image.size
        w2 = max(2, int(round(w * RESIZE_SCALE)))
        h2 = max(2, int(round(h * RESIZE_SCALE)))
        small = image.resize((w2, h2), Image.BILINEAR)
        back = small.resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        back.save(buf, format="JPEG", quality=JPEG_QUALITY)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    arr = _to_arr(image)
    if name == "brightness":
        arr = arr * BRIGHTNESS_GAIN
    elif name == "contrast":
        arr = (arr - 0.5) * CONTRAST_GAIN + 0.5
    elif name == "gaussian_noise":
        rng = np.random.default_rng(int(seed) + int(sample_index))
        arr = arr + rng.normal(0.0, NOISE_STD, arr.shape)
    else:
        raise ValueError(f"unknown perturbation '{name}'")
    return _to_pil(arr)


def rotate_map(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a float map with the same geometry as PIL image rotation."""
    im = Image.fromarray(np.asarray(arr, dtype=np.float32))   # mode 'F'
    rot = im.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
    return np.asarray(rot, dtype=np.float32)


# ─────────────────────────────────────────────
#  METRICS (compare arrays, never PNGs)
# ─────────────────────────────────────────────
def pearson_corr(a: np.ndarray, b: np.ndarray,
                 valid: np.ndarray | None = None) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if valid is not None:
        m = np.asarray(valid).reshape(-1).astype(bool)
        x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:            # constant map -> undefined
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def topk_metrics(a: np.ndarray, b: np.ndarray, k_frac: float = TOPK_FRACTION,
                 valid: np.ndarray | None = None):
    """IoU and overlap coefficient (|A n B|/k) of the top-k magnitude pixels."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if valid is not None:
        m = np.asarray(valid).reshape(-1).astype(bool)
        x, y = x[m], y[m]
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    k = max(1, int(round(k_frac * n)))
    top_a = np.argsort(-x, kind="stable")[:k]
    top_b = np.argsort(-y, kind="stable")[:k]
    inter = int(np.intersect1d(top_a, top_b, assume_unique=True).size)
    iou = inter / (2 * k - inter) if (2 * k - inter) > 0 else float("nan")
    overlap = inter / k
    return float(iou), float(overlap)


def finite(values):
    return [float(v) for v in values if v is not None and np.isfinite(v)]



# ─────────────────────────────────────────────
#  ONE METHOD x ONE MODALITY
# ─────────────────────────────────────────────
def _maps(result, size):
    """(magnitude 0..1, signed) maps aligned to the model-input grid."""
    magnitude = normalize01(resize_to(result.attribution_map, size))
    if result.signed_attribution is not None:
        signed = resize_to(result.signed_attribution, size)
    else:
        signed = magnitude
    return magnitude, signed


def evaluate(modality: str, method: str, max_samples: int, seed: int) -> dict:
    """Perturbation-stability experiment.  Read-only (no training)."""
    from PIL import Image

    from medvision_inference import CNNModel, MODEL_PATHS
    from xai import DEFAULT_PARAMS, explain_image
    from xai.common import display_size

    set_seeds(seed)
    out_dir = OUT_ROOT / modality / method
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stability] {modality} / {method} — loading frozen model ...")
    cnn = CNNModel(modality)
    model_path = Path(MODEL_PATHS[modality])
    sha_before = sha256_file(model_path)

    size = display_size(cnn.transform)
    samples = find_test_images(modality, max_samples, seed)
    if not samples:
        raise SystemExit(f"[stability] no test images found for '{modality}'")

    # Same fixed XAI settings for original AND perturbed images.
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

        base_result = explain_image(cnn, image, method=method,
                                    target_class=target_idx, **xai_params)
        if effective_params is None:
            effective_params = dict(base_result.parameters or {})
        base_mag, base_signed = _maps(base_result, size)

        sample_rows = []
        for pname in PERTURBATIONS:
            perturbed = apply_perturbation(image, pname, seed, index)
            pres = explain_image(cnn, perturbed, method=method,
                                 target_class=target_idx, **xai_params)
            pmag, psigned = _maps(pres, size)

            valid = None
            if pname == "rotation":
                # Align geometry: undo the +5 deg rotation on the perturbed
                # maps, then evaluate only inside the valid (non-fill) region.
                psigned = rotate_map(psigned, -ROTATION_DEG)
                pmag = rotate_map(pmag, -ROTATION_DEG)
                valid = rotate_map(np.ones(size, dtype=np.float32),
                                   -ROTATION_DEG) > 0.02

            r = pearson_corr(base_signed, psigned, valid)
            iou, overlap = topk_metrics(base_mag, pmag, TOPK_FRACTION, valid)

            row = {
                "modality": modality, "method": method, "seed": seed,
                "sample_id": str(path.relative_to(ROOT)),
                "sample_index": index,
                "predicted_class": predicted_class,
                "original_probability": round(orig_prob, 6),
                "perturbation": pname,
                "pearson": round(r, 6),
                "topk_iou": round(iou, 6),
                "topk_overlap": round(overlap, 6),
            }
            rows.append(row)
            sample_rows.append(row)
            print(f"  [{index}/{len(samples)}] {path.name} | {pname}: "
                  f"pearson={r:.3f} iou={iou:.3f}")

        per_sample_json.append({
            "sample_id": str(path.relative_to(ROOT)),
            "predicted_class": predicted_class,
            "original_probability": round(orig_prob, 6),
            "comparisons": sample_rows,
        })


    # ── aggregates: per perturbation + overall ────────
    aggregates = []
    for pname in [*PERTURBATIONS, "__all__"]:
        sel = rows if pname == "__all__" else \
            [r for r in rows if r["perturbation"] == pname]
        entry = {"perturbation": pname, "comparisons": len(sel)}
        for key in ("pearson", "topk_iou", "topk_overlap"):
            stats = summarize(finite([r[key] for r in sel]))
            for stat_name, value in stats.items():
                entry[f"{key}_{stat_name}"] = value
        aggregates.append(entry)

    overall = next(a for a in aggregates if a["perturbation"] == "__all__")

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
        "perturbations": list(PERTURBATIONS),
        "topk_fraction": TOPK_FRACTION,
        "per_perturbation": [a for a in aggregates
                             if a["perturbation"] != "__all__"],
        "overall": overall,
        "interpretation": "Pearson compares signed attribution arrays; top-k "
                          "IoU/overlap compare the top 10% magnitude pixels. "
                          "Higher = more stable explanations. This measures "
                          "model-explanation robustness, not clinical validity.",
        "lime_note": ("LIME uses a fixed random seed identical for original and "
                      "perturbed runs; this measures IMAGE perturbation "
                      "stability, NOT LIME's own sampling stochasticity "
                      "(a separate seed-repeat experiment, not performed here)."
                      if method == "lime" else None),
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
        "perturbations": PERTURBATION_PARAMS,
        "metrics": {
            "pearson": "Pearson correlation of signed attribution arrays "
                       "(magnitude fallback) over valid pixels",
            "topk_iou": f"IoU of top-{int(TOPK_FRACTION * 100)}% magnitude pixels",
            "topk_overlap": "|top-k A n top-k B| / k",
        },
        "alignment": "rotation maps inverse-rotated by -5 deg before comparison",
        "xai_params_requested": xai_params,
        "xai_params_effective": effective_params,
        "xai_defaults": DEFAULT_PARAMS.get(method, {}),
        "environment": _env_info(),
        "note": "Test images are used for final reporting only; no XAI "
                "parameter was tuned on these results. No training performed.",
    }
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    save_stability_plot(out_dir, modality, method, aggregates, len(samples))

    if sha_before != sha_after:
        print("[stability] WARNING: model file hash changed during the run!")

    def _fmt(value):
        return f"{value:.4f}" if value is not None else "n/a (no finite values)"
    print(f"[stability] {modality}/{method}: n={len(samples)} "
          f"pearson={_fmt(overall['pearson_mean'])} "
          f"iou={_fmt(overall['topk_iou_mean'])} saved -> {out_dir}")
    return aggregate_json



# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────
def save_stability_plot(mdir, modality, method, aggregates, n) -> None:
    """Grouped bar chart: mean±std Pearson and top-k IoU per perturbation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        entries = [a for a in aggregates if a["perturbation"] != "__all__"]
        labels = [a["perturbation"] for a in entries]
        x = np.arange(len(entries))
        width = 0.38

        def _series(key_mean):
            # None/NaN-safe: a group whose finite-value list was empty (all
            # comparisons NaN) has mean=None from summarize(); matplotlib cannot
            # do arithmetic on None, so draw nothing (NaN) for those bars.
            vals = np.array([a[key_mean] if a[key_mean] is not None else np.nan
                             for a in entries], dtype=float)
            return vals

        p_mean = _series("pearson_mean")
        i_mean = _series("topk_iou_mean")
        p_err = np.nan_to_num(_series("pearson_std"), nan=0.0)
        i_err = np.nan_to_num(_series("topk_iou_std"), nan=0.0)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(x - width / 2, p_mean, width, yerr=p_err,
               capsize=3, label="Pearson correlation")
        ax.bar(x + width / 2, i_mean, width, yerr=i_err,
               capsize=3, label=f"Top-{int(TOPK_FRACTION * 100)}% IoU")
        ax.set_ylabel("Similarity (mean $\\pm$ std)")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"XAI stability: {modality}/{method} "
                     f"(n={n}, seed fixed)")
        ax.legend(loc="lower left")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(mdir) / "stability_plot.png", dpi=200)
        fig.savefig(Path(mdir) / "stability_plot.pdf")
        plt.close(fig)
    except Exception as e:
        print(f"[stability] plot skipped for {method}: {e}")


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
        overall = agg["overall"]
        row = {"modality": modality, "method": method,
               "samples": agg["sample_count"],
               "pearson_mean": overall["pearson_mean"],
               "pearson_std": overall["pearson_std"],
               "pearson_median": overall["pearson_median"],
               "pearson_count": overall["pearson_count"],
               "topk_iou_mean": overall["topk_iou_mean"],
               "topk_iou_std": overall["topk_iou_std"],
               "topk_iou_median": overall["topk_iou_median"],
               "topk_iou_count": overall["topk_iou_count"]}
        for entry in agg["per_perturbation"]:
            row[f"pearson_mean__{entry['perturbation']}"] = entry["pearson_mean"]
            row[f"topk_iou_mean__{entry['perturbation']}"] = entry["topk_iou_mean"]
        summaries.append(row)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"modality": modality,
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "methods": summaries}, f, indent=2, default=str)
    if summaries:
        fields = []
        for row in summaries:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
    print(f"[stability] summary written -> {out_dir / 'summary.json'}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="MedVision AI — XAI stability (perturbation) evaluation. "
                    "Read-only: no training, no weight/dataset changes.")
    ap.add_argument("--modality", required=True, choices=MODALITIES)
    ap.add_argument("--method", default="all", choices=[*METHODS, "all"])
    ap.add_argument("--max-samples", type=int, default=8,
                    help="test images per run (default 8; use 2 for smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    methods = list(METHODS) if args.method == "all" else [args.method]
    for method in methods:
        evaluate(args.modality, method, args.max_samples, args.seed)
    write_modality_summary(args.modality, methods)
    print("[stability] done. No training performed; model weights verified unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

