"""
MedVision AI - Eye Diabetic Retinopathy Trainer [V5 - FIVE-CLASS STAGED TRANSFER LEARNING]
=========================================================================================

TASK (unchanged, always five classes - never binary):
    0 = No_DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferate_DR
Prediction = argmax over the 5 logits; confidence = softmax(logits). The product
contract stays exactly: image -> 5-class prediction -> confidence -> Grad-CAM ->
report -> existing UI.

WHAT V5 IS (a transfer-learning upgrade of the same application, not a redesign)
-------------------------------------------------------------------------------
Backbone  torchvision EfficientNet-B4 with the OFFICIAL ImageNet IMAGENET1K_V1
          pretrained weights (torchvision downloads/caches them once; the
          backbone is NEVER initialised randomly and NEVER started from a
          V1..V4 checkpoint). backbone.classifier -> Identity.
Heads     primary 5-class head   : Dropout(0.35) -> Linear(1792, 5)
          auxiliary ordinal head : Linear(1792, 4)   (CORAL-style cumulative
                                   thresholds y>0, y>1, y>2, y>3)
          forward(x) ALWAYS returns the plain (B,5) logits, so Grad-CAM, the
          evaluator and the application never see the auxiliary head.
Input     the existing deterministic fundus448 RGB cache (validated by
          cnn/eye_make_fundus_v5.py): bbox trim (gray>10) -> 2% pad -> square
          crop of the retinal field -> bilinear 448x448 -> circular
          retinal-field mask -> ImageNet normalization. Training and inference
          share ONE preprocessing implementation
          (cnn/eye_make_fundus_v5.process_fundus), so there is no train/infer skew.

STAGED TRANSFER LEARNING (3 phases, differential learning rates)
----------------------------------------------------------------
P0 head warm-up    backbone fully frozen (BN kept in eval mode); only the two
                   heads train (lr 3e-4), up to 4 epochs - the randomly
                   initialised heads adapt to the retinal features first.
P1 upper unfreeze  the last 3 of the 9 MBConv stages (features[6:]) become
                   trainable together with the heads: backbone lr 1.5e-5,
                   head lr 8e-5, up to 8 epochs.
P2 full fine-tune  every backbone stage + both heads: differential LRs
                   features[0:4] 1e-5 | features[4:9] 2e-5 | heads 8e-5,
                   up to 24 epochs, cosine decay, early stopping patience 8.
The pretrained groups always train ~10x slower than the new heads and the stem /
early stages are always the slowest, so ImageNet features are adapted gradually
rather than destroyed. AdamW with no weight decay on 1-D (BN / bias) parameters.

LOSS (one classification mechanism + one auxiliary ordinal objective)
---------------------------------------------------------------------
primary   class-weighted soft-ordinal CE - soft target 0.80 on the true grade,
          0.10 per adjacent grade (0.20 on the single adjacent grade at the
          boundaries), zero mass on distant grades; per-sample weight = sqrt
          inverse-frequency class weight (alpha=0.5, mean-normalised, TRAIN only).
auxiliary 0.25 * BCE-with-logits on the 4 cumulative ordinal thresholds.
The 5-class softmax stays the prediction; the ordinal term only shapes the shared
representation towards the severity ordering (fewer distant confusions).

AUGMENTATION (mild, fundus-appropriate - nothing that erases micro-lesions)
---------------------------------------------------------------------------
hflip 0.5, RandomAffine(rot +-10deg, translate 6%, scale 0.90-1.10),
ColorJitter(0.2, 0.2, 0.2), GaussianBlur p=0.15, ImageNet normalization.
Val/test: deterministic ToTensor + normalize. MixUp/CutMix are deliberately NOT
used - they were never shown to help on validation here and blur micro-lesions.

CHECKPOINT SELECTION (VALIDATION ONLY - the test set is untouched until the end)
-------------------------------------------------------------------------------
score = 0.25*val_acc + 0.40*val_macro_recall + 0.25*val_macro_F1
      + 0.05*val_severe_recall + 0.05*val_pdr_recall          (recall-first)
Candidates with val_acc < ACC_FLOOR (0.60) are REJECTED, so a model can never win
by trading accuracy away for minority recall. Both the RAW and the EMA weights are
validated every epoch and the better one (on validation) becomes the candidate.
TTA (orig / hflip / +-4deg) is chosen on VALIDATION only; whichever mode wins is
applied once to the untouched test set at the very end.

ARTEFACTS (new files only; V1..V4 and the binary model are never modified)
--------------------------------------------------------------------------
cnn/eye_efficientnet_b4_v5.pth              final five-class checkpoint
cnn/eye_meta_v5.json                        checkpoint metadata
cnn/eye_v5_results.json                     validation + single test evaluation
cnn/eye_efficientnet_b4_v5_p0/p1/p2.pth     per-phase safety checkpoints
cnn/eye_efficientnet_b4_v5_last.pth         resumable training state (--resume)

Usage
-----
  python cnn/eye_trainer_v5.py                      # full training (GPU, AMP)
  python cnn/eye_trainer_v5.py --smoke              # 1 epoch/phase on a small
                                                    # subset, validation only,
                                                    # never writes V5 artefacts
  python cnn/eye_trainer_v5.py --resume             # continue an interrupted run
  python cnn/eye_trainer_v5.py --no-ordinal-head    # primary head only (ablation)
"""

import argparse
import copy
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.transforms import functional as TF

# ------------------------------------------------------------------
#  Version / paths (V1..V4 + binary artefacts are never touched)
# ------------------------------------------------------------------
VERSION = "V5"
ARCH_NAME = "efficientnet_b4_v5"
BACKBONE_NAME = "efficientnet_b4 (torchvision, official ImageNet IMAGENET1K_V1 pretrained)"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "cnn" / "eye"
MODEL_DIR = BASE_DIR / "cnn"
MODEL_PATH = MODEL_DIR / "eye_efficientnet_b4_v5.pth"
META_PATH = MODEL_DIR / "eye_meta_v5.json"
RESULTS_PATH = MODEL_DIR / "eye_v5_results.json"
LAST_STATE_PATH = MODEL_DIR / "eye_efficientnet_b4_v5_last.pth"
SMOKE_PREFIX = "eye_v5_smoke"

CACHE_ROOT = DATA_DIR / "fundus448"          # existing V4/V5 rgb cache - reused as-is
PREPROCESS_DESC = ("bbox trim (gray>10) -> 2% pad -> square center-crop of retinal "
                   "field -> bilinear 448x448 -> circular retinal-field mask (no CLAHE)")

IMG_SIZE, NUM_CLASSES = 448, 5
NUM_ORDINAL_THRESHOLDS = NUM_CLASSES - 1     # CORAL cumulative thresholds (y>0..y>3)
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet
SPLITS = {"train": ("train_1.csv", 2930), "val": ("valid.csv", 366), "test": ("test.csv", 366)}

# ---- staged transfer-learning configuration ------------------------
P0_EPOCHS, P0_PATIENCE, P0_HEAD_LR = 4, 4, 3e-4      # head warm-up (frozen backbone)
P1_EPOCHS, P1_PATIENCE = 8, 5                        # upper backbone unfreeze
P1_FIRST_STAGE = 6                                   # train features[6:] (last 2 MBConv
                                                     # stages + the 1x1 head conv)
P1_BB_LR, P1_HEAD_LR = 1.5e-5, 8e-5
P2_EPOCHS, P2_PATIENCE = 24, 8                       # full fine-tune
P2_LATE_FIRST_STAGE = 4                              # features[:4] slow, features[4:] faster
P2_EARLY_BB_LR, P2_LATE_BB_LR, P2_HEAD_LR = 1e-5, 2e-5, 8e-5
WEIGHT_DECAY = 1e-4                                  # applied to >=2-D params only
GRAD_CLIP = 5.0
EMA_DECAY = 0.998
HEAD_DROPOUT = 0.35
SEED = 42

ORD_TARGET_MAIN = 0.80      # soft ordinal target on the true grade
ORD_TARGET_ADJ = 0.10       # per adjacent grade (boundaries: 0.20 on one side)
ORD_AUX_WEIGHT = 0.25       # auxiliary CORAL head weight (secondary objective)
W_ALPHA = 0.5               # sqrt inverse-frequency class weights (train split only)
ACC_FLOOR = 0.60            # reject candidates whose validation accuracy collapsed

DEFAULT_BATCH = 12          # preflight-measured VRAM fit for B4 @ 448 on 6GB
DEFAULT_WORKERS = 2         # pagefile-safe on this Windows laptop

TTA_ANGLE = 4.0
TTA_VIEWS = ["orig", "hflip", "rot+4deg", "rot-4deg"]
GRADCAM_LAYER = "backbone.features[-1]  # (B,1792,14,14) @ 448px"
SCORE_WEIGHTS = {"accuracy": 0.25, "macro_recall": 0.40, "macro_f1": 0.25,
                 "severe_recall": 0.05, "pdr_recall": 0.05}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
CHANNELS_LAST = USE_AMP     # cudnn-friendly memory format on the RTX 4050

_HEAD_PREFIXES = ("head.", "ordinal_head.")
# torchvision EfficientNet-B4 layout: features[0] = stem, features[1:8] = 7 MBConv
# stages, features[8] = final 1x1 conv head (=> 1792 channels at features[-1]).
B4_FEATURE_BLOCKS = 9
_EARLY_BB_PREFIXES = tuple(f"backbone.features.{i}."
                           for i in range(P2_LATE_FIRST_STAGE))
_LATE_BB_PREFIXES = tuple(f"backbone.features.{i}."
                          for i in range(P2_LATE_FIRST_STAGE, B4_FEATURE_BLOCKS))
_UPPER_BB_PREFIXES = tuple(f"backbone.features.{i}."
                           for i in range(P1_FIRST_STAGE, B4_FEATURE_BLOCKS))

PHASES = [
    {"tag": "P0", "name": "head warm-up (backbone frozen)", "epochs": P0_EPOCHS,
     "patience": P0_PATIENCE, "freeze_bn": True,
     "groups": [("heads", P0_HEAD_LR, _HEAD_PREFIXES)]},
    {"tag": "P1", "name": f"upper backbone unfreeze (features[{P1_FIRST_STAGE}:9])",
     "epochs": P1_EPOCHS, "patience": P1_PATIENCE, "freeze_bn": False,
     "groups": [("backbone_upper", P1_BB_LR, _UPPER_BB_PREFIXES),
                ("heads", P1_HEAD_LR, _HEAD_PREFIXES)]},
    {"tag": "P2", "name": "full fine-tune (differential LRs)", "epochs": P2_EPOCHS,
     "patience": P2_PATIENCE, "freeze_bn": False,
     "groups": [("backbone_early", P2_EARLY_BB_LR, _EARLY_BB_PREFIXES),
                ("backbone_late", P2_LATE_BB_LR, _LATE_BB_PREFIXES),
                ("heads", P2_HEAD_LR, _HEAD_PREFIXES)]},
]


# ------------------------------------------------------------------
#  Data integrity (hard asserts; identical rules to every other version)
# ------------------------------------------------------------------
def load_split_df(split: str) -> pd.DataFrame:
    csv_path = DATA_DIR / SPLITS[split][0]
    expected = SPLITS[split][1]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"id_code": str})
    df["id_code"] = df["id_code"].str.strip()
    df["diagnosis"] = df["diagnosis"].astype(int)
    if len(df) != expected:
        raise ValueError(f"[{split}] expected {expected} rows, found {len(df)}")
    if not df["diagnosis"].isin(range(NUM_CLASSES)).all():
        raise ValueError(f"[{split}] labels outside 0..4")
    if df["id_code"].duplicated().any():
        raise ValueError(f"[{split}] duplicate id_code entries")
    if not (CACHE_ROOT / split).is_dir():
        raise FileNotFoundError(f"Missing cache dir: {CACHE_ROOT / split}")
    on_disk = {p.stem for p in (CACHE_ROOT / split).glob("*.png")}
    if on_disk != set(df["id_code"]):
        raise ValueError(f"[{split}] cache membership != CSV membership "
                         f"(missing={len(set(df['id_code']) - on_disk)}, "
                         f"extra={len(on_disk - set(df['id_code']))})")
    return df


def verify_all(verbose: bool = True) -> dict:
    dfs, ids = {}, {}
    for split in ("train", "val", "test"):
        dfs[split] = load_split_df(split)
        ids[split] = set(dfs[split]["id_code"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = ids[a] & ids[b]
        if ov:
            raise ValueError(f"OVERLAP between {a} and {b}: {len(ov)} ids")
    counts = {s: dfs[s]["diagnosis"].value_counts().sort_index().to_dict() for s in dfs}
    if verbose:
        print("Integrity OK: train=2930 val=366 test=366, zero overlap, cache==CSV")
        for s in ("train", "val", "test"):
            print(f"  {s:<5} " + " ".join(f"{CLASS_NAMES[k]}={counts[s].get(k, 0)}"
                                          for k in range(NUM_CLASSES)))
    return dfs


# ------------------------------------------------------------------
#  Dataset (cached 448px PNGs - the exact V5 preprocessing output)
# ------------------------------------------------------------------
TRAIN_TF = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomAffine(10, translate=(0.06, 0.06), scale=(0.90, 1.10),
                            interpolation=TF.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class FundusCachedDataset(Dataset):
    def __init__(self, split: str, ids, labels, train: bool):
        self.root = CACHE_ROOT / split
        self.ids, self.labels = [str(i) for i in ids], [int(v) for v in labels]
        self.tf = TRAIN_TF if train else EVAL_TF

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = Image.open(self.root / f"{self.ids[idx]}.png").convert("RGB")
        return self.tf(img), self.labels[idx]


def make_loader(ds, batch_size, workers, train):
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, drop_last=train)


def build_loaders(dfs, batch_size, workers, limit_per_class=None):
    """Train/val loaders shared by all phases. limit_per_class builds a small,
    class-balanced subset for --smoke runs (all five classes stay represented)."""
    def _take(df, n):
        if n is None:
            return df
        parts = [g.sample(n=min(len(g), n), random_state=SEED)
                 for _, g in df.groupby("diagnosis")]
        return pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    tr = _take(dfs["train"], limit_per_class)
    va = _take(dfs["val"], max(4, (limit_per_class or 0) // 2) if limit_per_class else None)
    train_loader = make_loader(FundusCachedDataset("train", tr["id_code"], tr["diagnosis"], True),
                               batch_size, workers, True)
    val_loader = make_loader(FundusCachedDataset("val", va["id_code"], va["diagnosis"], False),
                             batch_size, workers, False)
    return train_loader, val_loader


# ------------------------------------------------------------------
#  Model - ImageNet-pretrained EfficientNet-B4 + 5-class head (+ auxiliary
#  ordinal head). forward(x) returns PLAIN (B,5) logits for Grad-CAM / app use.
# ------------------------------------------------------------------
class EyeEffNetV5(nn.Module):
    """EfficientNet-B4 backbone + primary 5-class head + auxiliary ordinal head.

    forward(x)          -> (B, NUM_CLASSES) logits ONLY (Grad-CAM / app contract)
    forward_with_aux(x) -> ((B,5) logits, (B,4) ordinal logits) for training
    Grad-CAM target layer: backbone.features[-1] -> (B, 1792, 14, 14) @ 448px.
    """

    def __init__(self, dropout: float = HEAD_DROPOUT, with_ordinal: bool = True):
        super().__init__()
        bb = efficientnet_b4(weights=None)
        if len(bb.features) != B4_FEATURE_BLOCKS:
            raise RuntimeError(f"unexpected torchvision EfficientNet-B4 layout: "
                               f"{len(bb.features)} feature blocks "
                               f"(the phase freeze ranges assume {B4_FEATURE_BLOCKS})")
        in_feat = bb.classifier[1].in_features              # 1792
        bb.classifier = nn.Identity()
        self.backbone = bb
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
        self.with_ordinal = bool(with_ordinal)
        self.ordinal_head = (nn.Linear(in_feat, NUM_ORDINAL_THRESHOLDS)
                             if with_ordinal else None)

    def forward(self, x):
        return self.head(self.backbone(x))

    def forward_with_aux(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        if self.ordinal_head is None:
            return logits, None
        return logits, self.ordinal_head(feat)


def build_model(pretrained: bool = True, with_ordinal: bool = True,
                verbose: bool = True) -> EyeEffNetV5:
    """The backbone ALWAYS starts from the official ImageNet IMAGENET1K_V1
    weights - never from a V1..V4 checkpoint, never randomly."""
    model = EyeEffNetV5(with_ordinal=with_ordinal)
    if pretrained:
        sd = EfficientNet_B4_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
        # the backbone classifier is replaced by Identity + our own heads, so the
        # ImageNet classifier keys are legitimately "unexpected"
        core_missing = [k for k in missing
                        if k.startswith(("features.", "conv_stem", "bn1"))]
        core_unexpected = [k for k in unexpected if not k.startswith("classifier.")]
        if core_missing or core_unexpected:
            raise RuntimeError(f"ImageNet weights did not cover the backbone cleanly: "
                               f"missing={core_missing[:5]} "
                               f"unexpected={core_unexpected[:5]}")
        if float(model.backbone.features[0][1].running_mean.abs().max()) == 0.0:
            raise RuntimeError("Backbone looks randomly initialised "
                               "(stem BN running_mean is all zero) - pretrained load failed")
        if verbose:
            print(f"ImageNet IMAGENET1K_V1 weights loaded into backbone "
                  f"({len(sd)} tensors; classifier re-headed to {NUM_CLASSES} classes)")
    return model


def pretrained_weights_info() -> dict:
    """Location + sha256 of the cached official ImageNet weights (provenance)."""
    roots = {Path(torch.hub.get_dir()) / "checkpoints",
             Path.home() / ".cache" / "torch" / "hub" / "checkpoints"}
    hits = sorted({h for r in roots if r.is_dir() for h in r.glob("efficientnet_b4*")})
    if not hits:
        return {"file": None, "sha256": None,
                "searched": [str(r) for r in roots],
                "note": "weights are resolved (and cached) by torchvision at load time"}
    path = hits[0]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"file": str(path), "sha256": h.hexdigest(),
            "source": "torchvision EfficientNet_B4_Weights.IMAGENET1K_V1"}


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """The ONLY Grad-CAM feature layer for V5: final MBConv stage of the B4."""
    return model.backbone.features[-1]


def to_channels_last(x):
    return x.contiguous(memory_format=torch.channels_last) if CHANNELS_LAST else x


# ------------------------------------------------------------------
#  Metrics (identical formulas to previous versions; no sklearn needed)
# ------------------------------------------------------------------
def confusion_matrix_np(y_true, y_pred, k=NUM_CLASSES):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm):
    eps = 1e-12
    prec = cm.diagonal() / np.maximum(cm.sum(axis=0), eps)
    rec = cm.diagonal() / np.maximum(cm.sum(axis=1), eps)
    f1 = 2 * prec * rec / np.maximum(prec + rec, eps)
    return prec, rec, f1


def cohen_kappa(cm):
    n = float(cm.sum())
    po = float(np.trace(cm)) / n
    pe = float((cm.sum(axis=1) * cm.sum(axis=0)).sum()) / (n * n)
    return (po - pe) / max(1.0 - pe, 1e-12)


def quadratic_weighted_kappa(cm):
    k = cm.shape[0]
    obs = cm.astype(np.float64)
    n = obs.sum()
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    idx = np.arange(k, dtype=np.float64)
    w_mat = (idx.reshape(-1, 1) - idx.reshape(1, -1)) ** 2 / (k - 1) ** 2
    return 1.0 - float((w_mat * obs).sum()) / max(float((w_mat * expected).sum()), 1e-12)


def checkpoint_score(acc, macro_recall, macro_f1, severe_recall, pdr_recall):
    """Recall-first validation score (weights in SCORE_WEIGHTS)."""
    return (SCORE_WEIGHTS["accuracy"] * acc
            + SCORE_WEIGHTS["macro_recall"] * macro_recall
            + SCORE_WEIGHTS["macro_f1"] * macro_f1
            + SCORE_WEIGHTS["severe_recall"] * severe_recall
            + SCORE_WEIGHTS["pdr_recall"] * pdr_recall)


def summarize(y_pred, y_true, loss=None):
    cm = confusion_matrix_np(y_true, y_pred)
    acc = float((y_true == y_pred).mean())
    prec, rec, f1 = per_class_metrics(cm)
    metrics = {
        "accuracy": acc,
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "cohen_kappa": cohen_kappa(cm),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(cm),
        "severe_recall": float(rec[3]),
        "pdr_recall": float(rec[4]),
        "per_class": {CLASS_NAMES[i]: {"precision": float(prec[i]),
                                       "recall": float(rec[i]),
                                       "f1": float(f1[i]),
                                       "support": int(cm[i].sum())}
                      for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
    }
    if loss is not None:
        metrics["loss"] = float(loss / max(len(y_true), 1))
    metrics["score"] = checkpoint_score(acc, metrics["macro_recall"], metrics["macro_f1"],
                                        metrics["severe_recall"], metrics["pdr_recall"])
    return metrics


# ------------------------------------------------------------------
#  Seed / class weights / losses / EMA
# ------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_df) -> torch.Tensor:
    """Single, conservative balancing mechanism: sqrt inverse-frequency weights
    (alpha=0.5), mean-normalised, computed from the TRAIN split only."""
    counts = train_df["diagnosis"].value_counts().sort_index()
    assert list(counts.index) == list(range(NUM_CLASSES)), "train labels 0..4 expected"
    n = counts.to_numpy().astype(np.float64)
    w = (n.sum() / (NUM_CLASSES * n)) ** W_ALPHA
    w = w / w.mean()
    print("Class weights (train-only, sqrt inverse-frequency, alpha=0.5): "
          + " ".join(f"{CLASS_NAMES[i]}={w[i]:.4f}" for i in range(NUM_CLASSES)))
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def ordinal_soft_targets(y: torch.Tensor) -> torch.Tensor:
    """(N,5) soft targets: 0.80 on the true grade, 0.10 on each valid adjacent
    grade (0.20 on the single adjacent grade at the boundaries). No mass on
    non-adjacent grades. argmax(logits) is still the prediction."""
    y = y.long()
    n = y.size(0)
    t = torch.zeros(n, NUM_CLASSES, dtype=torch.float32, device=y.device)
    t.scatter_(1, y.unsqueeze(1), ORD_TARGET_MAIN)
    lo, hi = y - 1, y + 1
    m_lo = (lo >= 0).float()
    m_hi = (hi <= NUM_CLASSES - 1).float()
    interior = (m_lo * m_hi) > 0
    adj = torch.where(interior,
                      torch.full((n,), ORD_TARGET_ADJ, dtype=torch.float32, device=y.device),
                      torch.full((n,), 2.0 * ORD_TARGET_ADJ, dtype=torch.float32,
                                 device=y.device))
    t.scatter_add_(1, lo.clamp(0, NUM_CLASSES - 1).unsqueeze(1), (adj * m_lo).unsqueeze(1))
    t.scatter_add_(1, hi.clamp(0, NUM_CLASSES - 1).unsqueeze(1), (adj * m_hi).unsqueeze(1))
    return t


def soft_weighted_ce(logits: torch.Tensor, y: torch.Tensor,
                     class_w: torch.Tensor) -> torch.Tensor:
    """Class-weighted soft-label cross-entropy (primary five-class objective):
    per-sample weight = class_w[true class]; target = ordinal soft targets."""
    logp = F.log_softmax(logits.float(), dim=1)
    t = ordinal_soft_targets(y)
    w = class_w[y.long()]
    return -(w * (t * logp).sum(dim=1)).mean()


def ordinal_threshold_targets(y: torch.Tensor) -> torch.Tensor:
    """CORAL-style cumulative targets [y>0, y>1, y>2, y>3] -> (N,4)."""
    y = y.long()
    thr = torch.arange(NUM_ORDINAL_THRESHOLDS, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > thr).float()


class V5Loss(nn.Module):
    """total = class-weighted soft-ordinal CE + ORD_AUX_WEIGHT * ordinal BCE.

    The auxiliary CORAL head is a secondary objective only: the official
    prediction is always argmax of the primary five-class logits.
    """

    def __init__(self, class_weights, aux_weight: float = ORD_AUX_WEIGHT):
        super().__init__()
        self.register_buffer("class_w", torch.as_tensor(class_weights, dtype=torch.float32))
        self.aux_weight = float(aux_weight)

    def forward(self, outputs, y):
        logits, ord_logits = outputs if isinstance(outputs, (tuple, list)) else (outputs, None)
        loss = soft_weighted_ce(logits, y, self.class_w)
        if ord_logits is not None and self.aux_weight > 0.0:
            loss = loss + self.aux_weight * F.binary_cross_entropy_with_logits(
                ord_logits.float(), ordinal_threshold_targets(y))
        return loss


class ModelEMA:
    """Exponential moving average of the weights. Whether EMA or RAW weights are
    used for the final checkpoint is decided ONLY on validation."""

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() if v.detach().is_floating_point()
                       else v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.detach().is_floating_point():
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                s.copy_(v.detach())

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


# ------------------------------------------------------------------
#  Phase trainability + optimizer (differential LRs, no decay on 1-D params)
# ------------------------------------------------------------------
def phase_param_groups(model: nn.Module, phase: dict):
    """Match every parameter to exactly ONE phase group; everything else frozen."""
    named = list(model.named_parameters())
    assign, groups = {}, []
    for label, lr, prefixes in phase["groups"]:
        hit = [(n, p) for n, p in named if n.startswith(tuple(prefixes))]
        if not hit:
            raise RuntimeError(f"[{phase['tag']}] group '{label}' matched no parameters "
                               f"(prefixes={prefixes})")
        for n, _ in hit:
            if n in assign:
                raise RuntimeError(f"[{phase['tag']}] parameter {n} matched two groups")
            assign[n] = label
        groups.append((label, lr, hit))
    for n, p in named:
        p.requires_grad_(n in assign)
    n_train = sum(1 for _, p in named if p.requires_grad)
    n_total = len(named)
    print(f"  [{phase['tag']}] trainable tensors: {n_train}/{n_total} | "
          + " | ".join(f"{lab} lr={lr:g} ({len(hit)} tensors)" for lab, lr, hit in groups),
          flush=True)
    return groups


def build_optimizer(model: nn.Module, phase: dict, weight_decay: float = WEIGHT_DECAY):
    specs = []
    for label, lr, hit in phase_param_groups(model, phase):
        decay = [p for _, p in hit if p.ndim > 1]
        no_decay = [p for _, p in hit if p.ndim <= 1]
        if decay:
            specs.append({"params": decay, "lr": lr, "weight_decay": weight_decay,
                          "group_name": f"{label}.decay"})
        if no_decay:
            specs.append({"params": no_decay, "lr": lr, "weight_decay": 0.0,
                          "group_name": f"{label}.nodecay"})
    return optim.AdamW(specs)


# ------------------------------------------------------------------
#  Train / evaluate
# ------------------------------------------------------------------
def train_epoch(model, ema, loader, optimizer, scaler, criterion, freeze_bn: bool) -> float:
    model.train()
    if freeze_bn:
        model.backbone.eval()            # warm-up: backbone statistics stay frozen
    loss_sum, seen = 0.0, 0
    for x, y in loader:
        x = to_channels_last(x.to(DEVICE, non_blocking=True))
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            outputs = model.forward_with_aux(x)
            loss = criterion(outputs, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        groups = [p for g in optimizer.param_groups for p in g["params"]]
        nn.utils.clip_grad_norm_(groups, GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        bs = y.size(0)
        loss_sum += loss.item() * bs
        seen += bs
    return loss_sum / max(seen, 1)


def tta_views_of(x):
    """The 4 TTA views (probabilities averaged): original, hflip, +-4 deg."""
    yield x
    yield torch.flip(x, dims=[3])
    yield TF.rotate(x, TTA_ANGLE)
    yield TF.rotate(x, -TTA_ANGLE)


@torch.no_grad()
def evaluate(model, loader, mode="std"):
    """mode='std': single-view. 'tta': probability-averaged 4-view TTA.
    'both': both from ONE pass (loss reported from the original view)."""
    model.eval()
    ys, preds_std, preds_tta = [], [], []
    loss_sum = 0.0
    for x, y in loader:
        x = to_channels_last(x.to(DEVICE, non_blocking=True))
        y_dev = y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            if mode == "std":
                logits = model(x)
                loss_sum += F.cross_entropy(logits.float(), y_dev, reduction="sum").item()
                preds_std.append(logits.argmax(1).cpu().numpy())
            else:
                probs, std_logits = None, None
                for i, view in enumerate(tta_views_of(x)):
                    logits = model(view)
                    if i == 0:
                        std_logits = logits
                    p = F.softmax(logits.float(), dim=1)
                    probs = p if probs is None else probs + p
                probs = probs / len(TTA_VIEWS)
                loss_sum += F.cross_entropy(std_logits.float(), y_dev, reduction="sum").item()
                if mode == "both":
                    preds_std.append(std_logits.argmax(1).cpu().numpy())
                preds_tta.append(probs.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    if mode == "both":
        return (summarize(np.concatenate(preds_std), y_true, loss_sum),
                summarize(np.concatenate(preds_tta), y_true, None), y_true)
    y_pred = np.concatenate(preds_std if mode == "std" else preds_tta)
    return summarize(y_pred, y_true, loss_sum)


# ------------------------------------------------------------------
#  Checkpoint persistence (atomic; interruption-safe) + phase runner
# ------------------------------------------------------------------
def atomic_save(path: Path, payload: dict, verbose: bool = True) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    if verbose:
        print(f"  checkpoint saved -> {path.name}", flush=True)


def save_last_state(path: Path, model, ema, optimizer, scheduler, scaler,
                    phase_tag, epoch, bad, best, history) -> None:
    """Resumable training state (written after EVERY epoch so an interrupted
    multi-hour run never loses more than one epoch)."""
    atomic_save(path, {
        "phase_tag": phase_tag, "epoch": epoch, "bad": bad, "best": best,
        "model_state_dict": model.state_dict(),
        "ema_shadow": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
    }, verbose=False)


def ckpt_meta(weight_kind, phase_tag, epoch, val_summary, with_ordinal, extra=None):
    meta = {
        "version": VERSION, "architecture": ARCH_NAME, "backbone": BACKBONE_NAME,
        "task": "five_class_dr_severity",
        "task_statement": ("TASK: Five-class Diabetic Retinopathy Severity | "
                           "0 = No_DR, 1 = Mild, 2 = Moderate, 3 = Severe, "
                           "4 = Proliferate_DR"),
        "class_names": CLASS_NAMES, "num_classes": NUM_CLASSES,
        "image_size": [IMG_SIZE, IMG_SIZE],
        "normalization": {"mean": MEAN, "std": STD},
        "preprocessing": PREPROCESS_DESC,
        "gradcam_layer": GRADCAM_LAYER,
        "forward_returns": "logits (B,5); probabilities = softmax(logits)",
        "auxiliary_ordinal_head": bool(with_ordinal),
        "auxiliary_ordinal_note": ("Linear(1792->4) CORAL head is used for training "
                                   f"only (loss weight {ORD_AUX_WEIGHT}); forward() "
                                   "returns the primary 5-class logits"),
        "pretrained_weights": pretrained_weights_info(),
        "weight_kind": weight_kind, "phase": phase_tag, "epoch": epoch,
        "validation": val_summary,
    }
    if extra:
        meta.update(extra)
    return meta


def run_phase(phase, model, ema, ema_model, loaders, criterion, history,
              last_state_path=None, resume=None, epochs_override=None,
              save_last=True, verbose=True, acc_floor: float = ACC_FLOOR):
    """One transfer-learning phase. Every epoch validates RAW and EMA weights;
    the better ELIGIBLE one (validation checkpoint score, val_acc >= acc_floor)
    becomes the candidate. Early stopping: phase patience. The best candidate is
    restored into `model` when the phase ends. If NO epoch is eligible the phase
    falls back to its best-scoring weights and flags the result as ineligible."""
    tag = phase["tag"]
    epochs = int(epochs_override or phase["epochs"])
    train_loader, val_loader = loaders

    optimizer = build_optimizer(model, phase)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    start_epoch, bad, best, fallback = 1, 0, None, None
    if resume is not None and resume.get("phase_tag") == tag:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        try:
            scaler.load_state_dict(resume["scaler"])
        except Exception:
            pass
        start_epoch = int(resume["epoch"]) + 1
        bad = int(resume.get("bad", 0))
        best = resume.get("best")
        print(f"  [{tag}] RESUME: continuing from epoch {start_epoch}"
              + (f" (phase best score {best['val']['score']:.4f})" if best else ""))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, ema, train_loader, optimizer, scaler,
                              criterion, freeze_bn=phase["freeze_bn"])
        scheduler.step()                      # AFTER optimizer.step() (correct order)
        val_raw = evaluate(model, val_loader, "std")
        ema_model.load_state_dict(ema.state_dict())
        val_ema = evaluate(ema_model, val_loader, "std")

        lrs = sorted({round(float(g["lr"]), 8) for g in optimizer.param_groups})
        cands = [("raw", val_raw, copy.deepcopy(model.state_dict())),
                 ("ema", val_ema, ema.state_dict())]
        eligible = [c for c in cands if c[1]["accuracy"] >= acc_floor]
        if not eligible:
            kind_fb, val_fb, state_fb = max(cands, key=lambda c: c[1]["score"])
            fallback = {"state": state_fb, "kind": kind_fb, "val": val_fb,
                        "epoch": epoch, "eligible": False}
            print(f"  [{tag} {epoch:02d}/{epochs}] REJECTED: both raw and EMA "
                  f"val_acc < acc_floor={acc_floor:.2f} "
                  f"(raw={val_raw['accuracy']:.4f}, ema={val_ema['accuracy']:.4f}) "
                  f"-> kept as ineligible fallback only", flush=True)
            bad += 1
            history.append({"phase": tag, "epoch": epoch, "train_loss": round(tr_loss, 4),
                            "val_raw_acc": round(val_raw["accuracy"], 4),
                            "val_ema_acc": round(val_ema["accuracy"], 4),
                            "candidate": "rejected", "lr": lrs})
            if save_last and last_state_path is not None:
                save_last_state(last_state_path, model, ema, optimizer, scheduler, scaler,
                                tag, epoch, bad, best or fallback, history)
            if bad >= phase["patience"]:
                print(f"  [{tag}] early stopping (accuracy collapse)", flush=True)
                break
            continue

        kind, val, state = max(eligible, key=lambda c: c[1]["score"])
        history.append({"phase": tag, "epoch": epoch, "train_loss": round(tr_loss, 4),
                        "val_raw_score": round(val_raw["score"], 4),
                        "val_raw_acc": round(val_raw["accuracy"], 4),
                        "val_raw_macro_recall": round(val_raw["macro_recall"], 4),
                        "val_ema_score": round(val_ema["score"], 4),
                        "val_ema_acc": round(val_ema["accuracy"], 4),
                        "val_ema_macro_recall": round(val_ema["macro_recall"], 4),
                        "candidate": kind, "candidate_score": round(val["score"], 4),
                        "lr": lrs})
        if verbose:
            print(f"  [{tag} {epoch:02d}/{epochs}] loss={tr_loss:.4f} "
                  f"raw={val_raw['score']:.4f} ema={val_ema['score']:.4f} "
                  f"-> cand={kind} {val['score']:.4f} "
                  f"(acc={val['accuracy']:.4f} mrec={val['macro_recall']:.4f}) "
                  f"({time.time() - t0:.0f}s)", flush=True)

        if best is None or val["score"] > best["val"]["score"]:
            best = {"state": state, "kind": kind, "val": val, "epoch": epoch,
                    "eligible": True}
            bad = 0
        else:
            bad += 1

        if save_last and last_state_path is not None:
            save_last_state(last_state_path, model, ema, optimizer, scheduler, scaler,
                            tag, epoch, bad, best, history)
        if bad >= phase["patience"]:
            print(f"  [{tag}] early stopping (no improvement in {phase['patience']} epochs)",
                  flush=True)
            break

    if best is None:
        if fallback is None:                     # no epoch completed at all
            raise RuntimeError(f"[{tag}] no epoch completed - cannot select a checkpoint")
        print(f"  [{tag}] WARNING: no epoch reached val_acc >= {acc_floor:.2f}; "
              f"keeping the best-scoring (ineligible) weights as fallback", flush=True)
        best = fallback
    model.load_state_dict(best["state"])
    ema.__init__(model, EMA_DECAY)          # re-anchor EMA at the restored best
    return best


# ------------------------------------------------------------------
#  In-run pre-flight: dummy forward/backward + Grad-CAM layer
# ------------------------------------------------------------------
def self_check(model, class_w, batch=2):
    """Dummy forward/backward, 5-class output, ordinal head, Grad-CAM layer."""
    print("Self-check: dummy forward/backward ...", flush=True)
    model.train()
    x = to_channels_last(torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))
    logits, ord_logits = model.forward_with_aux(x)
    assert tuple(logits.shape) == (batch, NUM_CLASSES), f"bad 5-class shape {logits.shape}"
    if model.ordinal_head is None:
        assert ord_logits is None
    else:
        assert tuple(ord_logits.shape) == (batch, NUM_ORDINAL_THRESHOLDS), \
            f"bad ordinal shape {ord_logits.shape}"
    y = torch.arange(batch, device=DEVICE) % NUM_CLASSES
    loss = V5Loss(class_w).to(DEVICE)((logits, ord_logits), y)
    loss.backward()
    if model.head[1].weight.grad is None:
        raise RuntimeError("no gradient reached the 5-class head")
    if model.ordinal_head is not None and model.ordinal_head.weight.grad is None:
        raise RuntimeError("no gradient reached the auxiliary ordinal head")
    model.zero_grad(set_to_none=True)

    acts = {}
    hook = gradcam_target_layer(model).register_forward_hook(
        lambda m, i, o: acts.setdefault("a", o.detach()))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))
    hook.remove()
    shape = tuple(acts["a"].shape)
    if shape[1] != 1792 or shape[3] != IMG_SIZE // 32:
        raise RuntimeError(f"unexpected Grad-CAM feature map {shape}")
    print(f"Self-check OK: logits{tuple(logits.shape)} | loss={loss.item():.4f} | "
          f"Grad-CAM layer {GRADCAM_LAYER.split()[0]} -> {shape}", flush=True)


def v3_reference() -> dict:
    """Documented V3 five-class test metrics (the model V5 must beat)."""
    ref = {"source": "cnn/eye_meta_v3.json", "model": "resnet50 @384px (V3)"}
    try:
        meta = json.loads((MODEL_DIR / "eye_meta_v3.json").read_text(encoding="utf-8"))
        tm = meta.get("test_metrics", {})
        ref["test_accuracy"] = tm.get("accuracy")
        ref["test_macro_recall"] = tm.get("macro_recall")
        ref["test_macro_f1"] = tm.get("macro_f1")
    except Exception:
        ref.update({"test_accuracy": 0.8333, "test_macro_recall": 0.6464,
                    "test_macro_f1": 0.6582, "note": "documented fallback values"})
    return ref


def print_final_report(official_mode, official, test_std, test_tta, val_std, val_tta,
                       global_best, phase_results, elapsed, results_path, smoke=False):
    pc = official["per_class"]
    cm = np.array(official["confusion_matrix"])
    acc_ok = official["accuracy"] >= 0.95
    mrec_ok = official["macro_recall"] >= 0.95
    ref = v3_reference()
    print("=" * 78)
    print("FINAL EYE V5 FIVE-CLASS REPORT"
          + ("   [SMOKE RUN - NOT A REAL RESULT]" if smoke else ""))
    print(f"architecture   : {ARCH_NAME} (EfficientNet-B4, ImageNet IMAGENET1K_V1)")
    print(f"input          : {IMG_SIZE}x{IMG_SIZE} fundus448 cache | {PREPROCESS_DESC}")
    print(f"best checkpoint: phase={global_best['phase']} epoch={global_best['epoch']} "
          f"weights={global_best['kind']} "
          f"eligible={bool(global_best.get('eligible', True))} "
          f"(selected on VALIDATION only)")
    print(f"val (std) acc={val_std['accuracy']:.4f} macro_recall={val_std['macro_recall']:.4f} "
          f"macro_f1={val_std['macro_f1']:.4f} score={val_std['score']:.4f}")
    print(f"val (TTA) acc={val_tta['accuracy']:.4f} macro_recall={val_tta['macro_recall']:.4f} "
          f"macro_f1={val_tta['macro_f1']:.4f} score={val_tta['score']:.4f}")
    print(f"official test mode: {official_mode.upper()} (chosen on VALIDATION only)")
    print("-" * 78)
    print(f"TEST ACCURACY     = {official['accuracy'] * 100:.2f}%")
    print(f"TEST MACRO RECALL = {official['macro_recall'] * 100:.2f}%")
    print(f"TEST MACRO F1     = {official['macro_f1'] * 100:.2f}%   "
          f"MACRO PRECISION = {official['macro_precision'] * 100:.2f}%")
    print(f"Cohen kappa = {official['cohen_kappa']:.4f}   "
          f"quadratic weighted kappa = {official['quadratic_weighted_kappa']:.4f}")
    print("Per-class recall (test):")
    for n in CLASS_NAMES:
        print(f"  {n:<15} R={pc[n]['recall']:.4f}  P={pc[n]['precision']:.4f}  "
              f"F1={pc[n]['f1']:.4f}  (n={pc[n]['support']})")
    print("Confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{n[:6]:>7}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n[:7]:>7} " + " ".join(f"{int(v):>7}" for v in cm[i]))
    print("-" * 78)
    if ref.get("test_accuracy") is not None:
        beats = (official["accuracy"] > ref["test_accuracy"]
                 and official["macro_recall"] > ref["test_macro_recall"])
        print(f"V3 reference ({ref['model']}): test_acc={ref['test_accuracy']:.4f} "
              f"macro_recall={ref['test_macro_recall']:.4f}")
        print("V5 vs V3 -> " + ("V5 BEATS V3 (integration candidate)"
                                if beats else "V5 does NOT beat V3 (keep V3 as fallback)"))
    print(f"TARGET accuracy>=95%    : {'ACHIEVED' if acc_ok else 'NOT ACHIEVED'} "
          f"({official['accuracy'] * 100:.2f}%)")
    print(f"TARGET macro-recall>=95%: {'ACHIEVED' if mrec_ok else 'NOT ACHIEVED'} "
          f"({official['macro_recall'] * 100:.2f}%)")
    print("-" * 78)
    print("Phase summary (validation-selected candidates):")
    for p in phase_results:
        v = p["val"]
        print(f"  {p['phase']} ({p['name']}): epoch={p['best_epoch']} "
              f"weights={p['weight_kind']} val_acc={v['accuracy']:.4f} "
              f"val_macro_recall={v['macro_recall']:.4f} score={v['score']:.4f}")
    print(f"results json   : {results_path.name}")
    print(f"total time     : {elapsed:.0f}s")
    print("=" * 78)


# ------------------------------------------------------------------
#  Main - staged transfer learning, validation-only selection, ONE test pass
# ------------------------------------------------------------------
def training_config_json(args, epochs_override):
    return {
        "transfer_learning": ("staged: P0 head warm-up (frozen ImageNet backbone) -> "
                              "P1 upper-stage unfreeze -> P2 full fine-tune with "
                              "differential LRs"),
        "phases": [{"tag": p["tag"], "name": p["name"],
                    "epochs_max": int(epochs_override or p["epochs"]),
                    "patience": p["patience"],
                    "groups": [{"name": lab, "lr": lr} for lab, lr, _ in p["groups"]]}
                   for p in PHASES],
        "batch_size": args.batch_size, "num_workers": args.workers,
        "optimizer": "AdamW (no weight decay on 1-D / BN / bias parameters)",
        "weight_decay": WEIGHT_DECAY, "scheduler": "CosineAnnealingLR (per phase)",
        "amp": USE_AMP, "channels_last": CHANNELS_LAST, "grad_clip": GRAD_CLIP,
        "ema_decay": EMA_DECAY, "seed": args.seed, "head_dropout": HEAD_DROPOUT,
        "loss": (f"class-weighted soft-ordinal CE (soft target {ORD_TARGET_MAIN} true / "
                 f"{ORD_TARGET_ADJ} adjacent, boundaries 0.20; sqrt inverse-frequency "
                 f"class weights alpha={W_ALPHA}, mean-normalised, TRAIN split only) "
                 f"+ {ORD_AUX_WEIGHT} * CORAL BCE on 4 cumulative ordinal thresholds"),
        "augmentation": ("hflip 0.5, affine rot 10deg / translate 6% / scale 0.90-1.10, "
                         "colorjitter 0.2, gaussian blur p=0.15, ImageNet normalisation; "
                         "val/test deterministic"),
        "mixup_cutmix": "not used (never validated as beneficial for this dataset)",
        "auxiliary_ordinal_head": bool(args.ordinal_head),
        "checkpoint_score": SCORE_WEIGHTS,
        "accuracy_floor_for_selection": args.acc_floor,
        "candidate_check": ("raw and EMA weights validated every epoch; the better "
                            "eligible one becomes the checkpoint"),
        "tta_selected_on": "validation only",
        "test_usage": "single pass, only after checkpoint + TTA mode were frozen",
    }


def main():
    ap = argparse.ArgumentParser(
        description="Eye V5 five-class trainer (EfficientNet-B4 @448, staged transfer learning)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                    help=f"default {DEFAULT_BATCH} (VRAM-measured); use 8 if OOM")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"default {DEFAULT_WORKERS} (pagefile-safe on Windows)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch per phase on a small class-balanced subset, "
                         "validation only, writes no production V5 artefacts")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from eye_efficientnet_b4_v5_last.pth")
    ap.add_argument("--ordinal-head", dest="ordinal_head", action="store_true", default=True)
    ap.add_argument("--no-ordinal-head", dest="ordinal_head", action="store_false",
                    help="ablation: primary 5-class head only, no auxiliary ordinal head")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--acc-floor", type=float, default=ACC_FLOOR,
                    help="reject validation candidates whose accuracy is below this "
                         f"floor, so a checkpoint can never win by trading accuracy "
                         f"away (default {ACC_FLOOR})")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("EYE V5 - FIVE-CLASS DR SEVERITY - STAGED TRANSFER LEARNING (EfficientNet-B4 @448)")
    print("TASK: 0=No_DR 1=Mild 2=Moderate 3=Severe 4=Proliferate_DR  (never binary)")
    print(f"Device: {DEVICE} | AMP: {USE_AMP} | channels_last: {CHANNELS_LAST} | "
          f"batch={args.batch_size} workers={args.workers}"
          + (" | *** SMOKE RUN ***" if args.smoke else ""))
    print("=" * 78)

    set_seed(args.seed)
    dfs = verify_all()
    class_w = compute_class_weights(dfs["train"])
    criterion = V5Loss(class_w).to(DEVICE)

    model = build_model(True, args.ordinal_head).to(DEVICE)
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    ema = ModelEMA(model, EMA_DECAY)
    ema_model = build_model(False, args.ordinal_head).to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {ARCH_NAME} | params={n_params / 1e6:.2f}M | "
          f"classifier out_features={model.head[1].out_features} | ordinal_head="
          + ("Linear(1792->4) auxiliary" if args.ordinal_head else "disabled"))
    self_check(model, class_w)

    limit = 48 if args.smoke else None
    loaders = build_loaders(dfs, args.batch_size, args.workers, limit)
    print(f"Loaders: train={len(loaders[0].dataset)} val={len(loaders[1].dataset)} "
          f"(batch={args.batch_size}, workers={args.workers})", flush=True)
    epochs_override = 1 if args.smoke else None

    if args.smoke:
        model_path = MODEL_DIR / f"{SMOKE_PREFIX}.pth"
        meta_path = MODEL_DIR / f"{SMOKE_PREFIX}_meta.json"
        results_path = MODEL_DIR / f"{SMOKE_PREFIX}_results.json"
        last_path = None
    else:
        model_path, meta_path, results_path = MODEL_PATH, META_PATH, RESULTS_PATH
        last_path = LAST_STATE_PATH

    history, phase_results, global_best = [], [], None
    start_phase, resume_state = 0, None
    if args.resume:
        if last_path is None or not last_path.is_file():
            raise FileNotFoundError(f"cannot --resume: {last_path} not found")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        ema.shadow = {k: (v.to(DEVICE) if v.is_floating_point() else v)
                      for k, v in ck["ema_shadow"].items()}
        history = list(ck.get("history", []))
        resume_state = ck
        tags = [p["tag"] for p in PHASES]
        if ck.get("phase_tag") not in tags:
            raise ValueError(f"unknown phase tag in resume state: {ck.get('phase_tag')}")
        start_phase = tags.index(ck["phase_tag"])
        print(f"RESUMED from {last_path.name}: phase={ck['phase_tag']} "
              f"epoch={ck['epoch']} | history={len(history)} entries", flush=True)

    # ---- staged transfer learning ------------------------------------
    for pi in range(start_phase, len(PHASES)):
        phase = PHASES[pi]
        ep = epochs_override or phase["epochs"]
        print(f"\n--- PHASE {phase['tag']}: {phase['name']} (max {ep} epochs, "
              f"patience {phase['patience']}) ---", flush=True)
        best = run_phase(phase, model, ema, ema_model, loaders, criterion, history,
                         last_state_path=last_path,
                         resume=(resume_state if pi == start_phase else None),
                         epochs_override=epochs_override, save_last=not args.smoke,
                         acc_floor=args.acc_floor)
        phase_results.append({"phase": phase["tag"], "name": phase["name"],
                              "best_epoch": best["epoch"], "weight_kind": best["kind"],
                              "eligible": bool(best.get("eligible", True)),
                              "val": best["val"]})
        print(f"  [{phase['tag']}] phase best: epoch={best['epoch']} "
              f"weights={best['kind']} eligible={bool(best.get('eligible', True))} "
              f"val_acc={best['val']['accuracy']:.4f} "
              f"val_macro_recall={best['val']['macro_recall']:.4f} "
              f"score={best['val']['score']:.4f}", flush=True)
        if not args.smoke:
            ph_path = MODEL_DIR / f"eye_efficientnet_b4_v5_{phase['tag'].lower()}.pth"
            atomic_save(ph_path, {"model_state_dict": best["state"],
                                  "meta": ckpt_meta(best["kind"], phase["tag"],
                                                    best["epoch"], best["val"],
                                                    args.ordinal_head)})
        rank = (bool(best.get("eligible", True)), best["val"]["score"])
        g_rank = ((bool(global_best.get("eligible", True)), global_best["val"]["score"])
                  if global_best else None)
        if g_rank is None or rank > g_rank:
            global_best = {"state": best["state"], "kind": best["kind"],
                           "val": best["val"], "epoch": best["epoch"],
                           "phase": phase["tag"],
                           "eligible": bool(best.get("eligible", True))}

    model.load_state_dict(global_best["state"])
    print(f"\nBest overall (validation only): phase={global_best['phase']} "
          f"epoch={global_best['epoch']} weights={global_best['kind']} "
          f"eligible={bool(global_best.get('eligible', True))} "
          f"val_acc={global_best['val']['accuracy']:.4f} "
          f"val_macro_recall={global_best['val']['macro_recall']:.4f} "
          f"score={global_best['val']['score']:.4f}", flush=True)
    if not global_best.get("eligible", True):
        print("WARNING: no checkpoint satisfied the accuracy floor - the selected "
              "weights are marked NOT eligible (do not ship without review).",
              flush=True)
    meta_base = ckpt_meta(global_best["kind"], global_best["phase"], global_best["epoch"],
                          global_best["val"], args.ordinal_head,
                          extra={"phase_results": phase_results,
                                 "checkpoint_eligible": bool(global_best.get("eligible", True)),
                                 "training_config": training_config_json(args, epochs_override),
                                 "training_history": history})

    # ---- TTA selection: VALIDATION ONLY -------------------------------
    val_std = evaluate(model, loaders[1], "std")
    val_tta = evaluate(model, loaders[1], "tta")
    use_tta = val_tta["score"] > val_std["score"]
    print(f"TTA selection (validation only): std={val_std['score']:.4f} "
          f"tta={val_tta['score']:.4f} -> {'TTA' if use_tta else 'standard'}", flush=True)

    # ---- Persist the frozen checkpoint + metadata ----------------------
    atomic_save(model_path, {"model_state_dict": global_best["state"], "meta": meta_base})
    meta_path.write_text(json.dumps(meta_base, indent=2), encoding="utf-8")
    print(f"metadata written -> {meta_path.name}", flush=True)
    if args.smoke:
        print("SMOKE RUN: test split NOT touched; stopping before the test pass.",
              flush=True)
        print_final_report("std", val_std, val_std, val_std, val_std, val_tta,
                           global_best, phase_results, time.time() - t0, results_path,
                           smoke=True)
        return

    # ---- FINAL TEST: ONE single pass, after every selection is frozen ---
    test_loader = make_loader(FundusCachedDataset(
        "test", dfs["test"]["id_code"], dfs["test"]["diagnosis"], False),
        args.batch_size, args.workers, False)
    print("--- SINGLE final test evaluation (standard + TTA, one pass) ---", flush=True)
    test_std, test_tta, _ = evaluate(model, test_loader, "both")
    official_mode = "tta" if use_tta else "std"
    official = test_tta if use_tta else test_std

    results = {
        "version": VERSION,
        "architecture": ARCH_NAME,
        "backbone": BACKBONE_NAME,
        "pretrained_weights": meta_base["pretrained_weights"],
        "task": "five_class_dr_severity",
        "task_statement": meta_base["task_statement"],
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "image_size": [IMG_SIZE, IMG_SIZE],
        "normalization": {"mean": MEAN, "std": STD},
        "preprocessing": PREPROCESS_DESC,
        "gradcam_layer": GRADCAM_LAYER,
        "forward_returns": meta_base["forward_returns"],
        "training_config": meta_base["training_config"],
        "phase_results": phase_results,
        "training_history": history,
        "validation_standard": val_std,
        "validation_tta": val_tta,
        "tta_selected_on": "validation only",
        "tta_used_for_official": bool(use_tta),
        "official_test_mode": official_mode,
        "official_test_metrics": official,
        "test_standard_inference": test_std,
        "test_tta": test_tta,
        "best_checkpoint": {"phase": global_best["phase"], "epoch": global_best["epoch"],
                            "weight_kind": global_best["kind"],
                            "eligible": bool(global_best.get("eligible", True)),
                            "validation": global_best["val"]},
        "v3_reference": v3_reference(),
        "target_achieved": {"accuracy_ge_95": bool(official["accuracy"] >= 0.95),
                            "macro_recall_ge_95": bool(official["macro_recall"] >= 0.95)},
        "test_set_usage": ("untouched during training: no threshold, epoch, architecture, "
                           "preprocessing or hyper-parameter choice used the test split; "
                           "evaluated exactly once here"),
        "training_time_s": round(time.time() - t0, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results written -> {results_path.name}", flush=True)
    print_final_report(official_mode, official, test_std, test_tta, val_std, val_tta,
                       global_best, phase_results, time.time() - t0, results_path)


if __name__ == "__main__":
    main()
