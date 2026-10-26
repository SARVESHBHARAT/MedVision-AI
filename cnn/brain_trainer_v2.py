"""
MedVision AI — Brain MRI Tumor CNN Trainer  (V2, single strong run)
====================================================================
Task   : Brain MRI tumor classification (4 classes)
Dataset: Balanced Brain MRI tumor dataset shipped as a fixed split
         cnn/brain/Training/ (5,600 = 1,400/class) + cnn/brain/Testing/
         (1,600 = 400/class).

V2 changes vs V1 (cnn/brain_trainer.py)
  * Fix silent train-augmentation bug: V1 mutated the *shared* ImageFolder
    transform via `val_ds.dataset.transform = val_transform`, so the training
    split also ran WITHOUT augmentation.  V2 uses an explicit per-subset
    transform wrapper -> train augmenter is actually applied.
  * Loss: class-weighted CrossEntropy (mild inverse-frequency) + label
    smoothing = 0.10  (V1 used plain, un-weighted CE).
  * Optimizer: AdamW (decoupled weight decay, excluded from BN/bias) instead
    of Adam.  Differential learning rates across backbone stages in Phase 2.
  * Scheduler: cosine annealing (V1 used ReduceLROnPlateau).
  * Checkpoint selection metric: **validation macro-F1** (V1 used val acc).
  * Pretrained weights: ImageNet1K V2 (resnet50-11ad3fa6) — already cached.
  * Saves **brain_resnet50_v2.pth** / **brain_meta_v2.json** — the existing
    brain_resnet50.pth is NEVER touched.

The final V2 state_dict keeps the EXACT same head layout as
medvision_inference.build_resnet50() so the trained file is a drop-in
for the app (no inference change required).
"""
import os
import copy
import json
import time
import random
import collections
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import torchvision

# --------------------------------------------------------------------------
#  Global configuration
# --------------------------------------------------------------------------
VERSION = "V2"

BASE_DIR   = Path(__file__).resolve().parent.parent
TRAIN_DIR  = BASE_DIR / "cnn" / "brain" / "Training"      # Kaggle train split
TEST_DIR   = BASE_DIR / "cnn" / "brain" / "Testing"       # Kaggle test split (HANDS OFF)
MODEL_DIR  = BASE_DIR / "cnn"

# Output files (V1 model brain_resnet50.pth is preserved)
MODEL_PATH_V2 = MODEL_DIR / "brain_resnet50_v2.pth"
META_PATH_V2  = MODEL_DIR / "brain_meta_v2.json"

IMG_SIZE    = 224          # must stay 224 (inference is hard-coded 224)
BATCH_SIZE  = 32           # safe on RTX 4050 8 GB at 224px
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED             = 42
VAL_FRAC         = 0.15            # stratified val carved ONLY from Training/
PHASE1_EPOCHS    = 3               # head warm-up, backbone frozen
PHASE2_EPOCHS    = 20              # full fine-tune, cosine decay
PATIENCE         = 6               # early stop on val macro-F1
WD               = 1e-4            # AdamW decoupled weight decay
LABEL_SMOOTHING  = 0.10
GRAD_CLIP        = 1.0

# differential learning rates (Phase 2)
LR_PHASE1_HEAD = 5e-3
LR_MAP = {
    "stem": 3e-6,   # conv1 + BN1
    "layer1": 1e-5,
    "layer2": 2e-5,
    "layer3": 4e-5,
    "layer4": 8e-5,
    "head":  2e-4,
}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True
print(f"[Brain CNN V2] Device: {DEVICE} | torch {torch.__version__} | torchvision {torchvision.__version__}")
# --------------------------------------------------------------------------
#  Augmentation / transforms
# --------------------------------------------------------------------------
def train_transform():
    # Medically reasonable: mild geometric + gentle grayscale intensity jitter.
    # MRI is effectively grayscale -> avoid aggressive hue/saturation distortion.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=12, fill=0),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def val_transform():
    # Identical to medvision_inference.img_transform so final numbers match app.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# --------------------------------------------------------------------------
#  Per-subset transform wrapper (fixes the V1 shared-transform bug)
# --------------------------------------------------------------------------
class TransformSubset(Dataset):
    """Subset of a base dataset with an INDEPENDENT per-split transform.

    Base ImageFolder is created with transform=None so images are loaded
    once; each split applies its own transform, avoiding mutation of a shared
    dataset object (the V1 bug that disabled training augmentation).
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return self.transform(img), label


# --------------------------------------------------------------------------
#  Stratified splitter (no sklearn dependency), on TRAIN_DIR only
# --------------------------------------------------------------------------
def stratify_split(full, val_frac, seed):
    rng = random.Random(seed)
    per_class = collections.defaultdict(list)
    for i, y in enumerate(full.targets):       # read labels, no image decode
        per_class[y].append(i)
    train_idx, val_idx = [], []
    for y, idxs in per_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(val_frac * len(idxs))))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


# --------------------------------------------------------------------------
#  Class weights (mild inverse-frequency) computed from the actual dataset
# --------------------------------------------------------------------------
def class_weights(full, train_idx):
    counts = collections.Counter(full.targets[i] for i in train_idx)
    n = len(train_idx)
    k = len(counts)
    w = {}
    for y, c in counts.items():
        v = n / (k * c)
        w[y] = max(0.5, min(2.0, v))     # clamp to avoid over/under emphasising
    ws = torch.tensor([w[y] for y in sorted(w)], dtype=torch.float32)
    ws = ws / ws.mean()                   # normalise to mean 1
    w_map = {y: float(ws[i]) for i, y in enumerate(sorted(w))}
    return ws, w_map


# --------------------------------------------------------------------------
#  Data leakage checks (training/validation NEVER touch Testing/)
# --------------------------------------------------------------------------
def leakage_check(base_train, test_ds):
    train_files = collections.Counter()
    for p, _ in base_train.samples:
        train_files[Path(p).name] += 1
    overlaps = 0
    shared = []
    for p, _ in test_ds.samples:
        name = Path(p).name
        if name in train_files:
            overlaps += 1
            shared.append(name)
    if overlaps:
        print(f"  [LEAK] WARNING: {overlaps} image file(s) appear in BOTH "
              f"Training/ and Testing/ -> {shared[:10]}")
    else:
        print("  [LEAK] OK — 0 overlapping image files between splits.")
    return overlaps
# --------------------------------------------------------------------------
#  Loaders
# --------------------------------------------------------------------------
def get_loaders():
    for d, label in ((TRAIN_DIR, "Training"), (TEST_DIR, "Testing")):
        if not d.is_dir():
            raise FileNotFoundError(
                f"[Brain CNN V2] {label} dataset not found at {d}.\n"
                f"Download the Kaggle 'Brain Tumor MRI Dataset' "
                f"(masoudnickparvar/brain-tumor-mri-dataset) and place its "
                f"Training/ and Testing/ folders under cnn/brain/ .")

    # Base datasets loaded WITHOUT transform; subsets apply their own.
    base_train = datasets.ImageFolder(TRAIN_DIR, transform=None)
    test_ds    = datasets.ImageFolder(TEST_DIR,  transform=val_transform())

    class_names = base_train.classes
    num_classes = len(class_names)

    train_idx, val_idx = stratify_split(base_train, VAL_FRAC, SEED)
    train_ds = TransformSubset(base_train, train_idx, train_transform())
    val_ds   = TransformSubset(base_train, val_idx,   val_transform())

    assert set(train_idx).isdisjoint(val_idx), "LEAK: train/val indices overlap!"

    weights, w_map = class_weights(base_train, train_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    train_counts = collections.Counter(base_train.targets[i] for i in train_idx)
    print(f"[Data] classes      : {class_names}")
    print(f"[Data] train        : {len(train_idx)} | val: {len(val_idx)} | "
          f"test: {len(test_ds)}")
    print(f"[Data] class counts (train): "
          f"{ {class_names[c]: train_counts[c] for c in sorted(train_counts)} }")
    print(f"[Data] class weights: "
          f"{ {class_names[k]: round(v, 3) for k, v in w_map.items()} }")
    overlap = leakage_check(base_train, test_ds)
    return (train_loader, val_loader, test_loader, class_names, num_classes,
            weights, overlap)


# --------------------------------------------------------------------------
#  Model : ResNet50 + EXACT inference-compatible head
# --------------------------------------------------------------------------
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def build_param_groups(model, phase1):
    """Optimizer param-groups with differential LRs.

    weight_decay is set to 0 for all BatchNorm weight/bias and all biases
    (AdamW keeps decoupled decay only on true weights)."""
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if phase1:
            if not name.startswith("fc."):
                continue
            lr, wd = LR_PHASE1_HEAD, 0.0
        else:
            if name.startswith("fc."):
                lr = LR_MAP["head"]
            elif name.startswith("layer4."):
                lr = LR_MAP["layer4"]
            elif name.startswith("layer3."):
                lr = LR_MAP["layer3"]
            elif name.startswith("layer2."):
                lr = LR_MAP["layer2"]
            elif name.startswith("layer1."):
                lr = LR_MAP["layer1"]
            else:
                lr = LR_MAP["stem"]
            wd = 0.0 if (param.ndim < 2 or "bn" in name.lower()) else WD
        groups.setdefault((lr, wd), []).append(param)
    return [{"params": params, "lr": lr, "weight_decay": wd}
            for (lr, wd), params in groups.items()]
# --------------------------------------------------------------------------
#  Metrics  (confusion-matrix based; macro recall / macro-F1 / accuracy)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=DEVICE)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x).argmax(1)
        for t, p in zip(y.cpu().tolist(), out.cpu().tolist()):
            conf[t, p] += 1
    tp = conf.diag().float()
    pred = conf.sum(1).float().clamp(min=1)    # per-class TP+FN
    cond = conf.sum(0).float().clamp(min=1)    # per-class TP+FP
    recall = (tp / pred).mean().item()          # = balanced accuracy (macro recall)
    prec   = (tp / cond).mean().item()
    f1     = (2 * tp / (pred + cond)).mean().item()
    acc    = (tp.sum() / conf.sum()).item()
    per_class_recall = (tp / pred).tolist()
    return {"acc": acc, "macro_recall": recall, "macro_f1": f1,
            "macro_prec": prec, "per_class_recall": per_class_recall}


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        running += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total   += x.size(0)
    return running / total, correct / total


# --------------------------------------------------------------------------
#  Training orchestration (Phase 1 warm-up -> Phase 2 fine-tune)
# --------------------------------------------------------------------------
def run_phases(model, criterion, train_loader, val_loader, num_classes, class_names):
    best = {"macro_f1": 0.0, "weights": None}
    history = []

    # Phase 1 : warm-up the randomly initialised head (backbone frozen)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    opt1 = optim.AdamW(build_param_groups(model, phase1=True))
    sched1 = CosineAnnealingLR(opt1, T_max=PHASE1_EPOCHS, eta_min=1e-4)
    print("\n[Phase 1] Warm-up head (backbone frozen)...")
    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        tr_l, tr_a = train_one_epoch(model, train_loader, criterion, opt1)
        sched1.step()
        v = evaluate(model, val_loader, num_classes)
        print(f"  E{epoch:02d}/{PHASE1_EPOCHS} | train {tr_a:.4f} | "
              f"val acc {v['acc']:.4f} | macro-F1 {v['macro_f1']:.4f} "
              f"| {time.time()-t0:.0f}s")

    # Phase 2 : full fine-tune, differential LRs, cosine decay
    for p in model.parameters():
        p.requires_grad = True
    opt2 = optim.AdamW(build_param_groups(model, phase1=False))
    sched2 = CosineAnnealingLR(opt2, T_max=PHASE2_EPOCHS, eta_min=1e-6)

    patience_ctr = 0
    print("\n[Phase 2] Full fine-tune (diff-LR, cosine)...")
    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        tr_l, tr_a = train_one_epoch(model, train_loader, criterion, opt2)
        sched2.step()
        v = evaluate(model, val_loader, num_classes)
        history.append({"phase": 2, "epoch": epoch, **v})
        msg = (f"  F{epoch:02d}/{PHASE2_EPOCHS} | train {tr_a:.4f} | "
               f"val acc {v['acc']:.4f} | macro-F1 {v['macro_f1']:.4f} "
               f"| {time.time()-t0:.0f}s")
        if v["macro_f1"] > best["macro_f1"]:
            best["macro_f1"] = v["macro_f1"]
            best["weights"] = copy.deepcopy(model.state_dict())
            best["meta"] = v
            patience_ctr = 0
            msg += "  <== BEST (macro-F1)"
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                msg += f"  | early stop @ {epoch}"
                print(msg)
                break
        print(msg)

    if best["weights"] is not None:
        model.load_state_dict(best["weights"])
    return best, history
# --------------------------------------------------------------------------
#  main
# --------------------------------------------------------------------------
def round_nested(obj, nd=4):
    """Recursively round floats, preserving lists/dicts/matrices intact.

    Numeric values are rounded to `nd` decimals; any nested list, tuple or
    dict (e.g. per-class recall lists or a confusion-matrix matrix) is
    preserved as-is (ints/strings pass through unchanged)."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_nested(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_nested(v, nd) for v in obj]
    return obj


def load_val_meta(m):
    """Build the best_val dict from evaluate() output (may be None)."""
    if not m:
        return {"macro_f1": 0.0, "acc": 0.0, "macro_recall": 0.0,
                "macro_prec": 0.0}
    # round scalars and keep any nested structure intact
    return round_nested({k: v for k, v in m.items()
                         if isinstance(v, (int, float))})


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if MODEL_PATH_V2.exists():
        # safety: never silently clobber a previous V2 run
        backup = MODEL_DIR / "brain_resnet50_v2_prev.pth"
        print(f"[Save] preserving previous v2 -> {backup.name}")
        backup.write_bytes(MODEL_PATH_V2.read_bytes())

    (train_loader, val_loader, test_loader, class_names, num_classes,
     weights, overlap) = get_loaders()
    if overlap:
        # do NOT proceed if the fixed Kaggle splits share files
        raise RuntimeError("Data leakage detected between Training/ and "
                           "Testing/; aborting.")

    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE),
                                    label_smoothing=LABEL_SMOOTHING)
    print(f"[Loss ] CrossEntropy(class_weighted + label_smoothing="
          f"{LABEL_SMOOTHING})")

    best, history = run_phases(model, criterion, train_loader, val_loader,
                               num_classes, class_names)

    # ----- final evaluation on the untouched test set -------------------
    te = evaluate(model, test_loader, num_classes)
    print(f"\n[Test ] acc={te['acc']:.4f} macro_recall={te['macro_recall']:.4f} "
          f"macro_f1={te['macro_f1']:.4f}")

    torch.save({
        "model_state_dict": best["weights"],
        "class_names": class_names,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "version": VERSION,
    }, MODEL_PATH_V2)

    wmap = {c: round(float(w), 4)
            for c, w in zip(class_names, weights.tolist())}
    meta = {
        "version": VERSION,
        "model_type": "resnet50",
        "task": "brain_tumor",
        "dataset": "balanced Brain MRI tumor dataset "
                    "(cnn/brain: 5600 Training / 1600 Testing, 1400+400 per class)",
        "class_names": class_names,
        "num_classes": num_classes,
        "class_weights": wmap,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "mean": MEAN,
        "std": STD,
        "loss": {
            "type": "cross_entropy",
            "class_weighted": True,
            "label_smoothing": LABEL_SMOOTHING,
        },
        "optimizer": {
            "type": "AdamW",
            "weight_decay": WD,
            "grad_clip": GRAD_CLIP,
            "phase1_head_lr": LR_PHASE1_HEAD,
            "phase2_lr_map": LR_MAP,
        },
        "scheduler": {"type": "CosineAnnealingLR",
                      "phase1_tmax": PHASE1_EPOCHS,
                      "phase2_tmax": PHASE2_EPOCHS},
        "pretrained_weights": "IMAGENET1K_V2 (resnet50-11ad3fa6)",
        "augmentation": "Resize232->RandomResizedCrop224(sc=0.75-1.0), "
                        "HFlip, Rot12, ColorJitter(brightness0.1, contrast0.1)",
        "val_split": {"frac": VAL_FRAC, "source": "Training/ only", "seed": SEED},
        "early_stopping": {"patience": PATIENCE, "metric": "val_macro_f1"},
        "best_val": load_val_meta(best.get("meta")),
        "test": round_nested(te),
        "per_class_recall_test":
            {c: round(float(r), 4) for c, r in
             zip(class_names, te["per_class_recall"])},
    }
    json.dump(meta, open(META_PATH_V2, "w"), indent=2)

    print(f"[Save] {MODEL_PATH_V2.name}")
    print(f"[Save] {META_PATH_V2.name}")
    print(f"\n[Brain CNN V2] Done. Best val macro-F1 {best.get('macro_f1', 0):.4f} "
          f"| Test acc {te['acc']:.4f} | Test macro-rec {te['macro_recall']:.4f}")


if __name__ == "__main__":
    main()