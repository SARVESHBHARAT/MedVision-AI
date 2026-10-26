"""
MedVision AI — Skin Cancer CNN Trainer (V3 aggressive config)
ResNet50 Transfer Learning | PyTorch
Dataset: ISIC Skin Cancer 9 Classes
Goal: maximize validation accuracy + macro recall, with extra attention to
      melanoma and the other malignant classes (test set untouched until end).
"""
import os, copy, time, json, random, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "cnn" / "skin" / "Skin cancer ISIC The International Skin Imaging Collaboration"
TRAIN_DIR  = DATA_DIR / "Train"
TEST_DIR   = DATA_DIR / "Test"
MODEL_DIR  = BASE_DIR / "cnn"
MODEL_PATH = MODEL_DIR / "skin_resnet50.pth"
META_PATH  = MODEL_DIR / "skin_meta.json"

IMG_SIZE    = 224
BATCH_SIZE  = 64

# ── V3: Training duration (no aggressive early stopping) ──────────────────
HEAD_EPOCHS      = 20       # head-only phase
FINETUNE_EPOCHS  = 50       # full fine-tuning phase
PATIENCE         = 200      # large patience -> effectively no early stopping

# ── V3: Learning rates (head/classifier larger, backbone smaller) ─────────
LR_HEAD              = 1e-3  # fresh classifier head, phase 1
LR_FINETUNE_HEAD     = 1e-4  # classifier head, phase 2
LR_FINETUNE_BACKBONE = 1e-5  # top pre-trained ResNet stage LR, phase 2

# ── V3: Class imbalance ───────────────────────────────────────────────────
MELANOMA_WEIGHT_MULTIPLIER    = 2.5   # extra loss+sampler weight for melanoma
MALIGNANT_WEIGHT_MULTIPLIER   = 1.5   # extra loss+sampler weight for other malignants
MINORITY_SAMPLER_POWER        = 1.3   # >1 -> stronger minority oversampling per-class

# ── V3: Validation checkpoint score (accuracy + macro recall + mel recall) ─
VAL_ACC_WEIGHT          = 0.25
VAL_MACRO_RECALL_WEIGHT = 0.45
MELANOMA_RECALL_WEIGHT  = 0.30

MELANOMA_CLASS    = "melanoma"
MALIGNANT_CLASSES = ("actinic keratosis", "basal cell carcinoma", "squamous cell carcinoma")

NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

print(f"[Skin CNN] Device: {DEVICE}")

# V3: lesion-preserving augmentation. Plain Resize (no aggressive RandomResizedCrop,
# which could cut away important lesion context). Vertical flips are intentionally
# dropped since mirrored dermoscopic texture is not a realistic presentation.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def get_loaders():
    # Two independent ImageFolder instances so train augmentation does NOT leak
    # into val (the old `val_ds.dataset.transform = val_transform` also flipped
    # the shared train ImageFolder, silently disabling all training augmentation).
    train_full = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_full   = datasets.ImageFolder(TRAIN_DIR, transform=val_transform)
    train_idx, val_idx = stratified_split(train_full, val_frac=0.15, seed=42)
    train_ds, val_ds   = Subset(train_full, train_idx), Subset(val_full, val_idx)
    test_ds   = datasets.ImageFolder(TEST_DIR, transform=val_transform)

    train_labels = [train_full.targets[i] for i in train_idx]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=make_class_sampler(train_labels, train_full.classes),
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_names  = train_full.classes
    class_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.int64),
                                  minlength=len(class_names))
    print(f"[Data] Classes : {class_names}")
    print(f"[Data] Train   : {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, class_names, class_counts

def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters(): param.requires_grad = False
    for param in model.layer4.parameters(): param.requires_grad = True
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)

def is_melanoma(name):
    return name.strip().lower() == MELANOMA_CLASS

def is_malignant(name):
    return name.strip().lower() in MALIGNANT_CLASSES

def get_class_weights(class_counts, class_names):
    counts  = class_counts.float()
    total   = counts.sum()
    weights = (total / (len(counts) * counts)).to(DEVICE)

    # Extra loss weight for melanoma + other malignant classes (9-class order kept).
    for i, name in enumerate(class_names):
        if is_melanoma(name):
            weights[i] *= MELANOMA_WEIGHT_MULTIPLIER
        elif is_malignant(name):
            weights[i] *= MALIGNANT_WEIGHT_MULTIPLIER

    weights = weights / weights.mean()

    print("[Class Weights]")
    for i, name in enumerate(class_names):
        print(f"  {name:>28}: {weights[i].item():.4f}")
    return weights

def make_class_sampler(labels, class_names):
    """V3: strong balanced WeightedRandomSampler.

    Per-class base weight = (1/count)**MINORITY_SAMPLER_POWER.  A power > 1 makes
    minority classes oversampled even harder than classic 1/count balancing, and
    melanoma/malignant classes get an extra multiplier on top.
    """
    num_classes = len(class_names)
    labels  = torch.tensor(labels, dtype=torch.int64)
    counts  = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    per_class = (1.0 / counts) ** MINORITY_SAMPLER_POWER
    for i, name in enumerate(class_names):
        if is_melanoma(name):
            per_class[i] *= MELANOMA_WEIGHT_MULTIPLIER
        elif is_malignant(name):
            per_class[i] *= MALIGNANT_WEIGHT_MULTIPLIER
    weights = per_class[labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def stratified_split(dataset, val_frac=0.15, seed=42):
    """Per-class stratified train/val split (stdlib only — no sklearn)."""
    rng = random.Random(seed)
    class_indices = {}
    for i, t in enumerate(dataset.targets):
        class_indices.setdefault(int(t), []).append(i)
    train_idx, val_idx = [], []
    for c in sorted(class_indices):
        idxs = class_indices[c]
        rng.shuffle(idxs)
        # Keep >=1 train sample for every class; >=1 val sample when class has >1.
        n_val = min(max(int(round(len(idxs) * val_frac)), 1), len(idxs) - 1) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx

def compute_confusion_matrix(preds, labels, num_classes):
    """Vectorized confusion matrix: rows = true label, cols = predicted label."""
    idx = labels.long() * num_classes + preds.long()
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)

def metrics_from_confusion_matrix(cm, class_names):
    cm = cm.float()
    tp        = cm.diag()
    support   = cm.sum(dim=1)   # actual count per class (row sum)
    predicted = cm.sum(dim=0)   # predicted count per class (col sum)

    recall    = torch.where(support > 0,   tp / support.clamp(min=1),   torch.zeros_like(tp))
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    f1        = torch.where((precision + recall) > 0,
                             2 * precision * recall / (precision + recall).clamp(min=1e-12),
                             torch.zeros_like(tp))

    return {
        "macro_recall":    recall.mean().item(),
        "macro_precision": precision.mean().item(),
        "macro_f1":        f1.mean().item(),
        "per_class_recall":    {c: round(r.item(), 4) for c, r in zip(class_names, recall)},
        "per_class_precision": {c: round(p.item(), 4) for c, p in zip(class_names, precision)},
        "per_class_f1":        {c: round(f.item(), 4) for c, f in zip(class_names, f1)},
    }

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x); loss = criterion(out, y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, num_classes):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x); loss = criterion(out, y)
        preds = out.argmax(1)
        loss_sum += loss.item() * x.size(0)
        correct  += (preds == y).sum().item()
        total    += x.size(0)
        cm += compute_confusion_matrix(preds.cpu(), y.cpu(), num_classes)
    return loss_sum / total, correct / total, cm

def make_head_optimizer(model):
    """Phase 1 optimizer: much larger LR for the new classifier head, small LR for
    the top ResNet stage that is also unfrozen during head training."""
    return optim.Adam([
        {"params": model.layer4.parameters(), "lr": LR_FINETUNE_BACKBONE},
        {"params": model.fc.parameters(),     "lr": LR_HEAD},
    ])

def unfreeze_backbone(model):
    """Phase 2 optimizer: full fine-tuning with differential LRs. Pre-trained
    backbone stages get small LRs (stages closer to the input even smaller) and
    the classifier head gets the largest LR — stable for ResNet50."""
    for param in model.parameters(): param.requires_grad = True
    return optim.Adam([
        {"params": model.layer1.parameters(), "lr": LR_FINETUNE_BACKBONE * 0.1},
        {"params": model.layer2.parameters(), "lr": LR_FINETUNE_BACKBONE * 0.3},
        {"params": model.layer3.parameters(), "lr": LR_FINETUNE_BACKBONE * 0.6},
        {"params": model.layer4.parameters(), "lr": LR_FINETUNE_BACKBONE},
        {"params": model.fc.parameters(),     "lr": LR_FINETUNE_HEAD},
    ])

def compute_checkpoint_score(val_acc, macro_recall, melanoma_recall):
    """V3 validation checkpoint score.

    Combines validation accuracy, validation macro recall and melanoma recall so
    we do NOT select a model that reaches good accuracy by sacrificing melanoma /
    minority-class recall. Validation metrics only — the test set is never used
    for model selection.
    """
    return (VAL_ACC_WEIGHT * val_acc
            + VAL_MACRO_RECALL_WEIGHT * macro_recall
            + MELANOMA_RECALL_WEIGHT * melanoma_recall)

def main():
    train_loader, val_loader, test_loader, class_names, class_counts = get_loaders()
    num_classes  = len(class_names)
    melanoma_idx = next((i for i, n in enumerate(class_names) if is_melanoma(n)), 0)

    model = build_model(num_classes)
    weights = get_class_weights(class_counts, class_names)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = make_head_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_score, best_acc = 0.0, 0.0
    best_macro_recall, best_mel_recall, best_weights = 0.0, 0.0, None
    patience_ctr = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_macro_recall": [], "val_melanoma_recall": [], "val_score": []}

    print("\n[Phase 1] Training head (backbone frozen except layer4)...")
    for epoch in range(1, HEAD_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_cm = evaluate(model, val_loader, criterion, num_classes)
        scheduler.step(vl_loss)
        vl_metrics      = metrics_from_confusion_matrix(vl_cm, class_names)
        vl_macro_recall = vl_metrics["macro_recall"]
        vl_mel_recall   = vl_metrics["per_class_recall"][class_names[melanoma_idx]]
        vl_score        = compute_checkpoint_score(vl_acc, vl_macro_recall, vl_mel_recall)
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc"],
                        [tr_loss, tr_acc, vl_loss, vl_acc]):
            history[k].append(v)
        history["val_macro_recall"].append(vl_macro_recall)
        history["val_melanoma_recall"].append(vl_mel_recall)
        history["val_score"].append(vl_score)
        print(f"  Epoch {epoch:02d}/{HEAD_EPOCHS} | Train: {tr_acc:.4f} | Val Acc: {vl_acc:.4f} | "
              f"MacroRecall: {vl_macro_recall:.4f} | MelRecall: {vl_mel_recall:.4f} | {time.time()-t0:.1f}s")
        if vl_score > best_score:
            best_score, best_acc = vl_score, vl_acc
            best_macro_recall, best_mel_recall = vl_macro_recall, vl_mel_recall
            best_weights = copy.deepcopy(model.state_dict()); patience_ctr = 0
            print(f"  ✅ Best score={best_score:.4f} | Acc={best_acc:.4f} | "
                  f"MacroRecall={best_macro_recall:.4f} | MelRecall={best_mel_recall:.4f}")
            print(f"     Per-class recall: {vl_metrics['per_class_recall']}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE: print(f"  ⏹ Early stop (patience {PATIENCE})"); break

    print("\n[Phase 2] Full fine-tuning backbone (differential LRs)...")
    model.load_state_dict(best_weights)
    optimizer2 = unfreeze_backbone(model)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode="min", factor=0.5, patience=3)
    patience_ctr2 = 0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer2)
        vl_loss, vl_acc, vl_cm = evaluate(model, val_loader, criterion, num_classes)
        scheduler2.step(vl_loss)
        vl_metrics      = metrics_from_confusion_matrix(vl_cm, class_names)
        vl_macro_recall = vl_metrics["macro_recall"]
        vl_mel_recall   = vl_metrics["per_class_recall"][class_names[melanoma_idx]]
        vl_score        = compute_checkpoint_score(vl_acc, vl_macro_recall, vl_mel_recall)
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc"],
                        [tr_loss, tr_acc, vl_loss, vl_acc]):
            history[k].append(v)
        history["val_macro_recall"].append(vl_macro_recall)
        history["val_melanoma_recall"].append(vl_mel_recall)
        history["val_score"].append(vl_score)
        print(f"  Fine {epoch:02d}/{FINETUNE_EPOCHS} | Train: {tr_acc:.4f} | Val Acc: {vl_acc:.4f} | "
              f"MacroRecall: {vl_macro_recall:.4f} | MelRecall: {vl_mel_recall:.4f} | {time.time()-t0:.1f}s")
        if vl_score > best_score:
            best_score, best_acc = vl_score, vl_acc
            best_macro_recall, best_mel_recall = vl_macro_recall, vl_mel_recall
            best_weights = copy.deepcopy(model.state_dict()); patience_ctr2 = 0
            print(f"  ✅ Best score={best_score:.4f} | Acc={best_acc:.4f} | "
                  f"MacroRecall={best_macro_recall:.4f} | MelRecall={best_mel_recall:.4f}")
            print(f"     Per-class recall: {vl_metrics['per_class_recall']}")
        else:
            patience_ctr2 += 1
            if patience_ctr2 >= PATIENCE: print(f"  ⏹ Early stop (patience {PATIENCE})"); break

    # V3: restore the best VALIDATION checkpoint, then evaluate the untouched test set ONCE.
    model.load_state_dict(best_weights)
    print(f"\n[Restore] Best validation checkpoint (score={best_score:.4f}) restored before test eval.")
    te_loss, te_acc, te_cm = evaluate(model, test_loader, criterion, num_classes)
    te_metrics = metrics_from_confusion_matrix(te_cm, class_names)

    print(f"\n[Test] Acc={te_acc:.4f} | MacroRecall={te_metrics['macro_recall']:.4f} | "
          f"MacroPrecision={te_metrics['macro_precision']:.4f} | MacroF1={te_metrics['macro_f1']:.4f}")
    print(f"[Test] Melanoma recall : {te_metrics['per_class_recall'][class_names[melanoma_idx]]:.4f}")
    print(f"[Test] Per-class recall   : {te_metrics['per_class_recall']}")
    print(f"[Test] Per-class precision: {te_metrics['per_class_precision']}")
    print(f"[Test] Per-class F1       : {te_metrics['per_class_f1']}")
    print(f"[Test] Confusion matrix (rows=true, cols=pred):\n{te_cm.tolist()}")

    torch.save(best_weights, MODEL_PATH)
    print(f"[Save] Best model weights -> {MODEL_PATH}")

    json.dump({
        "model_type": "resnet50", "task": "skin_cancer", "class_names": class_names,
        "num_classes": num_classes, "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
        "trainer_config": {
            "version": "V3",
            "head_epochs": HEAD_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS,
            "batch_size": BATCH_SIZE, "patience": PATIENCE,
            "lr_head_phase1": LR_HEAD,
            "lr_finetune_head": LR_FINETUNE_HEAD,
            "lr_finetune_backbone": LR_FINETUNE_BACKBONE,
            "melanoma_weight_multiplier": MELANOMA_WEIGHT_MULTIPLIER,
            "malignant_weight_multiplier": MALIGNANT_WEIGHT_MULTIPLIER,
            "minority_sampler_power": MINORITY_SAMPLER_POWER,
            "checkpoint_score_weights": {
                "accuracy": VAL_ACC_WEIGHT,
                "macro_recall": VAL_MACRO_RECALL_WEIGHT,
                "melanoma_recall": MELANOMA_RECALL_WEIGHT,
            },
        },
        "melanoma_class": class_names[melanoma_idx],
        "best_val_score": round(best_score, 4),
        "best_val_acc": round(best_acc, 4),
        "best_val_macro_recall": round(best_macro_recall, 4),
        "best_val_melanoma_recall": round(best_mel_recall, 4),
        "test_acc": round(te_acc, 4),
        "test_macro_recall": round(te_metrics["macro_recall"], 4),
        "test_macro_precision": round(te_metrics["macro_precision"], 4),
        "test_macro_f1": round(te_metrics["macro_f1"], 4),
        "test_melanoma_recall": round(te_metrics["per_class_recall"][class_names[melanoma_idx]], 4),
        "test_per_class_recall": te_metrics["per_class_recall"],
        "test_per_class_precision": te_metrics["per_class_precision"],
        "test_per_class_f1": te_metrics["per_class_f1"],
        "test_confusion_matrix": te_cm.tolist(),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.items()},
    }, open(META_PATH, "w"), indent=2)
    print(f"[Save] Metadata -> {META_PATH}")

    print(f"[Save] {MODEL_PATH}\n[Skin CNN] Done! \U0001F389 "
          f"Best Val Acc: {best_acc:.4f} | Best Val MacroRecall: {best_macro_recall:.4f} | "
          f"Best Val MelRecall: {best_mel_recall:.4f} | "
          f"Test Acc: {te_acc:.4f} | Test MacroRecall: {te_metrics['macro_recall']:.4f}")

if __name__ == "__main__":
    main()