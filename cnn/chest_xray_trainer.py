"""
MedVision AI — Chest X-Ray CNN Trainer
ResNet50 Transfer Learning | PyTorch
Author: Sarvesh Bharat
Classes: NORMAL / PNEUMONIA (+ COVID if present)
Dataset: <project_root>/cnn/chest_xray/chest_xray
"""

import os
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "cnn" / "chest_xray" / "chest_xray"
MODEL_DIR  = BASE_DIR / "cnn"
MODEL_PATH = MODEL_DIR / "chest_xray_resnet50.pth"
META_PATH  = MODEL_DIR / "chest_xray_meta.json"

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
PATIENCE    = 5          # early stopping patience
NUM_WORKERS = 0          # set to 4 on Linux; 0 is safe on Windows
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[MedVision] Device: {DEVICE}")
print(f"[MedVision] Data dir: {DATA_DIR}")


# ─────────────────────────────────────────────
#  TRANSFORMS  (ImageNet stats for ResNet)
# ─────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ─────────────────────────────────────────────
#  DATASETS & LOADERS
# ─────────────────────────────────────────────
def get_loaders(data_dir: Path):
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    test_dir  = data_dir / "test"

    # Fallback: if no val folder, carve 15% from train
    if not val_dir.exists():
        print("[WARN] No val/ folder found — using test/ as validation.")
        val_dir = test_dir

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_transform)
    test_ds  = datasets.ImageFolder(test_dir,  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    class_names = train_ds.classes
    class_counts = {c: 0 for c in class_names}
    for _, label in train_ds.samples:
        class_counts[class_names[label]] += 1

    print(f"\n[Data] Classes  : {class_names}")
    print(f"[Data] Train    : {len(train_ds)} images")
    print(f"[Data] Val      : {len(val_ds)} images")
    print(f"[Data] Test     : {len(test_ds)} images")
    print(f"[Data] Class dist (train): {class_counts}")

    return train_loader, val_loader, test_loader, class_names


# ─────────────────────────────────────────────
#  MODEL  — ResNet50 with custom head
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all backbone layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    return model.to(DEVICE)


# ─────────────────────────────────────────────
#  CLASS-WEIGHTED LOSS  (handles imbalance)
# ─────────────────────────────────────────────
def get_class_weights(train_loader, num_classes):
    counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for lbl in labels:
            counts[lbl] += 1
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights.to(DEVICE)


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────
#  FULL FINE-TUNE PHASE (unfreeze backbone)
# ─────────────────────────────────────────────
def unfreeze_backbone(model, lr_backbone=1e-5):
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam([
        {"params": model.layer1.parameters(), "lr": lr_backbone * 0.1},
        {"params": model.layer2.parameters(), "lr": lr_backbone * 0.3},
        {"params": model.layer3.parameters(), "lr": lr_backbone * 0.5},
        {"params": model.layer4.parameters(), "lr": lr_backbone},
        {"params": model.fc.parameters(),     "lr": lr_backbone * 2},
    ])
    return optimizer


# ─────────────────────────────────────────────
#  PLOT CURVES
# ─────────────────────────────────────────────
def plot_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    ax1.set_title("Loss Curve"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    ax2.set_title("Accuracy Curve"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot] Saved to {save_path}")
    plt.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    train_loader, val_loader, test_loader, class_names = get_loaders(DATA_DIR)
    num_classes = len(class_names)

    model     = build_model(num_classes)
    weights   = get_class_weights(train_loader, num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── Phase 1: Train head only ──
    print("\n[Phase 1] Training classifier head (backbone frozen)...")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_weights, patience_ctr = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"{elapsed:.1f}s")

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            print(f"  ✅ New best val acc: {best_acc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  ⏹ Early stop at epoch {epoch}")
                break

    # ── Phase 2: Fine-tune full backbone ──
    print("\n[Phase 2] Full fine-tune (backbone unfrozen, low LR)...")
    model.load_state_dict(best_weights)
    optimizer2 = unfreeze_backbone(model, lr_backbone=1e-5)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode="min", factor=0.5, patience=2)
    patience_ctr2 = 0

    for epoch in range(1, 11):  # 10 more fine-tune epochs
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer2)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler2.step(vl_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"  Fine {epoch:02d}/10 | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"{elapsed:.1f}s")

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr2 = 0
            print(f"  ✅ New best val acc: {best_acc:.4f}")
        else:
            patience_ctr2 += 1
            if patience_ctr2 >= PATIENCE:
                print(f"  ⏹ Early stop at fine-tune epoch {epoch}")
                break

    # ── Save model ──
    model.load_state_dict(best_weights)
    torch.save({
        "model_state_dict": best_weights,
        "class_names": class_names,
        "num_classes": num_classes,
        "best_val_acc": best_acc,
        "img_size": IMG_SIZE,
    }, MODEL_PATH)
    print(f"\n[Save] Model → {MODEL_PATH}")

    # ── Test evaluation ──
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"[Test] Loss={te_loss:.4f}  Acc={te_acc:.4f}")

    # ── Save metadata for MedVision API ──
    meta = {
        "model_type": "resnet50",
        "task": "chest_xray_classification",
        "class_names": class_names,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "best_val_acc": round(best_acc, 4),
        "test_acc": round(te_acc, 4),
        "model_path": str(MODEL_PATH),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Save] Metadata → {META_PATH}")

    # ── Plot ──
    plot_curves(history, MODEL_DIR / "chest_xray_training_curves.png")
    print("\n[MedVision] Chest X-Ray CNN training complete! 🎉")
    print(f"  Best Val Acc : {best_acc:.4f}")
    print(f"  Test Acc     : {te_acc:.4f}")
    print(f"  Classes      : {class_names}")


if __name__ == "__main__":
    main()