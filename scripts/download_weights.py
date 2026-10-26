"""
MedVision AI — Model Weights Verification and Setup Utility

This script checks for the presence and integrity of pre-trained model weights
required by the MedVision AI inference engine.

Required Checkpoints:
  - cnn/brain_resnet50_v2.pth   (94.50 MB)
  - cnn/chest_xray_resnet50.pth (94.50 MB)
  - cnn/eye_resnet50_v3.pth     (94.50 MB)
  - cnn/skin_resnet50.pth       (94.50 MB)
  - dqn_model.pth               (0.51 MB)
"""

import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CNN_DIR = BASE_DIR / "cnn"

RELEASE_TAG = "v1.0.0"
BASE_RELEASE_URL = f"https://github.com/SARVESHBHARAT/MedVision-AI/releases/download/{RELEASE_TAG}"

MODELS = {
    "brain_resnet50_v2.pth": {
        "dest": CNN_DIR / "brain_resnet50_v2.pth",
        "size_mb": 94.50,
        "trainer": "cnn/brain_trainer_v2.py",
        "description": "Brain MRI 4-class classifier (Glioma, Meningioma, No Tumor, Pituitary)",
    },
    "chest_xray_resnet50.pth": {
        "dest": CNN_DIR / "chest_xray_resnet50.pth",
        "size_mb": 94.50,
        "trainer": "cnn/chest_xray_trainer.py",
        "description": "Chest Radiograph Pneumonia vs. Normal classifier",
    },
    "eye_resnet50_v3.pth": {
        "dest": CNN_DIR / "eye_resnet50_v3.pth",
        "size_mb": 94.50,
        "trainer": "cnn/eye_trainer_v5.py",
        "description": "Diabetic Retinopathy 5-stage fundus classifier (384px)",
    },
    "skin_resnet50.pth": {
        "dest": CNN_DIR / "skin_resnet50.pth",
        "size_mb": 94.50,
        "trainer": "cnn/skin_trainer.py",
        "description": "ISIC dermoscopy skin lesion multi-class classifier",
    },
    "dqn_model.pth": {
        "dest": BASE_DIR / "dqn_model.pth",
        "size_mb": 0.51,
        "trainer": "train_dqn.py",
        "description": "Deep Q-Network reinforcement learning symptom triage policy",
    },
}


def check_status() -> dict:
    """Inspect local directory for expected model checkpoints."""
    print("=" * 65)
    print(" MedVision AI — Model Weights Status Verification")
    print("=" * 65)
    CNN_DIR.mkdir(parents=True, exist_ok=True)

    results = {"found": 0, "missing": 0}

    for name, info in MODELS.items():
        dest = info["dest"]
        rel_path = dest.relative_to(BASE_DIR)

        if dest.exists():
            actual_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [FOUND]   {rel_path:<30} ({actual_mb:6.2f} MB)")
            results["found"] += 1
        else:
            print(f"  [MISSING] {rel_path:<30} (expected ~{info['size_mb']:5.2f} MB)")
            print(f"            Description: {info['description']}")
            print(f"            Can train locally via: python {info['trainer']}")
            results["missing"] += 1

    print("-" * 65)
    print(f"Summary: {results['found']} found, {results['missing']} missing.")

    if results["missing"] > 0:
        print("\nNote: MedVision AI runs gracefully without model files:")
        print("  - The application backend and UI will start normally.")
        print("  - Untrained/missing modalities are skipped gracefully.")
        print("  - To obtain pre-trained weights:")
        print(f"    1. Download from GitHub Releases: {BASE_RELEASE_URL}/")
        print("    2. Or train models from scratch using the included trainer scripts.")
        print("    3. Or run: python scripts/download_weights.py --download")
    else:
        print("\nAll model weights are present and ready for clinical inference.")

    print("=" * 65)
    return results


def download_models():
    """Attempt downloading missing models from the published GitHub Release."""
    CNN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Attempting download from release tag: {RELEASE_TAG} ...\n")

    for name, info in MODELS.items():
        dest = info["dest"]
        if dest.exists():
            print(f"[SKIP] {name} already exists.")
            continue

        url = f"{BASE_RELEASE_URL}/{name}"
        print(f"[FETCH] Downloading {name} from:\n        {url}")

        try:
            # Stream download with simple progress
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100.0, block_num * block_size * 100 / total_size)
                    sys.stdout.write(f"\r        Progress: {percent:5.1f}%")
                    sys.stdout.flush()

            temp_dest = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(url, temp_dest, reporthook=report_progress)
            sys.stdout.write("\n")
            temp_dest.replace(dest)
            print(f"        Successfully saved to {dest.relative_to(BASE_DIR)}")

        except urllib.error.HTTPError as exc:
            sys.stdout.write("\n")
            print(f"        [NOTICE] HTTP {exc.code} {exc.reason}.")
            print("        Release assets may not be published yet.")
            print(f"        Please train locally with: python {info['trainer']}")
        except Exception as exc:
            sys.stdout.write("\n")
            print(f"        [ERROR] Download failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="MedVision AI Weights Verification")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt to download missing weights from GitHub Releases",
    )
    args = parser.parse_args()

    check_status()
    if args.download:
        download_models()
        check_status()


if __name__ == "__main__":
    main()
