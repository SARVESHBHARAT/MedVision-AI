"""Image validation and preprocessing for the MedVision AI baseline pipeline."""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageOps

TARGET_SIZE = (224, 224)
SUPPORTED = {".jpg", ".jpeg", ".png"}

@dataclass
class QualityReport:
    width: int
    height: int
    brightness: float
    contrast: float
    accepted: bool
    reason: str

def preprocess(path: str | Path) -> tuple[np.ndarray, QualityReport]:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError("Supported formats are PNG, JPG, and JPEG")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    if min(width, height) < 64:
        report = QualityReport(width, height, 0, 0, False, "Image is smaller than 64 pixels on one side")
        return np.empty((0,)), report
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    brightness, contrast = float(pixels.mean()), float(pixels.std())
    if contrast < 0.03:
        report = QualityReport(width, height, brightness, contrast, False, "Image contrast is too low")
        return np.empty((0,)), report
    resized = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = (tensor - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    report = QualityReport(width, height, brightness, contrast, True, "Passed basic quality checks")
    return np.transpose(tensor, (2, 0, 1)), report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a PNG or JPEG image")
    args = parser.parse_args()
    tensor, report = preprocess(args.image)
    print(json.dumps({"quality": asdict(report), "tensor_shape": list(tensor.shape)}))
