"""Create a JSON quality summary for a folder of candidate images."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from preprocess import SUPPORTED, preprocess

def build_report(folder: Path) -> dict:
    records = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        try:
            tensor, quality = preprocess(path)
            records.append({"file": path.name, "quality": asdict(quality), "tensor_shape": list(tensor.shape)})
        except (OSError, ValueError) as error:
            records.append({"file": path.name, "error": str(error)})
    accepted = sum(item.get("quality", {}).get("accepted", False) for item in records)
    return {"total_images": len(records), "accepted_images": accepted, "rejected_images": len(records) - accepted, "images": records}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("quality_report.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_report(args.folder), indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
