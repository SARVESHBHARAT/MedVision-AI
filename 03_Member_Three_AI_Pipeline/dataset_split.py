"""Create reproducible patient-level train/validation/test manifests."""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

def bucket(patient_id: str) -> str:
    value = int(hashlib.sha256(patient_id.encode()).hexdigest()[:8], 16) % 100
    return 'train' if value < 70 else 'validation' if value < 85 else 'test'

def split_manifest(source: Path, target: Path) -> dict[str, int]:
    groups = defaultdict(list)
    with source.open(newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            groups[row['patient_id']].append(row)
    output, counts = [], defaultdict(int)
    for patient_id, rows in groups.items():
        partition = bucket(patient_id)
        for row in rows:
            row['split'] = partition
            output.append(row)
            counts[partition] += 1
    if not output:
        raise ValueError('Manifest contains no rows')
    with target.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[*output[0].keys()])
        writer.writeheader(); writer.writerows(output)
    return dict(counts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(split_manifest(args.manifest, args.output))
