#!/usr/bin/env python3
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
detail_files = sorted(result_dir.glob("*_detail.json"))

summary = {}
for f in detail_files:
    data = json.load(f.open())
    correct = sum(1 for item in data if item.get("correct", False))
    acc = correct / len(data) if data else 0.0

    condition = f.stem.replace("_detail", "")
    summary[condition] = acc
    print(f"{condition}: {correct}/{len(data)} = {acc:.4f}")

json.dump(summary, (result_dir / "summary.json").open("w"), indent=2)
print(f"\nSummary saved to {result_dir / 'summary.json'}")
