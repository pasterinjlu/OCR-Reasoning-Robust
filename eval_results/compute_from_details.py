#!/usr/bin/env python3
import json
from pathlib import Path

eval_dir = Path(__file__).resolve().parent

def compute_from_details(dataset, model_name):
    details_dir = eval_dir / f"{dataset}_{model_name}_details"
    if not details_dir.exists():
        print(f"Details dir not found: {details_dir}")
        return

    results = {}
    for json_file in sorted(details_dir.glob("*.json")):
        pert_name = json_file.stem
        with open(json_file) as f:
            data = json.load(f)

        correct = sum(1 for item in data if item.get("correct", False))
        total = len(data)
        accuracy = (correct / total * 100) if total > 0 else 0

        results[pert_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

    output_file = eval_dir / f"{dataset}_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Generated {dataset}_{model_name}.json")

compute_from_details("ocr1.0", "Qwen3_VL_8B")
compute_from_details("ocr2.0", "Qwen3_VL_8B")
