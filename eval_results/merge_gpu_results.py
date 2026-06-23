#!/usr/bin/env python3
import json
from pathlib import Path
import shutil

eval_dir = Path(__file__).resolve().parent
old_dir = eval_dir / "old"

# Models that need merging (file size = 2 bytes)
incomplete_models = [
    ("ocr1.0", "InternVL3_5-14B", "InternVL3.5_14B"),
    ("ocr1.0", "InternVL3_5-4B", "InternVL3.5_4B"),
    ("ocr1.0", "MM-Eureka-Qwen-7B", "MM_Eureka_Qwen_7B"),
    ("ocr1.0", "Qwen3_VL_4B_Thinking", "Qwen3_VL_4B_Thinking"),
    ("ocr1.0", "VL-Rethinker-7B", "VL_Rethinker_7B"),
    ("ocr2.0", "InternVL3_5-14B", "InternVL3.5_14B"),
    ("ocr2.0", "InternVL3_5-4B", "InternVL3.5_4B"),
    ("ocr2.0", "MM-Eureka-Qwen-7B", "MM_Eureka_Qwen_7B"),
    ("ocr2.0", "VL-Rethinker-7B", "VL_Rethinker_7B"),
]

def merge_results(dataset, model_name, old_model_name):
    gpu_files = sorted(old_dir.glob(f"{dataset}_{old_model_name}_gpu*.json"))
    if not gpu_files:
        print(f"No GPU files found for {dataset}_{model_name}")
        return

    merged = {}
    for gpu_file in gpu_files:
        with open(gpu_file) as f:
            data = json.load(f)
        for pert, stats in data.items():
            if pert not in merged:
                merged[pert] = {"correct": 0, "total": 0}
            merged[pert]["correct"] += stats["correct"]
            merged[pert]["total"] += stats["total"]

    for pert in merged:
        merged[pert]["accuracy"] = (merged[pert]["correct"] / merged[pert]["total"]) * 100

    output_file = eval_dir / f"{dataset}_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"✓ Merged {dataset}_{model_name}.json")

    # Merge details directories
    details_dir = eval_dir / f"{dataset}_{model_name}_details"
    details_dir.mkdir(exist_ok=True)

    gpu_detail_dirs = sorted(old_dir.glob(f"{dataset}_{old_model_name}_details_gpu*"))
    for gpu_dir in gpu_detail_dirs:
        for json_file in gpu_dir.glob("*.json"):
            dest = details_dir / json_file.name
            if not dest.exists():
                shutil.copy2(json_file, dest)
    print(f"✓ Merged {dataset}_{model_name}_details/")

for dataset, model_name, old_model_name in incomplete_models:
    merge_results(dataset, model_name, old_model_name)
