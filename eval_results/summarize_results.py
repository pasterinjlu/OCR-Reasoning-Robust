#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

results_dir = Path(__file__).parent
models = [
    "InternVL3.5_8B", "InternVL3_5-14B", "InternVL3_5-4B",
    "MM-Eureka-Qwen-7B", "Qwen3_VL_4B_Instruct", "Qwen3_VL_4B_Thinking",
    "Qwen3_VL_8B", "Qwen3_VL_8B_Thinking", "Qwen_Qwen3_VL_32B_Instruct",
    "VL-Rethinker-7B", "gemini_3_flash_preview_nothinking",
    "gemini_3_flash_preview_thinking", "gpt_5.1_all", "gpt_5.2",
    "llama_4_scout_17b_16e_instruct"
]

datasets = ["ocr1.0", "ocr2.0"]
perturbations = ["baseline", "pixelate_1.5", "pixelate_2", "pixelate_2.5",
                 "glass_blur_2", "glass_blur_2.5", "glass_blur_3",
                 "color_shift_3", "color_shift_4", "color_shift_5",
                 "elastic_10", "elastic_15", "elastic_20",
                 "motion_blur_5", "motion_blur_6", "motion_blur_7",
                 "snow_0.1", "snow_0.2", "snow_0.3"]

all_data = []
for dataset in datasets:
    for model in models:
        result_file = results_dir / f"{dataset}_{model}.json"
        if not result_file.exists():
            continue

        with open(result_file) as f:
            data = json.load(f)

        row = {"Dataset": dataset, "Model": model}
        for pert in perturbations:
            if pert in data:
                row[pert] = f"{data[pert]['accuracy']:.2f}"
        all_data.append(row)

df = pd.DataFrame(all_data)
output_csv = results_dir / "summary.csv"
df.to_csv(output_csv, index=False)
print(f"Summary saved to {output_csv}")
print(f"\nTotal models: {len(models)}")
print(f"Models with results: {len(df['Model'].unique())}")
