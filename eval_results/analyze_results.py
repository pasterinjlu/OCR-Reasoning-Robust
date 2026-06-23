#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_csv("summary.csv")

print("=" * 80)
print("OCR Reasoning Robustness Benchmark Analysis")
print("=" * 80)

# 1. Overall Performance Ranking
print("\n1. BASELINE PERFORMANCE RANKING")
print("-" * 80)
for dataset in ["ocr1.0", "ocr2.0"]:
    print(f"\n{dataset.upper()}:")
    subset = df[df["Dataset"] == dataset].copy()
    subset = subset.dropna(subset=["baseline"])
    subset = subset.sort_values("baseline", ascending=False)
    for idx, row in subset.iterrows():
        print(f"  {row['baseline']:5.2f}% - {row['Model']}")

# 2. Robustness Analysis (Average drop across perturbations)
print("\n\n2. ROBUSTNESS ANALYSIS (Avg accuracy drop from baseline)")
print("-" * 80)
pert_cols = [c for c in df.columns if c not in ["Dataset", "Model", "baseline"]]

for dataset in ["ocr1.0", "ocr2.0"]:
    print(f"\n{dataset.upper()}:")
    subset = df[df["Dataset"] == dataset].copy()

    for idx, row in subset.iterrows():
        if pd.isna(row["baseline"]):
            continue
        baseline = float(row["baseline"])
        pert_vals = [float(row[c]) for c in pert_cols if pd.notna(row[c])]
        if not pert_vals:
            continue
        avg_pert = np.mean(pert_vals)
        drop = baseline - avg_pert
        print(f"  {row['Model']:40s} | Baseline: {baseline:5.2f}% | Avg Pert: {avg_pert:5.2f}% | Drop: {drop:5.2f}%")

# 3. Most challenging perturbations
print("\n\n3. MOST CHALLENGING PERTURBATIONS (Largest accuracy drops)")
print("-" * 80)
for dataset in ["ocr1.0", "ocr2.0"]:
    print(f"\n{dataset.upper()}:")
    subset = df[df["Dataset"] == dataset].copy()

    pert_drops = {}
    for col in pert_cols:
        drops = []
        for idx, row in subset.iterrows():
            if pd.notna(row["baseline"]) and pd.notna(row[col]):
                drops.append(float(row["baseline"]) - float(row[col]))
        if drops:
            pert_drops[col] = np.mean(drops)

    sorted_perts = sorted(pert_drops.items(), key=lambda x: x[1], reverse=True)
    for pert, drop in sorted_perts[:5]:
        print(f"  {pert:20s}: {drop:5.2f}% avg drop")

print("\n" + "=" * 80)
