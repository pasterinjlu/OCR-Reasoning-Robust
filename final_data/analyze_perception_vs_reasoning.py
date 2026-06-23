#!/usr/bin/env python3
"""
Analyze perception vs reasoning failures by comparing Real OCR vs Oracle OCR
Usage:
  python analyze_perception_vs_reasoning.py --dataset ocr1.0 --llm-model qwen3-8b
"""
import json, argparse, csv
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent

def load_results(pipeline, dataset):
    """Load summary and details for a pipeline"""
    result_dir = SCRIPT_DIR / "results" / pipeline / dataset
    summary_file = result_dir / "summary.json"

    if not summary_file.exists():
        return None, {}

    summary = json.load(summary_file.open())
    details = {}

    for condition in summary.keys():
        detail_file = result_dir / f"{condition}_detail.json"
        if detail_file.exists():
            details[condition] = json.load(detail_file.open())

    return summary, details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ocr1.0", "ocr2.0"])
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--output", default="perception_vs_reasoning_analysis.csv")
    args = parser.parse_args()

    real_pipeline = f"paddleocr+{args.llm_model}"
    oracle_pipeline = f"oracle+{args.llm_model}"

    real_summary, real_details = load_results(real_pipeline, args.dataset)
    oracle_summary, oracle_details = load_results(oracle_pipeline, args.dataset)

    if not real_summary or not oracle_summary:
        print("Error: Missing results. Run evaluations first.")
        return

    print(f"\n{'='*80}")
    print(f"Perception vs Reasoning Analysis: {args.dataset} + {args.llm_model}")
    print(f"{'='*80}\n")

    # Accuracy comparison table
    print(f"{'Condition':<20} {'Real OCR':>10} {'Oracle OCR':>12} {'Δ (Gain)':>12}")
    print("-" * 80)

    analysis_data = []
    for condition in sorted(real_summary.keys()):
        real_acc = real_summary.get(condition, 0)
        oracle_acc = oracle_summary.get(condition, 0)
        gain = oracle_acc - real_acc

        print(f"{condition:<20} {real_acc:>9.1%} {oracle_acc:>11.1%} {gain:>11.1%}")

        # Per-sample failure mode analysis
        if condition in real_details and condition in oracle_details:
            real_items = {item["image_id"]: item for item in real_details[condition]}
            oracle_items = {item["image_id"]: item for item in oracle_details[condition]}

            perception_failures = 0
            reasoning_failures = 0
            both_correct = 0

            for img_id in real_items:
                if img_id not in oracle_items:
                    continue

                real_correct = real_items[img_id].get("correct", False)
                oracle_correct = oracle_items[img_id].get("correct", False)

                if not real_correct and oracle_correct:
                    perception_failures += 1
                elif not real_correct and not oracle_correct:
                    reasoning_failures += 1
                elif real_correct and oracle_correct:
                    both_correct += 1

            total = len(real_items)
            analysis_data.append({
                "condition": condition,
                "real_acc": real_acc,
                "oracle_acc": oracle_acc,
                "gain": gain,
                "perception_failures": perception_failures,
                "reasoning_failures": reasoning_failures,
                "both_correct": both_correct,
                "total": total
            })

    print("\n" + "="*80)
    print("Failure Mode Breakdown")
    print("="*80 + "\n")
    print(f"{'Condition':<20} {'Perception':>12} {'Reasoning':>12} {'Both OK':>12}")
    print("-" * 80)

    for row in analysis_data:
        print(f"{row['condition']:<20} {row['perception_failures']:>12} "
              f"{row['reasoning_failures']:>12} {row['both_correct']:>12}")

    # Save to CSV
    output_file = SCRIPT_DIR / args.output
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "real_acc", "oracle_acc", "gain",
            "perception_failures", "reasoning_failures", "both_correct", "total"
        ])
        writer.writeheader()
        writer.writerows(analysis_data)

    print(f"\n{'='*80}")
    print(f"Analysis saved to: {output_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
