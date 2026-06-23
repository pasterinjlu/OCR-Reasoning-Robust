"""
Merge results from multi-GPU evaluation
Usage: python merge_results.py --dataset ocr1.0 --model Qwen3_VL_4B_Instruct --num-gpus 3
"""
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen3_VL_4B_Instruct)")
    parser.add_argument("--num-gpus", type=int, required=True)
    args = parser.parse_args()

    MODEL_NAME = args.model

    output_dir = SCRIPT_DIR.parent / "eval_results"
    merged_results = {}

    # Merge summary JSONs
    for gpu_id in range(args.num_gpus):
        result_file = output_dir / f"{args.dataset}_{MODEL_NAME}_gpu{gpu_id}.json"
        if result_file.exists():
            with open(result_file) as f:
                gpu_results = json.load(f)
                if not merged_results:
                    merged_results = {k: {"correct": 0, "total": 0} for k in gpu_results}
                for cond, metrics in gpu_results.items():
                    merged_results[cond]["correct"] += metrics["correct"]
                    merged_results[cond]["total"] += metrics["total"]

    # Calculate accuracy
    for cond in merged_results:
        c, t = merged_results[cond]["correct"], merged_results[cond]["total"]
        merged_results[cond]["accuracy"] = c / t * 100 if t > 0 else 0

    # Save merged summary
    output_file = output_dir / f"{args.dataset}_{MODEL_NAME}.json"
    with open(output_file, "w") as f:
        json.dump(merged_results, f, indent=2)
    print(f"Merged summary saved to {output_file}")

    # Merge detail JSONs
    details_dir = output_dir / f"{args.dataset}_{MODEL_NAME}_details"
    details_dir.mkdir(exist_ok=True)

    # Get all condition names from first GPU
    first_gpu_dir = output_dir / f"{args.dataset}_{MODEL_NAME}_details_gpu0"
    if first_gpu_dir.exists():
        for detail_file in first_gpu_dir.glob("*.json"):
            cond_name = detail_file.name
            merged_details = []
            for gpu_id in range(args.num_gpus):
                gpu_detail_file = output_dir / f"{args.dataset}_{MODEL_NAME}_details_gpu{gpu_id}" / cond_name
                if gpu_detail_file.exists():
                    with open(gpu_detail_file) as f:
                        merged_details.extend(json.load(f))

            with open(details_dir / cond_name, "w") as f:
                json.dump(merged_details, f, indent=2, ensure_ascii=False)
        print(f"Merged details saved to {details_dir}")

if __name__ == "__main__":
    main()
