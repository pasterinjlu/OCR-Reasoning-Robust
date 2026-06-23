"""
Evaluate VLM on baseline + perturbations using vLLM
Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --dataset ocr1.0 --gpu 0
  CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --dataset ocr2.0 --gpu 0
"""
import os, sys, json, argparse, re
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from scoring.matcher import extract_gt_answer
from scoring import score_answer

SCRIPT_DIR = Path(__file__).parent

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_MODEL_NAME = "Qwen3-VL-8B"

PERTURBATIONS = {
    "pixelate": [1.5, 2, 2.5],
    "glass_blur": [2, 2.5, 3],
    "color_shift": [3, 4, 5],
    "elastic": [10, 15, 20],
    "motion_blur": [5, 6, 7],
    "snow": [0.1, 0.2, 0.3],
}

_llm = None

def get_llm(model_path, model_name, gpu_id=0, tensor_parallel_size=1):
    global _llm
    if _llm is None:
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        from vllm import LLM
        print(f"Loading {model_name}...")
        _llm = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=16384,
            gpu_memory_utilization=0.85,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
            allowed_local_media_path=str(SCRIPT_DIR.parent),
            enforce_eager=True,
            disable_custom_all_reduce=True,
            tensor_parallel_size=tensor_parallel_size,
        )
    return _llm

def batch_inference(llm, data, img_dir, cot=False):
    from vllm import SamplingParams
    from PIL import Image
    params = SamplingParams(max_tokens=16384, temperature=0.6, top_p=0.95)

    valid_idx, convs = [], []
    for i, item in enumerate(data):
        img_name = Path(item["image_path"]).name
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        # Validate image file
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"Skipping corrupted image: {img_path} - {e}")
            continue

        # Build prompt with format hint
        fmt = item.get("format", "")
        if cot:
            if fmt:
                prompt = f"{item['question']}\n{fmt}\nPlease think step by step, then output 'The final answer is: <your answer>'."
            else:
                prompt = f"{item['question']}\nPlease think step by step, then output 'The final answer is: <your answer>'."
        else:
            if fmt:
                prompt = f"{item['question']}\n{fmt}\nDirectly output the answer only, without any explanation."
            else:
                prompt = f"{item['question']}\nDirectly output the answer only, without any explanation."

        convs.append([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"file://{img_path}"}},
            {"type": "text", "text": prompt},
        ]}])
        valid_idx.append(i)

    outputs = llm.chat(convs, sampling_params=params)
    pred_map = {}
    for idx, out in zip(valid_idx, outputs):
        text = out.outputs[0].text.strip()
        # Strip thinking content (handles both <think>...</think> and bare ...&lt;/think>)
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        # For CoT: extract final answer after common markers
        if cot:
            for marker in ["The final answer is:", "the final answer is:", "Final answer:", "final answer:", "The answer is", "the answer is", "Answer:"]:
                if marker in text:
                    text = text.split(marker)[-1].strip().strip(".")
                    break
        pred_map[idx] = text
    return [(item, pred_map.get(i)) for i, item in enumerate(data)]

def score_batch(batch_results, use_judge=True):
    correct, total, details = 0, 0, []
    for item, pred in batch_results:
        if pred is None:
            continue
        gt = item["answers"][0] if item["answers"] else ""
        result = score_answer(gt, pred, item["question"], use_judge=use_judge)
        if result is not None:
            if result:
                correct += 1
            total += 1
        details.append({"question": item["question"], "gt": gt, "pred": pred, "correct": bool(result) if result is not None else None})
    return correct, total, details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ocr1.0", "ocr2.0"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--gpu-id", type=int, default=0, help="Current GPU ID (0 to num-gpus-1)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to model")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Model display name")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    model_name = args.model_name
    cot = args.cot
    cot_tag = "_cot" if cot else ""

    data_json = SCRIPT_DIR / args.dataset / "dataset.json"
    with open(data_json) as f:
        all_data = json.load(f)

    # Data sharding for multi-GPU
    data = [item for i, item in enumerate(all_data) if i % args.num_gpus == args.gpu_id]

    output_dir = SCRIPT_DIR.parent / "eval_results"
    output_dir.mkdir(exist_ok=True)
    suffix = f"_gpu{args.gpu_id}" if args.num_gpus > 1 else ""
    details_dir = output_dir / f"{args.dataset}_{model_name.replace('-', '_')}{cot_tag}_details{suffix}"
    details_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{args.dataset}_{model_name.replace('-', '_')}{cot_tag}{suffix}.json"

    print(f"GPU {args.gpu_id}/{args.num_gpus}: Dataset {args.dataset} ({len(data)}/{len(all_data)} samples)")
    llm = get_llm(args.model_path, model_name, args.gpu, args.tensor_parallel_size)
    use_judge = not args.no_judge
    all_results = {}

    # Baseline
    print("Baseline...")
    baseline_dir = SCRIPT_DIR / args.dataset / "images"
    batch = batch_inference(llm, data, baseline_dir, cot=cot)
    c, t, details = score_batch(batch, use_judge)
    acc = c / t * 100 if t > 0 else 0
    all_results["baseline"] = {"accuracy": acc, "correct": c, "total": t}
    with open(details_dir / "baseline.json", "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"  {acc:.2f}% ({c}/{t})")

    # Perturbations
    for pname, levels in PERTURBATIONS.items():
        for level in levels:
            cond = f"{pname}_{level}"
            print(f"{cond}...")
            img_dir = SCRIPT_DIR.parent / "perturbations" / cond / args.dataset
            if not img_dir.exists():
                print(f"  Skipped (dir not found)")
                continue
            batch = batch_inference(llm, data, img_dir, cot=cot)
            c, t, details = score_batch(batch, use_judge)
            acc = c / t * 100 if t > 0 else 0
            all_results[cond] = {"accuracy": acc, "correct": c, "total": t}
            with open(details_dir / f"{cond}.json", "w") as f:
                json.dump(details, f, indent=2, ensure_ascii=False)
            print(f"  {acc:.2f}% ({c}/{t})")

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()
