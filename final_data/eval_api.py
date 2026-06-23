"""
Evaluate VLM on baseline + perturbations using API
Usage:
  python eval_api.py --dataset ocr1.0 --model Qwen/Qwen3-VL-8B-Instruct
"""
import os, sys, json, argparse, base64
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from scoring.matcher import extract_gt_answer
from scoring import score_answer

API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

SCRIPT_DIR = Path(__file__).parent

PERTURBATIONS = {
    # "pixelate": [1.5, 2, 2.5],
    # "glass_blur": [2, 2.5, 3],
    # "color_shift": [3, 4, 5],
    # "elastic": [10, 15, 20],
    "motion_blur": [7],
    "snow": [0.1, 0.2, 0.3],
}

def encode_image(path, max_size=1024):
    from PIL import Image
    import io
    img = Image.open(path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def inference_single(client, model, img_path, question, fmt=""):
    b64 = encode_image(img_path)
    mime = "image/jpeg"

    # Build prompt with format hint
    if fmt:
        prompt = f"{question}\n{fmt}\nDirectly output the answer only, without any explanation."
    else:
        prompt = f"{question}\nDirectly output the answer only, without any explanation."

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=256,
        temperature=0,
    )
    return str(resp.choices[0].message.content).strip()

def process_item(args):
    client, model, item, img_dir, use_judge = args
    img_name = Path(item["image_path"]).name
    img_path = img_dir / img_name
    if not img_path.exists():
        return None
    try:
        fmt = item.get("format", "")
        pred = inference_single(client, model, img_path, item["question"], fmt)
        gt = item["answers"][0] if item["answers"] else ""
        result = score_answer(gt, pred, item["question"], use_judge=use_judge)
        correct = bool(result) if result is not None else None
        return {"question": item["question"], "gt": gt, "pred": pred, "correct": correct}
    except Exception as e:
        print(f"  Error {img_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ocr1.0", "ocr2.0"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running API-based evaluation.")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    model_name = args.model.replace("/", "_").replace("-", "_")

    data_json = SCRIPT_DIR / args.dataset / "dataset.json"
    with open(data_json) as f:
        data = json.load(f)

    output_dir = SCRIPT_DIR.parent / "eval_results"
    output_dir.mkdir(exist_ok=True)
    details_dir = output_dir / f"{args.dataset}_{model_name}_details"
    details_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{args.dataset}_{model_name}.json"

    print(f"Dataset: {args.dataset} ({len(data)} samples)")
    print(f"Model: {args.model}, Concurrency: {args.concurrency}")
    use_judge = not args.no_judge
    all_results = {}

    # Baseline
    print("Baseline...")
    baseline_dir = SCRIPT_DIR / args.dataset / "images"
    task_args = [(client, args.model, item, baseline_dir, use_judge) for item in data]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        details = list(tqdm(pool.map(process_item, task_args), total=len(data), desc="  baseline"))
    details = [d for d in details if d is not None]
    c = sum(1 for d in details if d["correct"])
    t = len(details)
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
            task_args = [(client, args.model, item, img_dir, use_judge) for item in data]
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                details = list(tqdm(pool.map(process_item, task_args), total=len(data), desc=f"  {cond}"))
            details = [d for d in details if d is not None]
            c = sum(1 for d in details if d["correct"])
            t = len(details)
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
