"""
Image-Free Baseline for Contamination Control
Usage:
  python eval_no_image.py --dataset ocr1.0 --model gpt-5.4
  python eval_no_image.py --dataset all --model all
"""
import os, sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from scoring import score_answer
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent
API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
MODELS = ["deepseek-v3.2","gemini-3-pro-preview-11-2025"]


def call_api(client, model, question, fmt=""):
    prompt = f"{question}\n{fmt}\nDirectly output the answer only, without any explanation." if fmt \
        else f"{question}\nDirectly output the answer only, without any explanation."
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=256, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return str(resp.choices[0].message.content).strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return ""


def process_item(args):
    client, model, item, use_judge = args
    try:
        pred = call_api(client, model, item["question"], item.get("format", ""))
        gt = item["answers"][0] if item["answers"] else ""
        result = score_answer(gt, pred, item["question"], use_judge=use_judge)
        correct = bool(result) if result is not None else None
        return {"question": item["question"], "gt": gt, "pred": pred, "correct": correct}
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run(client, model, dataset, use_judge):
    model_name = model.replace("/", "_").replace("-", "_")
    data_json = SCRIPT_DIR / dataset / "dataset.json"
    with open(data_json) as f:
        data = json.load(f)

    out_dir = SCRIPT_DIR.parent / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{dataset}_{model_name}_no_image.json"
    if out_file.exists():
        print(f"  {dataset}/{model}: already done, skip")
        return

    print(f"\n[{model}] [{dataset}] {len(data)} samples (no image)")
    task_args = [(client, model, item, use_judge) for item in data]
    with ThreadPoolExecutor(max_workers=10) as pool:
        details = list(tqdm(pool.map(process_item, task_args), total=len(data)))
    details = [d for d in details if d]
    c = sum(1 for d in details if d["correct"])
    t = len(details)
    acc = c / t * 100 if t else 0
    print(f"  => {acc:.2f}% ({c}/{t})")

    with open(out_file, "w") as f:
        json.dump({"accuracy": acc, "correct": c, "total": t, "details": details},
                  f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["ocr1.0", "ocr2.0", "all"])
    parser.add_argument("--model", default="all")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running API-based evaluation.")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    datasets = ["ocr1.0", "ocr2.0"] if args.dataset == "all" else [args.dataset]
    models = MODELS if args.model == "all" else [args.model]

    for model in models:
        for ds in datasets:
            run(client, model, ds, not args.no_judge)
    print("\nAll done.")


if __name__ == "__main__":
    main()
