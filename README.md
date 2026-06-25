
# OCR Reasoning Robustness Benchmark

This repository contains evaluation code and aggregate results for benchmarking vision-language models on OCR reasoning tasks under visual perturbations.

The benchmark evaluates whether models can answer text-centric visual questions when the input image is degraded by common perturbations such as pixelation, blur, chromatic shift, elastic distortion, motion blur, and snow.


The benchmark data can be downloaded from Baidu Netdisk: [Baidu Netdisk Download Link](https://pan.baidu.com/s/1XoXxHioo55X0mOjCpO2baA?pwd=u2hu)

## Repository Structure

```text
.
├── final_data/
│   ├── ocr1.0/                  # Put ocr1.0 dataset.json and images/ here
│   ├── ocr2.0/                  # Put ocr2.0 dataset.json and images/ here
│   ├── scoring/                 # Rule-based and LLM-as-judge answer scoring
│   ├── generate_perturbations.py
│   ├── eval_api.py              # OpenAI-compatible API evaluation
│   ├── eval_vllm.py             # Local vLLM evaluation
│   ├── eval_vllm_cot.py         # Local vLLM evaluation with CoT prompting
│   ├── eval_no_image.py         # Image-free contamination/control baseline
│   └── run_*.sh                 # Example launch scripts
├── eval_results/                # Aggregate result JSON/CSV files
├── docs/
│   ├── perturbation_catalog.md
│   └── perturbations_figure.png
├── .env.example
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For API-based models, set:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_BASE="https://api.openai.com/v1"
```

`OPENAI_API_BASE` may point to any OpenAI-compatible endpoint.

## Data Layout

Place each dataset under `final_data/`:

```text
final_data/
├── ocr1.0/
│   ├── dataset.json
│   └── images/
└── ocr2.0/
    ├── dataset.json
    └── images/
```

Each item in `dataset.json` should follow:

```json
{
  "image_id": 0,
  "image_path": "images/0.png",
  "question": "What is the total amount on the receipt?",
  "answers": ["$24.50"],
  "format": "Answer in English.",
  "language": "English",
  "type": "receipt"
}
```

## Generate Perturbations

```bash
cd final_data
python generate_perturbations.py
```

This creates:

```text
final_data/perturbations/<perturbation>_<level>/<dataset>/
```

The default perturbation suite is:

| Type | Levels |
| --- | --- |
| pixelate | 1.5, 2, 2.5 |
| glass_blur | 2, 2.5, 3 |
| color_shift | 3, 4, 5 |
| elastic | 10, 15, 20 |
| motion_blur | 5, 6, 7 |
| snow | 0.1, 0.2, 0.3 |

## Evaluate API Models

```bash
cd final_data
python eval_api.py --dataset ocr1.0 --model gpt-4o
```

## Evaluate Local Models with vLLM

```bash
cd final_data
CUDA_VISIBLE_DEVICES=0 python eval_vllm.py \
  --dataset ocr1.0 \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --model-name Qwen3-VL-8B
```


## Results

New evaluations are written to `eval_results/`:

- `<dataset>_<model>.json`: aggregate accuracy for each condition.
- `<dataset>_<model>_details/`: per-sample predictions.
- `summary.csv`: tabular summary across models.

The published `eval_results/` directory contains lightweight aggregate files only.

## Notes

- Do not commit private API keys. Use environment variables or a local `.env` file.
- Large generated folders such as `final_data/perturbations/`, `eval_results/*_details/`, and raw dataset images are ignored by default.
- See `docs/perturbation_catalog.md` for perturbation implementation details.
