#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for MODEL in "deepseek-v3.2" "gemini-3-pro-preview-11-2025"; do
    echo "=============================="
    echo "Model: $MODEL"
    echo "=============================="
    python "$SCRIPT_DIR/eval_no_image.py" --model "$MODEL" --dataset all
done

echo "All done."
