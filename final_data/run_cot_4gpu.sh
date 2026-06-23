#!/bin/bash
# Phase 1: InternVL3.5-8B ocr1.0 (GPU 0,1) + Qwen3-VL-4B ocr1.0 (GPU 2,3)
# Phase 2: InternVL3.5-8B ocr2.0 (GPU 0,1) + Qwen3-VL-4B ocr2.0 (GPU 2,3)
# Phase 3: Qwen3-VL-8B ocr2.0 CoT (GPU 0,1)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/eval_vllm_cot.py"

INTERNVL_PATH="${INTERNVL_PATH:-OpenGVLab/InternVL3_5-8B}"
QWEN4B_PATH="${QWEN4B_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
QWEN8B_PATH="${QWEN8B_PATH:-Qwen/Qwen3-VL-8B-Instruct}"

echo "===== Phase 1: ocr1.0 ====="

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
    --dataset ocr1.0 --cot --num-gpus 2 --gpu-id 0 \
    --model-path "$INTERNVL_PATH" --model-name InternVL3.5-8B &

CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
    --dataset ocr1.0 --cot --num-gpus 2 --gpu-id 1 \
    --model-path "$INTERNVL_PATH" --model-name InternVL3.5-8B &

CUDA_VISIBLE_DEVICES=2 python "$SCRIPT" \
    --dataset ocr1.0 --cot --num-gpus 2 --gpu-id 0 \
    --model-path "$QWEN4B_PATH" --model-name Qwen3-VL-4B &

CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" \
    --dataset ocr1.0 --cot --num-gpus 2 --gpu-id 1 \
    --model-path "$QWEN4B_PATH" --model-name Qwen3-VL-4B &

wait
echo "===== Phase 1 done ====="

echo "===== Phase 2: ocr2.0 ====="

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
    --dataset ocr2.0 --cot --num-gpus 2 --gpu-id 0 \
    --model-path "$INTERNVL_PATH" --model-name InternVL3.5-8B &

CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
    --dataset ocr2.0 --cot --num-gpus 2 --gpu-id 1 \
    --model-path "$INTERNVL_PATH" --model-name InternVL3.5-8B &

CUDA_VISIBLE_DEVICES=2 python "$SCRIPT" \
    --dataset ocr2.0 --cot --num-gpus 2 --gpu-id 0 \
    --model-path "$QWEN4B_PATH" --model-name Qwen3-VL-4B &

CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" \
    --dataset ocr2.0 --cot --num-gpus 2 --gpu-id 1 \
    --model-path "$QWEN4B_PATH" --model-name Qwen3-VL-4B &

wait
echo "===== Phase 2 done ====="

echo "===== Phase 3: Qwen3-VL-8B ocr2.0 CoT ====="

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
    --dataset ocr2.0 --cot --num-gpus 4 --gpu-id 0 \
    --model-path "$QWEN8B_PATH" --model-name Qwen3-VL-8B &

CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
    --dataset ocr2.0 --cot --num-gpus 4 --gpu-id 1 \
    --model-path "$QWEN8B_PATH" --model-name Qwen3-VL-8B &

CUDA_VISIBLE_DEVICES=2 python "$SCRIPT" \
    --dataset ocr2.0 --cot --num-gpus 4 --gpu-id 2 \
    --model-path "$QWEN8B_PATH" --model-name Qwen3-VL-8B &

CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" \
    --dataset ocr2.0 --cot --num-gpus 4 --gpu-id 3 \
    --model-path "$QWEN8B_PATH" --model-name Qwen3-VL-8B &

wait
echo "===== All done ====="
