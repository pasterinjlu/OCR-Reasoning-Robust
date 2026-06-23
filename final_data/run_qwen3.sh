#!/bin/bash
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Thinking}"
MODEL_NAME="Qwen3-VL-4B-Thinking"

CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --dataset ocr1.0 --gpu 0 --model-path $MODEL_PATH --model-name $MODEL_NAME &
CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --dataset ocr2.0 --gpu 0 --model-path $MODEL_PATH --model-name $MODEL_NAME &

wait
echo "Both datasets completed!"
