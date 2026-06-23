#!/bin/bash
# Run InternVL 38B on both datasets using 4x 5090 (2 GPUs per dataset)

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-38B}"
MODEL_NAME="InternVL3.5-38B"

# GPU 0,1 for ocr1.0
CUDA_VISIBLE_DEVICES=0,1 python eval_vllm.py \
    --dataset ocr1.0 \
    --tensor-parallel-size 2 \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" &

sleep 60

# GPU 2,3 for ocr2.0
CUDA_VISIBLE_DEVICES=2,3 python eval_vllm.py \
    --dataset ocr2.0 \
    --tensor-parallel-size 2 \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" &

wait
echo "Both datasets completed"
