#!/bin/bash
# Run evaluation on 3 GPUs in parallel
# Usage: bash run_3gpu.sh ocr1.0

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: bash run_3gpu.sh <dataset>"
    exit 1
fi

# Launch 3 processes in parallel
CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --dataset $DATASET --gpu 0 --num-gpus 3 --gpu-id 0 &
CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --dataset $DATASET --gpu 0 --num-gpus 3 --gpu-id 1 &
CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --dataset $DATASET --gpu 0 --num-gpus 3 --gpu-id 2 &

# Wait for all to complete
wait

# Merge results
python merge_results.py --dataset $DATASET --num-gpus 3

echo "Done! Results saved to ../eval_results/${DATASET}_Qwen3_VL_8B.json"
