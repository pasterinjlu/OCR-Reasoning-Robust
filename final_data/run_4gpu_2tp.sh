#!/bin/bash
# Run InternVL3.5-14B evaluation on 4 GPUs (2 tensor parallel instances)
# Usage: bash run_4gpu_2tp.sh

cd "$(dirname "$0")"

MODEL_NAME="InternVL3.5-38B"

for DATASET in ocr1.0 ocr2.0; do
    echo "=== Evaluating $MODEL_NAME on $DATASET (4 GPUs, 2 instances) ==="

    CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn python eval_vllm.py --dataset $DATASET --gpu 0 --num-gpus 2 --gpu-id 0 --tensor-parallel-size 2 &
    sleep 60
    CUDA_VISIBLE_DEVICES=2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn python eval_vllm.py --dataset $DATASET --gpu 0 --num-gpus 2 --gpu-id 1 --tensor-parallel-size 2 &

    wait

    python merge_results.py --dataset $DATASET --model "${MODEL_NAME//./_}" --num-gpus 2
    echo "=== $DATASET done ==="
done

echo "All done!"
