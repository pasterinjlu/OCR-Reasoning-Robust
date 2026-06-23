#!/bin/bash
# Run InternVL 38B on 4 GPUs with tensor parallelism

cd "$(dirname "$0")"

MODEL_NAME="InternVL3.5-38B"

for DATASET in ocr1.0 ocr2.0; do
    echo "=== Evaluating $MODEL_NAME on $DATASET (4 GPUs tensor parallel) ==="

    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_vllm.py \
        --dataset $DATASET \
        --tensor-parallel-size 4

    echo "=== $DATASET done ==="
done

echo "All done!"
