#!/bin/bash
# Run Qwen3-VL Thinking models evaluation on 4 GPUs in parallel
# Usage: bash run_4gpu.sh

cd "$(dirname "$0")"

MODELS=(
    "${INTERNVL_38B_PATH:-OpenGVLab/InternVL3_5-38B}|InternVL3.5-38B"
)

for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_ENTRY%%|*}"
    MODEL_NAME="${MODEL_ENTRY##*|}"

    for DATASET in ocr2.0; do
        echo "=== Evaluating $MODEL_NAME on $DATASET (4 GPUs) ==="

        for i in 0 1 2 3; do
            CUDA_VISIBLE_DEVICES=$i python eval_vllm.py --dataset $DATASET --gpu 0 --num-gpus 4 --gpu-id $i \
                --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" &
        done

        wait

        python merge_results.py --dataset $DATASET --model "${MODEL_NAME//-/_}" --num-gpus 4
        echo "=== $MODEL_NAME $DATASET done ==="
    done
done

echo "All done!"
