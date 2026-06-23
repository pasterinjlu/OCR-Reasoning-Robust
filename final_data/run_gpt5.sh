#!/bin/bash
# Run GPT-5.2 evaluation on both datasets with concurrency

MODEL="gpt-5.2"
CONCURRENCY=20

echo "Evaluating $MODEL on ocr1.0..."
python eval_api.py --dataset ocr1.0 --model $MODEL --concurrency $CONCURRENCY

echo "Evaluating $MODEL on ocr2.0..."
python eval_api.py --dataset ocr2.0 --model $MODEL --concurrency $CONCURRENCY

echo "Done! Results saved to ../eval_results/"
