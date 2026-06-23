#!/bin/bash
# Run both datasets on 3 GPUs

echo "Starting ocr1.0..."
bash run_3gpu.sh ocr1.0

echo "Starting ocr2.0..."
bash run_3gpu.sh ocr2.0

echo "All done!"
