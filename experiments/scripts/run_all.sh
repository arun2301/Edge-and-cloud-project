#!/bin/bash
 
OUTPUT="../results/results.csv"
ANNOTATIONS="../../data/annotations/captions_val2017.json"
IMAGES_DIR="../../data/val2017"
 
# Remove old results to start fresh
rm -f "$OUTPUT"
 
for BS in 1 2 4 8 16; do
    echo "============================================================"
    echo "Batch size $BS — CUDA FP32"
    python run_experiments.py --device cuda --precision fp32 --batch_size $BS --output_file "$OUTPUT" --annotations "$ANNOTATIONS" --images_dir "$IMAGES_DIR"
 
    echo "Cooling down 60s..."
    sleep 60
 
    echo "Batch size $BS — CUDA FP16"
    python run_experiments.py --device cuda --precision fp16 --batch_size $BS --output_file "$OUTPUT" --annotations "$ANNOTATIONS" --images_dir "$IMAGES_DIR"
 
    echo "Cooling down 60s..."
    sleep 60
done
 
echo "============================================================"
echo "CPU FP32 runs"
for BS in 1 2 4 8 16; do
    echo "Batch size $BS — CPU FP32"
    python run_experiments.py --device cpu --precision fp32 --batch_size $BS --output_file "$OUTPUT" --annotations "$ANNOTATIONS" --images_dir "$IMAGES_DIR"
done
 
echo "All done! Results saved to $OUTPUT"
