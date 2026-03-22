#!/bin/bash
 
ANNO=/ssd/IISC/Sem5_Jan_2026/Edge_Cloud_Computing/Project/data/annotations/captions_val2017.json
IMGS=/ssd/IISC/Sem5_Jan_2026/Edge_Cloud_Computing/Project/data/val2017
 
# Interleave FP32 and FP16 per batch size so both run at similar board temperature
for bs in 1 2 4 8 16; do
    echo "=== CUDA FP32 batch_size=$bs ==="
    python run_experiments.py --device cuda --precision fp32 --batch_size $bs --annotations $ANNO --images_dir $IMGS
    echo "Cooling down 60s..."
    sleep 60
 
    echo "=== CUDA FP16 batch_size=$bs ==="
    python run_experiments.py --device cuda --precision fp16 --batch_size $bs --annotations $ANNO --images_dir $IMGS
    echo "Cooling down 60s..."
    sleep 60
done
 
# CPU FP32
for bs in 1 2 4 8 16; do
    echo "=== CPU FP32 batch_size=$bs ==="
    python run_experiments.py --device cpu --precision fp32 --batch_size $bs --annotations $ANNO --images_dir $IMGS
done
 
echo "All experiments complete!"
