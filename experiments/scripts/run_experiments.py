import torch
from transformers import CLIPProcessor, CLIPModel
import time
import argparse
import pandas as pd
import numpy as np
import os
import re
import json
import subprocess
from PIL import Image
from tqdm import tqdm
 
 
def get_device_info(device):
    if device.type == 'cuda':
        return torch.cuda.get_device_name(0)
    return "CPU"
 
 
def calculate_recall_at_k(image_features, text_features, k=5):
    logits_per_image = torch.matmul(image_features, text_features.t())
    num_samples = image_features.shape[0]
    k = min(k, num_samples)
    _, top_k_indices = torch.topk(logits_per_image, k, dim=1)
    targets = torch.arange(num_samples, device=image_features.device)
    correct_predictions = torch.sum(top_k_indices == targets.view(-1, 1))
    return correct_predictions.item() / num_samples
 
 
def parse_tegrastats(log_file):
    """Parse VDD_CPU_GPU_CV (compute-only power) from tegrastats log."""
    power_readings = []
    try:
        with open(log_file) as f:
            for line in f:
                match = re.search(r'VDD_CPU_GPU_CV (\d+)mW/\d+mW', line)
                if match:
                    power_readings.append(int(match.group(1)))
    except FileNotFoundError:
        return 0
    return sum(power_readings) / len(power_readings) if power_readings else 0
 
 
def measure_idle_power(duration_sec=3):
    """Measure idle board power before inference starts."""
    idle_log = "tegrastats_idle.txt"
    subprocess.run(['sudo', 'tegrastats', '--stop'], stderr=subprocess.DEVNULL)
    subprocess.Popen(
        ['sudo', 'tegrastats', '--interval', '500', '--logfile', idle_log],
        stderr=subprocess.DEVNULL
    )
    time.sleep(duration_sec)
    subprocess.run(['sudo', 'tegrastats', '--stop'], stderr=subprocess.DEVNULL)
    return parse_tegrastats(idle_log)
 
 
def load_coco_data(annotations_path, images_dir, num_samples=50):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
 
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
 
    seen_ids = {}
    for ann in coco_data['annotations']:
        if ann['image_id'] not in seen_ids:
            seen_ids[ann['image_id']] = ann['caption']
 
    items = list(seen_ids.items())[:num_samples]
 
    images = [
        Image.open(os.path.join(images_dir, id_to_filename[img_id])).convert("RGB")
        for img_id, _ in items
    ]
    texts = [caption for _, caption in items]
    return images, texts
 
 
def run_inference(args):
    # 1. Setup Device and Precision
    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == 'fp16' else torch.float32
 
    if args.precision == 'fp16' and args.device == 'cpu':
        raise ValueError("FP16 is not supported on CPU. Use --precision fp32.")
 
    print(f"Running on: {get_device_info(device)}")
    print(f"Using Precision: {args.precision.upper()}")
    print(f"Batch Size: {args.batch_size}")
 
    # 2. Load Model and Processor
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device, dtype=dtype)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
 
    # 3. Load Dataset
    print("Loading MS-COCO dataset subset from local files...")
    images, texts = load_coco_data(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        num_samples=args.num_samples
    )
    num_samples = len(images)
    print(f"Loaded {num_samples} image-caption pairs.")
 
    # 4. Warmup run
    print("Performing a warm-up run...")
    inputs = processor(text=[texts[0]], images=[images[0]], return_tensors="pt", padding=True).to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
    with torch.no_grad():
        _ = model(**inputs)
 
    # 5. Measure idle power baseline (GPU only)
    idle_power_mw = 0
    if device.type == 'cuda':
        print("Measuring idle power baseline (3 seconds)...")
        idle_power_mw = measure_idle_power(duration_sec=3)
        print(f"Idle power: {idle_power_mw:.0f} mW")
 
    # 6. Start active energy logging
    energy_log = "tegrastats_active.txt"
    if device.type == 'cuda':
        subprocess.run(['sudo', 'tegrastats', '--stop'], stderr=subprocess.DEVNULL)
        subprocess.Popen(
            ['sudo', 'tegrastats', '--interval', '500', '--logfile', energy_log],
            stderr=subprocess.DEVNULL
        )
 
    # 7. Run Experiment
    latencies = []
    total_recall = 0
 
    print("Starting experiment...")
    start_time_total = time.time()
 
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, args.batch_size)):
            batch_images = images[i:i + args.batch_size]
            batch_texts = texts[i:i + args.batch_size]
 
            if not batch_images:
                continue
 
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True).to(device)
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
 
            start_time_batch = time.time()
            outputs = model(**inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time_batch = time.time()
 
            latencies.append(end_time_batch - start_time_batch)
 
            recall = calculate_recall_at_k(outputs.image_embeds, outputs.text_embeds, k=5)
            total_recall += recall * len(batch_images)
 
    end_time_total = time.time()
 
    # Stop energy logging
    if device.type == 'cuda':
        subprocess.run(['sudo', 'tegrastats', '--stop'], stderr=subprocess.DEVNULL)
 
    # 8. Collect and Save Results
    avg_latency_ms = np.mean(latencies[1:]) * 1000 if len(latencies) > 1 else latencies[0] * 1000
    avg_throughput_fps = num_samples / (end_time_total - start_time_total)
    final_recall = total_recall / num_samples
    max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0
 
    active_power_mw = parse_tegrastats(energy_log) if device.type == 'cuda' else 0
    net_compute_power_mw = max(0, active_power_mw - idle_power_mw)
 
    # Energy per image in millijoules
    energy_per_image_mj = (net_compute_power_mw * avg_latency_ms) / (args.batch_size * 1000) if net_compute_power_mw > 0 else 0
 
    results = {
        'device': get_device_info(device),
        'precision': args.precision.upper(),
        'batch_size': args.batch_size,
        'avg_latency_ms': avg_latency_ms,
        'throughput_fps': avg_throughput_fps,
        'recall_at_5': final_recall,
        'max_memory_mb': max_memory_mb,
        'idle_power_mw': idle_power_mw,
        'active_power_mw': active_power_mw,
        'net_compute_power_mw': net_compute_power_mw,
        'energy_per_image_mj': energy_per_image_mj,
    }
 
    print("\n--- Results ---")
    for key, val in results.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")
 
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
 
    df = pd.DataFrame([results])
    if os.path.exists(args.output_file):
        df.to_csv(args.output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(args.output_file, mode='w', header=True, index=False)
 
    print(f"\nResults appended to {args.output_file}")
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CLIP performance experiments.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--output_file', type=str, default='../results/results.csv')
    parser.add_argument('--annotations', type=str, default='../../data/annotations/captions_val2017.json')
    parser.add_argument('--images_dir', type=str, default='../../data/val2017')
 
    args = parser.parse_args()
    run_inference(args)
