# Add this to scripts/03_batch_sweep.py
import torch, open_clip, time, pandas as pd
from PIL import Image
import os, json
from tqdm import tqdm

DATA_DIR    = '../data/val2017'
ANNOT_FILE  = '../data/annotations/captions_val2017.json'
RESULTS_DIR = '../results'
DEVICE = 'cuda'
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(DEVICE).eval()

with open(ANNOT_FILE) as f:
    coco = json.load(f)
img_files = [os.path.join(DATA_DIR, img['file_name']) for img in coco['images'][:256]]

results = []
for bs in BATCH_SIZES:
    print(f'Batch size: {bs}')
    torch.cuda.reset_peak_memory_stats()
    latencies = []

    for i in range(0, len(img_files) - bs, bs):
        batch = img_files[i:i+bs]
        imgs  = torch.stack([preprocess(Image.open(f).convert('RGB')) for f in batch]).to(DEVICE)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.encode_image(imgs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000 / bs)

    results.append({'batch_size': bs,
                    'avg_latency_per_img_ms': sum(latencies)/len(latencies),
                    'peak_mem_mb': torch.cuda.max_memory_allocated()/1024**2,
                    'throughput': bs / (sum(latencies)/len(latencies) * 1000)})

pd.DataFrame(results).to_csv(f'{RESULTS_DIR}/03_batch_sweep.csv', index=False)
print(pd.DataFrame(results))