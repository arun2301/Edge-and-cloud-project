import torch, open_clip, time, pandas as pd
from PIL import Image
import os, json
from tqdm import tqdm

DATA_DIR   = '../data/val2017'
ANNOT_FILE = '../data/annotations/captions_val2017.json'
RESULTS_DIR = '../results'
N_IMAGES = 1000
BATCH_SIZE = 32
DEVICE = 'cuda'

# Load model in FP32 first (always start from FP32)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(DEVICE).eval()

tokenizer = open_clip.get_tokenizer('ViT-B-32')

with open(ANNOT_FILE) as f:
    coco = json.load(f)

img_files = [os.path.join(DATA_DIR, img['file_name'])
             for img in coco['images'][:N_IMAGES]]

results = []

for precision_label, dtype in [('FP32', torch.float32), ('FP16', torch.float16)]:
    print(f'\n--- Running {precision_label} ---')
    
    # Convert model to the target dtype
    mdl = model.to(dtype=dtype)
    
    torch.cuda.reset_peak_memory_stats()
    latencies = []

    for i in tqdm(range(0, N_IMAGES, BATCH_SIZE)):
        batch = img_files[i:i+BATCH_SIZE]
        imgs  = torch.stack([preprocess(Image.open(f).convert('RGB')) for f in batch]).to(DEVICE, dtype=dtype)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = mdl.encode_image(imgs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = sum(latencies) / len(latencies) / BATCH_SIZE  # per image
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    results.append({'precision': precision_label,
                    'avg_latency_per_img_ms': avg_lat,
                    'peak_mem_mb': peak_mem})

df = pd.DataFrame(results)
df.to_csv(f'{RESULTS_DIR}/02_fp16_comparison.csv', index=False)
print(df)