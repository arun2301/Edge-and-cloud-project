import torch                   # PyTorch — our deep learning framework
import open_clip               # CLIP model library
import time                    # For measuring latency
import numpy as np             # Numerical operations
import pandas as pd            # For saving results as CSV
import json                    # For reading COCO annotations
import os                      # File path operations
from PIL import Image          # Loading images
from tqdm import tqdm          # Progress bar

# ── CONFIG ─────────────────────────────────────────────────────────
DATA_DIR    = '../data/val2017'
ANNOT_FILE  = '../data/annotations/captions_val2017.json'
RESULTS_DIR = '../results'
N_IMAGES    = 1000    # How many images to use
BATCH_SIZE  = 32      # Process 32 images at a time
N_WARMUP    = 5       # Warmup runs (GPU needs to 'wake up')
N_RUNS      = 3       # Repeat each experiment for reliability
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── LOAD MODEL ──────────────────────────────────────────────────────
# ViT-B/32 = Vision Transformer with 32x32 patches, ~150MB
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(DEVICE).eval()   # .eval() disables dropout for inference
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print('Model loaded successfully')

# ── LOAD COCO ANNOTATIONS ───────────────────────────────────────────
with open(ANNOT_FILE, 'r') as f:
    coco_data = json.load(f)

# Build a dict: image_id -> list of captions
img_captions = {}
for ann in coco_data['annotations']:
    iid = ann['image_id']
    if iid not in img_captions:
        img_captions[iid] = []
    img_captions[iid].append(ann['caption'])

# Get image file list
all_images = coco_data['images'][:N_IMAGES]
image_ids  = [img['id'] for img in all_images]
image_files = [os.path.join(DATA_DIR, img['file_name']) for img in all_images]

print(f'Loaded {len(image_files)} images from COCO')

# ── HELPER: MEASURE PEAK GPU MEMORY ─────────────────────────────────
def get_peak_memory_mb():
    if DEVICE == 'cuda':
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

# ── STAGE 1A: LATENCY & THROUGHPUT ──────────────────────────────────
print('\n=== Stage 1A: Latency Measurement ===')

latency_records = []

for batch_start in tqdm(range(0, N_IMAGES, BATCH_SIZE)):
    batch_files = image_files[batch_start:batch_start + BATCH_SIZE]

    # Load and preprocess images
    images = []
    for f in batch_files:
        img = Image.open(f).convert('RGB')  # Ensure 3-channel
        images.append(preprocess(img))       # Resize + normalize

    # Stack into a tensor: shape = [batch_size, 3, 224, 224]
    image_tensor = torch.stack(images).to(DEVICE)

    # Reset memory stats before this batch
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Warmup (first batch only)
    if batch_start == 0:
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = model.encode_image(image_tensor)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # Wait for GPU to finish

    # Timed run
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)

    if DEVICE == 'cuda':
        torch.cuda.synchronize()  # GPU is async — must sync before timing

    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000
    peak_mem   = get_peak_memory_mb()

    latency_records.append({
        'batch_size': len(batch_files),
        'latency_ms': elapsed_ms,
        'latency_per_image_ms': elapsed_ms / len(batch_files),
        'throughput_img_per_sec': len(batch_files) / (elapsed_ms / 1000),
        'peak_gpu_mem_mb': peak_mem,
        'device': DEVICE
    })

df_latency = pd.DataFrame(latency_records)
df_latency.to_csv(f'{RESULTS_DIR}/01_baseline_latency_{DEVICE}.csv', index=False)
print(df_latency.describe())

# ── STAGE 1B: RECALL@K ───────────────────────────────────────────────
print('\n=== Stage 1B: Computing Recall@1 and Recall@5 ===')

# Collect one caption per image for retrieval
texts = [img_captions[iid][0] for iid in image_ids if iid in img_captions]
N = min(len(texts), N_IMAGES)
texts = texts[:N]
image_files_eval = image_files[:N]

# Encode all images
all_img_feats = []
for i in tqdm(range(0, N, BATCH_SIZE), desc='Encoding images'):
    batch = image_files_eval[i:i+BATCH_SIZE]
    imgs  = torch.stack([preprocess(Image.open(f).convert('RGB')) for f in batch]).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # Normalise
    all_img_feats.append(feats.cpu())

img_matrix = torch.cat(all_img_feats)  # Shape: [N, 512]

# Encode all texts
all_txt_feats = []
for i in tqdm(range(0, N, BATCH_SIZE), desc='Encoding texts'):
    batch_txt = texts[i:i+BATCH_SIZE]
    tokens    = tokenizer(batch_txt).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    all_txt_feats.append(feats.cpu())

txt_matrix = torch.cat(all_txt_feats)  # Shape: [N, 512]

# Similarity matrix: [N_images, N_texts]
sim_matrix = img_matrix @ txt_matrix.T

# Recall@K: for each image i, does text i appear in top-K?
def recall_at_k(sim, k):
    top_k = sim.topk(k, dim=1).indices  # [N, k]
    correct = (top_k == torch.arange(len(sim)).unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

r1 = recall_at_k(sim_matrix, 1)
r5 = recall_at_k(sim_matrix, 5)
print(f'Recall@1: {r1:.4f} ({r1*100:.1f}%)')
print(f'Recall@5: {r5:.4f} ({r5*100:.1f}%)')

pd.DataFrame([{'device': DEVICE, 'precision': 'FP32',
               'recall_at_1': r1, 'recall_at_5': r5}]).to_csv(
    f'{RESULTS_DIR}/01_recall_{DEVICE}.csv', index=False
)
print('Done. Results saved to results/')