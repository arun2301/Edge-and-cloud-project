import torch, open_clip, json, os
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity

DATA_DIR   = '../data/val2017'
ANNOT_FILE = '../data/annotations/captions_val2017.json'
RESULTS_DIR = '../results'
BATCH_SIZE = 32
DEVICE = 'cuda'

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

with open(ANNOT_FILE) as f:
    coco = json.load(f)
img_files = [os.path.join(DATA_DIR, img['file_name']) for img in coco['images'][:BATCH_SIZE]]
texts = ['a photo of ' + ann['caption'] for ann in coco['annotations'][:BATCH_SIZE]]

# ── Warmup ──────────────────────────────────────────────────────────
imgs_w = torch.stack([preprocess(Image.open(f).convert('RGB')) for f in img_files]).to(DEVICE)
for _ in range(5):
    with torch.no_grad():
        model.encode_image(imgs_w)
torch.cuda.synchronize()

# ── Profiled run ─────────────────────────────────────────────────────
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:

    with record_function('1_data_loading'):
        raw_images = [Image.open(f).convert('RGB') for f in img_files]

    with record_function('2_preprocessing'):
        tensors = [preprocess(img) for img in raw_images]
        batch   = torch.stack(tensors)

    with record_function('3_host_device_transfer'):
        batch = batch.to(DEVICE)

    with record_function('4_image_forward_pass'):
        with torch.no_grad():
            img_feats = model.encode_image(batch)

    with record_function('5_text_encoding'):
        tokens = tokenizer(texts).to(DEVICE)
        with torch.no_grad():
            txt_feats = model.encode_text(tokens)

    with record_function('6_similarity_computation'):
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        sim = img_feats @ txt_feats.T

torch.cuda.synchronize()

# ── Print results ─────────────────────────────────────────────────────
print('\n=== Pipeline Breakdown ===')
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))

# Save trace for visualization
prof.export_chrome_trace(f'{RESULTS_DIR}/04_trace_{DEVICE}.json')
print(f'\nTrace saved. Open in chrome://tracing or https://ui.perfetto.dev')