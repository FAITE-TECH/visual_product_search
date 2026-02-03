import os
import torch
import open_clip
from huggingface_hub import hf_hub_download

CACHE_DIR = os.path.abspath("model_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP model...")

hf_hub_download(
    repo_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    filename="open_clip_model.safetensors",
    cache_dir=CACHE_DIR,
    resume_download=True
)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    cache_dir=CACHE_DIR
)

model = model.to(device)
model.eval()

print("CLIP model loaded successfully")