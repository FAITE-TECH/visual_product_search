# Visual Product Search

An AI-powered **Visual Product Search** system that allows users to upload an image and find visually similar products using deep learning embeddings (CLIP).

This project is designed with **real-world ML best practices**, separating source code from large model files and runtime data.

## Features

- Image-based product search
- Uses **CLIP (Contrastive Language–Image Pretraining)** for embeddings
- Fast similarity search
- Clean API-based backend
- Scalable project structure
- Production-friendly (no large models committed to Git)

## Tech Stack

- **Python**
- **FastAPI**
- **PyTorch**
- **Hugging Face Transformers**
- **CLIP (ViT-B/32)**
- **NumPy**
- **Uvicorn**

## Important Note About Models
This project **does NOT commit ML models to GitHub**.

Why?
- Model files are **~1.13GB**
- GitHub (even with LFS) has strict limits
- Industry standard is **runtime download**

The models will be **automatically downloaded** when the app runs for the first time.

## Model Download (Automatic)
CLIP models are downloaded automatically via Hugging Face and cached locally.

Example used in code:

```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="model_cache"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="model_cache"
)

##Setup Instructions
-1 clone the repo
-2 create the virtual env
-3 Install dependencies
-4 set the correcet .env
-5 Run the app
```bash
uvicorn app.main:app --reload
```
- 6The app will be available at
  ```bash
http://127.0.0.1:8000/docs
```
