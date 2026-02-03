from PIL import Image
import torch
from .clip_model import model, preprocess, device

def encode_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0].tolist()
