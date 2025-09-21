from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

# Text embedding model
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Image embedding model
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(text: str):
    emb = text_model.encode([text])[0]
    return emb / np.linalg.norm(emb)

def get_image_embedding(image_file):
    image = Image.open(image_file).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = image_model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb[0].numpy()
