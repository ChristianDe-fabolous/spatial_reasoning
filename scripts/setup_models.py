import os
from huggingface_hub import hf_hub_download
from transformers import ViTModel, DetrForObjectDetection

# Configuration: Define where models should go
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = [
    {"repo": "model", "file": "resnet50.ckpt"},
    {"repo": "bert-base-uncased", "file": "pytorch_model.bin"}
]

def download():
    print(f"Downloading models to {MODEL_DIR}...")
    for m in MODELS:
        # Example using HF hub; replace with direct download logic if needed
        path = hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=MODEL_DIR)
        print(f"Fetched: {path}")

if __name__ == "__main__":
    download()