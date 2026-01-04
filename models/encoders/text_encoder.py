import torch.nn as nn
from transformers import CLIPTextModel

class TextEncoder(nn.Module):
    def __init__(self, hf_model, out_dim, pretrained=True, freeze=True):
        model = CLIPTextModel.from_pretrained(hf_model)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        self.model = model
        self.out_dim = out_dim

    def forward(self, text):
        return self.model.encode_text(text)
