import torch.nn as nn
from transformers import CLIPVisionModel

class ImageEncoder(nn.Module):
    def __init__(self, hf_model, out_dim, pretrained=True, freeze=True):
        model = CLIPVisionModel.from_pretrained(hf_model)

        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        self.model = model
        self.out_dim = out_dim

    def forward(self, images):
        return self.model.encode_image(images)

