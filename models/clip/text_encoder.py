import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, text):
        return self.clip.encode_text(text)
