import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, images):
        return self.clip.encode_image(images)
