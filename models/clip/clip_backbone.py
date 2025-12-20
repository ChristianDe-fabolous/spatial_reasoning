import torch
import torch.nn as nn
import open_clip

class CLIPBackbone(nn.Module):
    def __init__(self, model_name, pretrained, freeze=True):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, texts):
        tokens = self.tokenizer(texts)
        return self.model.encode_text(tokens)

    def forward(self, images, texts):
        img = self.encode_image(images)
        txt = self.encode_text(texts)
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return img, txt
