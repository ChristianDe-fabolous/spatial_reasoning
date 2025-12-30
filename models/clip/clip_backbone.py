import torch
import torch.nn as nn
import open_clip
from transformers import AutoModel, AutoConfig, AutoTokenizer

class CLIPBackbone(nn.Module):
    def __init__(self, model_name_image_enc, model_name_obj,_enc, model_name_depth_enc, model_name_traj_enc, pretrained, freeze=True):
        super().__init__()
        # self.model, _, _ = open_clip.create_model_and_transforms(
        #     model_name, pretrained=pretrained
        # )
        self.image_encoder = AutoModel.from_pretrained(
            model_name_image_enc,
            trust_remote_code=True
        )

        self.object_encoder = AutoModel.from_pretrained(
            model_name_obj,
            trust_remote_code=True
        )

        self.depth_encoder = AutoModel.from_pretrained(
            model_name_depth_enc,
            trust_remote_code=True
        )

        self.trajectory_encoder = AutoModel.from_pretrained(
            model_name_traj_enc,
            trust_remote_code=True
        )

        self.image_tokenizer = AutoTokenizer.from_pretrained(
            model_name_image_enc,
            trust_remote_code=True
        )

        self.object_tokenizer = AutoTokenizer.from_pretrained(
            model_name_obj,
            trust_remote_code=True
        )

        self.depth_tokenizer = AutoTokenizer.from_pretrained(
            model_name_depth_enc,
            trust_remote_code=True
        )

        self.traj_tokenizer = AutoTokenizer.from_pretrained(
            model_name_traj_enc,
            trust_remote_code=True
        )

        if freeze:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.object_encoder.parameters():
                p.requires_grad = False
            for p in self.depth_encoder.parameters():
                p.requires_grad = False
            

    def encode_image(self, images):
        tokens = self.image_tokenizer(images)
        return self.image_encoder.encode(tokens)

    def encode_object(self, images):
        tokens = self.object_tokenizer(images)
        return self.object_encoder.encode(tokens)

    def encode_text(self, texts):
        tokens = self.tokenizer(texts)
        return self.model.encode_text(tokens)

    def encode_depth(self, images):
        tokens = self.depth_tokenizer(images)
        return self.depth_encoder.encode(tokens)
    

    def forward(self, images, texts):
        '''
        Params:
        images:
        texts:

        Return:
        img:
        txt:
        obj:
        depth:
        '''
        img = self.encode_image(images)
        img = img / img.norm(dim=-1, keepdim=True)

        obj = None
        depth = None

        if self.object_encoder:
            obj = self.encode_object(images)
            obj = obj / obj.norm(dim=-1, keepdim=True)
        if self.depth_encoder:
            depth = self.encode_depth(images)
            depth = depth / depth.norm(dim=-1, keepdim=True)
        if self.text_encoder:
            txt = self.encode_text(texts)
            txt = txt / txt.norm(dim=-1, keepdim=True)

        return img, txt, obj, depth