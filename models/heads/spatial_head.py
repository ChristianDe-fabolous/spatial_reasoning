import torch
import torch.nn as nn

class SpatialHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, image_feats, text_feats=None, obj_feats=None, depth_feats=None):
        joint = torch.cat([image_feats, text_feats, obj_feats, depth_feats], dim=-1)
        return self.mlp(joint)

