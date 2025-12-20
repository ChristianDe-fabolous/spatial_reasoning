import torch
import torch.nn as nn

class SpatialHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, image_feats, text_feats):
        joint = torch.cat([image_feats, text_feats], dim=-1)
        return self.mlp(joint)

