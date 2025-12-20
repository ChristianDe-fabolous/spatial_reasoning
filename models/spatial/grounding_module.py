import torch
import torch.nn as nn

class GroundingModule(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, region_feats, text_feats):
        region_feats = self.proj(region_feats)
        scores = region_feats @ text_feats.unsqueeze(-1)
        return scores.squeeze(-1)

