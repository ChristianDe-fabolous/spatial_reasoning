import torch
import torch.nn as nn
import hydra


class DiffusionPredictor(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # ONE shared attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout)

    def forward(self, noisy_future, context):
        # 1. Map everything to the same 'hidden_dim'
        query = self.traj_proj(noisy_future)
        out, _ = self.cross_attn(query=query, key=context, value=context)
        
        return out
    
    
