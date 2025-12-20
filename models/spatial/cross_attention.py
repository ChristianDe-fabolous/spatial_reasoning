import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, queries, keys, values):
        out, _ = self.attn(queries, keys, values)
        return out

