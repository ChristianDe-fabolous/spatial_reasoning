import torch
import torch.nn as nn

class CoordinateEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, boxes):
        return self.mlp(boxes)

