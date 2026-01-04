import torch
import torch.nn as nn
from typing import Literal


class TrajectoryEncoder(nn.Module):
    """
    Trajectory encoder for imitation learning.
    Produces either:
      - a single global embedding (CLIP-style)
      - or a sequence of tokens (for cross-attention)
    """

    def __init__(
        self,
        t_past: int,
        hidden_dim: int,
        embed_dim: int,
        output_type: Literal["global", "tokens"] = "global",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.t_past = t_past
        self.embed_dim = embed_dim
        self.output_type = output_type

        self.input_proj = nn.Linear(2, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, ego_past: torch.Tensor) -> torch.Tensor:
        """
        ego_past: (B, T_past, 2)

        returns:
          - (B, embed_dim) if output_type == "global"
          - (B, T_past, embed_dim) if output_type == "tokens"
        """
        B, T, _ = ego_past.shape
        assert T == self.t_past, f"Expected T_past={self.t_past}, got {T}"

        # per-timestep embedding
        x = self.input_proj(ego_past)      # (B, T, hidden)
        x = self.mlp(x)                    # (B, T, hidden)
        x = self.output_proj(x)            # (B, T, embed_dim)

        if self.output_type == "tokens":
            return x

        # CLIP-style global trajectory embedding
        return x.mean(dim=1)

