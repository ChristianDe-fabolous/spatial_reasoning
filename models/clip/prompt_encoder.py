import torch
import torch.nn as nn

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, num_tokens):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, embed_dim))

    def forward(self, batch_size):
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)
