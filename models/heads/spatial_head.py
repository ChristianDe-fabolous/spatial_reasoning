import torch
import torch.nn as nn

class SpatialHead(nn.Module):
    def __init__(self, hidden_dim, num_latents=16):
        super().__init__()
        # 1. Define multiple latents (learned summary tokens)
        # Shape: (num_latents, hidden_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            batch_first=True
        )

    def forward(self, **modalities):
        # 1. Filter out None and concatenate available tokens
        # modalities is a dict: {"image": tensor, "text": None, ...}
        available_feats = [v for v in modalities.values() if v is not None]
        context = torch.cat(available_feats, dim=1) # (Batch, Total_Tokens, hidden_dim)

        # 2. Expand latents to match Batch Size
        b = context.shape[0]
        # (num_latents, hidden_dim) -> (Batch, num_latents, hidden_dim)
        query = self.latents.unsqueeze(0).repeat(b, 1, 1)

        # 3. Cross-Attention
        # Latents (Q) suck up info from the Context (K, V)
        attn_out, _ = self.cross_attn(query=query, key=context, value=context)
        
        # 4. Residual connection (optional but recommended)
        return query + attn_out


