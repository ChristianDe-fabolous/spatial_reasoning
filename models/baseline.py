import torch
import torch.nn as nn
import hydra

class SpatialCrossAttention(nn.Module):
    def __init__(
        self,
        encoders,
        spatial_head,
        temporal_head=None,
        predictor=None,
    ):
        super().__init__()

        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        fused_dim = 0

        for name, cfg in encoders.items():
            enc = hydra.utils.instantiate(cfg)
            self.encoders[name] = enc
            self.projectors[name] = nn.Linear(enc.out_dim, enc.out_dim)
            fused_dim += enc.out_dim

        self.spatial_head = hydra.utils.instantiate(
            spatial_head,
            input_dim=fused_dim,
        )

        spatial_out_dim = self.spatial_head.out_dim

        if temporal_head is not None:
            self.temporal_head = hydra.utils.instantiate(
                temporal_head,
                input_dim=spatial_out_dim,
            )
            temporal_out_dim = self.temporal_head.out_dim
        else:
            self.temporal_head = None
            temporal_out_dim = spatial_out_dim

        self.out_dim = temporal_out_dim

        self.predictor = hydra.utils.instantiate(predictor)

    def forward(self, batch):
        context_tokens = []

        for name, encoder in self.encoders.items():
            raw_feat = encoder(batch[name])

            projected = self.predictors[name](raw_feat)
            context_tokens.append(projected)


        context = torch.cat(context_tokens, dim=1)

        x = self.spatial_head(x)

        if self.temporal_head is not None:
            x = self.temporal_head(x)

        noisy_future = torch.randn(batch["trajectory"].shape).to(self.device)
        x = self.predictor(x)

        