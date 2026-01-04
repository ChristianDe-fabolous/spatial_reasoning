import torch
import torch.nn as nn
import hydra

class SpatialClip(nn.Module):
    def __init__(
        self,
        encoders,
        active_encoders,
        spatial_head,
        temporal_head=None,
        predictor=None,
    ):
        super().__init__()

        self.encoders = nn.ModuleDict()
        fused_dim = 0

        for name, cfg in encoders.items():
            if active_encoders.get(name):
                enc = hydra.utils.instantiate(cfg)
                self.encoders[name] = enc
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

    def forward(self, batch):
        feats = []

        if "image" in self.encoders:
            feats.append(self.encoders["image"](batch["images"]))

        if "text" in self.encoders:
            feats.append(self.encoders["text"](batch["text"]))

        if "object" in self.encoders:
            feats.append(self.encoders["object"](batch["objects"]))

        x = torch.cat(feats, dim=-1)

        x = self.spatial_head(x)

        if self.temporal_head is not None:
            x = self.temporal_head(x)

        return x