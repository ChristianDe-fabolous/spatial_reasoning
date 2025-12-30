import torch
import torch.nn as nn

class SpatialHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Instantiate spatial head
        input_dim = cfg.model.image_dim
        if cfg.active_encoders.use_text:
            input_dim += cfg.model.text_dim
        if cfg.active_encoders.use_obj:
            input_dim += cfg.model.obj_dim
        if cfg.active_encoders.use_depth:
            input_dim += cfg.model.depth_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, cfg.models.spatial_head.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.models.spatial.hidden_dim, cfg.models.spatial.out_dim),
        )

    def forward(self, image_feats, text_feats=None, obj_feats=None, depth_feats=None):
        joint = torch.cat([image_feats, text_feats, obj_feats, depth_feats], dim=-1)
        return self.mlp(joint)

