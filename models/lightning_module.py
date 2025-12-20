import pytorch_lightning as pl
import torch
from torch.optim import AdamW

class CLIPSpatialLightningModule(pl.LightningModule):
    def __init__(
        self,
        clip_model,
        spatial_head,
        loss_fn,
        optimizer_cfg,
        scheduler_cfg=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["clip_model", "spatial_head", "loss_fn"])
        self.clip = clip_model
        self.spatial_head = spatial_head
        self.loss_fn = loss_fn

    def forward(self, batch):
        image_feats, text_feats = self.clip(
            batch["images"],
            batch["text"]
        )
        spatial_preds = self.spatial_head(image_feats, text_feats)
        return spatial_preds

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss_dict = self.loss_fn(preds, batch)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()},
                      prog_bar=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        loss_dict = self.loss_fn(preds, batch)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()},
                      prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.optimizer_cfg.lr,
            weight_decay=self.hparams.optimizer_cfg.weight_decay,
        )
        if self.hparams.scheduler_cfg is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler_cfg.t_max
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }
