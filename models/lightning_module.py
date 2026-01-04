import pytorch_lightning as pl
import torch
from torch.optim import AdamW

class SpatialClipLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,              
        loss_fn,
        optimizer_cfg,
        scheduler_cfg=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn


    def forward(self, batch):
        return self.model(batch)


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
            T_max=self.hparams.scheduler_cfg.t_max,
            eta_min=self.hparams.scheduler_cfg.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }
