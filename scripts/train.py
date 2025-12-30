import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig
from models.lightning_module import CLIPSpatialLightningModule
from models.clip.clip_backbone import CLIPBackbone
from models.spatial.spatial_head import SpatialHead
from losses.spatial_relation_loss import SpatialRelationLoss

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Instantiate backbone
    clip_model = CLIPBackbone(cfg.model.clip.model_name, pretrained=True)


    spatial_head = SpatialHead(cfg)

    # Loss
    loss_fn = SpatialRelationLoss()

    # Lightning module
    model = CLIPSpatialLightningModule(
        clip_model=clip_model,
        spatial_head=spatial_head,
        loss_fn=loss_fn,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
    )

    # Datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)


    # Callbacks
    checkpoint_cb = ModelCheckpoint(monitor="val/loss", save_top_k=1)
    early_stop_cb = EarlyStopping(monitor="val/loss", patience=cfg.trainer.patience)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        gpus=cfg.trainer.gpus,
        strategy=cfg.trainer.strategy,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

