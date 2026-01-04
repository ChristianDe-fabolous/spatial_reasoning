import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from models.lightning_module import SpatialClipLightningModule
from loss.spatial_relation_loss import SpatialRelationLoss

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    # Logger
    if cfg.logger._target_ == "lightning.pytorch.loggers.WandbLogger":
        cfg.logger.project = cfg.task_name
    logger = hydra.utils.instantiate(cfg.logger)

    # Model
    model = hydra.utils.instantiate(cfg.model) 

    # Lightning Module
    loss_fn = hydra.utils.instantiate(cfg.loss)
    model = SpatialClipLightningModule(
        model=model,
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

