import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from models.lightning_module import CLIPSpatialLightningModule
from data.datamodules.clevr_dm import CLEVRDataModule

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Load LightningModule from checkpoint
    model = CLIPSpatialLightningModule.load_from_checkpoint(cfg.eval.checkpoint_path)
    
    # Datamodule for evaluation
    datamodule = CLEVRDataModule(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    
    trainer = pl.Trainer(
        gpus=cfg.trainer.gpus,
        strategy=cfg.trainer.strategy,
    )

    # Evaluate
    results = trainer.validate(model, datamodule=datamodule)
    print("Validation results:", results)

if __name__ == "__main__":
    main()

