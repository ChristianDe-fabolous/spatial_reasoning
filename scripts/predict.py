import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from models.lightning_module import CLIPSpatialLightningModule
from data.datamodules.clevr_dm import CLEVRDataModule

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Load trained LightningModule
    model = CLIPSpatialLightningModule.load_from_checkpoint(cfg.predict.checkpoint_path)
    model.eval()
    
    # Datamodule for inference
    datamodule = CLEVRDataModule(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    
    trainer = pl.Trainer(
        gpus=cfg.trainer.gpus,
        strategy=cfg.trainer.strategy,
    )

    # Predict
    predictions = trainer.predict(model, datamodule=datamodule)
    
    # Flatten predictions for convenience
    flat_preds = [p for batch in predictions for p in batch]
    print(f"Predicted {len(flat_preds)} samples")
    
    # Optionally save predictions
    import pickle
    with open(cfg.predict.output_path, "wb") as f:
        pickle.dump(flat_preds, f)
    print(f"Predictions saved to {cfg.predict.output_path}")

if __name__ == "__main__":
    main()

