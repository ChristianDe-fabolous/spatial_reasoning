import lightning as L
from torch.utils.data import DataLoader
from datasets.waymo_dataset import WaymoTrajectoryDataset
# from your_file import YourDatasetClass

class CLIPDataModule(L.LightningDataModule):
    def __init__(self, dataset_cfg, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        # Pass the flags from Hydra directly to your Dataset
        self.train_ds = WaymoTrajectoryDataset(
            data_path="data/waymo/processed/train",
            use_text=self.hparams.dataset_cfg.use_text,
            use_depth=self.hparams.dataset_cfg.use_depth,
            use_obj=self.hparams.dataset_cfg.use_obj
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=True
        )
