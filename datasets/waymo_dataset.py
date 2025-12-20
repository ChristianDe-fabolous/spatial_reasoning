from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import os


class WaymoTrajectoryDataset(Dataset):
    def __init__(self, root_dir, T_past, T_future, transform=None):
        self.root_dir = root_dir
        self.samples = self.list_all_samples(root_dir)
        self.T_past = T_past
        self.T_future = T_future
        self.transform = transform

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        
        image_paths = sorted((sample_dir / "images").glob("*.jpg"))
        
        images = [self.load_image(p) for p in image_paths]
        # Load and sort images
        if self.transform:
            images = [self.transform(img) for img in images]
        images = torch.stack(images)

        # Load trajectories
        ego_past = torch.from_numpy(np.load(sample_dir / 'ego_past.npy')).float()
        ego_future = torch.from_numpy(np.load(sample_dir / 'ego_future.npy')).float()
        objects = torch.from_numpy(np.load(sample_dir / 'objects.npy')).float()

        return images, ego_past, ego_future, objects


    def list_all_samples(self, root_dir):
        root = Path(root_dir)
        return sorted([d for d in root.iterdir() if d.is_dir()])
    

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return T.ToTensor()(img)  # returns (C,H,W) tensor

