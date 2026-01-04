import torch
import torch.nn as nn

class SpatialRelationLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        predictions: logits (B, num_classes)
        targets: class indices (B,)
        """
        loss = self.loss_fn(predictions, targets)
        return self.weight * loss
