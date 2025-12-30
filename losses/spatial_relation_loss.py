import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialRelationLoss(nn.Module):
    def __init__(self):
        super(SpatialRelationLoss, self).__init__()
        # Standard CrossEntropy for categorical spatial relationships
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Logits from SpatialHead (Batch, Num_Relations)
            targets: Ground truth class indices (Batch)
        """
        return self.loss_fn(predictions, targets)