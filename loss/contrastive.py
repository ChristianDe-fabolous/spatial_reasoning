import torch
import torch.nn.functional as F

class ContrastiveLoss:
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def __call__(self, preds, batch):
        logits = preds / self.temperature
        targets = torch.arange(len(logits), device=logits.device)
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss}

