import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for multi-label classification.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()
