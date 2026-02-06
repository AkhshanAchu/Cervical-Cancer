import torch
import torch.nn as nn


bce =  nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def segmentation_loss(out, cyt_gt, nuc_gt, lambda_nuc=1.0):
    cyt_loss = bce(out["cyt"], cyt_gt) + dice_loss(out["cyt"], cyt_gt)
    nuc_loss = bce(out["nuc"], nuc_gt) + dice_loss(out["nuc"], nuc_gt)
    return cyt_loss + lambda_nuc * nuc_loss

@torch.no_grad()
def dice_score(logits, targets, eps=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # This should be a tensor (or None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
        else:
            alpha = None

        ce_loss = F.cross_entropy(inputs, targets, weight=alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_loss(counts = [247, 247, 260, 187, 160,]):
    class_counts = torch.tensor(counts, dtype=torch.float)
    alpha = 1.0 / class_counts
    alpha = alpha / alpha.sum()  # Normalize
    return FocalLoss(alpha=alpha, gamma=2.0)
