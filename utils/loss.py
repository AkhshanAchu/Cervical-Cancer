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
