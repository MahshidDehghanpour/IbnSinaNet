"""
losses.py
=========
سه تا loss function برای مقایسه در ablation:
  1. CrossEntropyLoss       (استاندارد — فعلی)
  2. DiceLoss               (مناسب برای class imbalance)
  3. FocalLoss              (مناسب برای class imbalance)
  4. CE + Dice ترکیبی       (بهترین گزینه در عمل)

استفاده در argprocess.py:
    opt.loss_type = 'CE'       # یا 'Dice' یا 'Focal' یا 'CE+Dice'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss برای segmentation با class imbalance.
    مستقیماً Dice coefficient رو optimize می‌کنه.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)  targets: (B, H, W) long
        probs = F.softmax(logits, dim=1)[:, 1, :, :]   # احتمال کلاس ضایعه
        targets_f = targets.float()

        intersection = (probs * targets_f).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets_f.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss — برای class imbalance طراحی شده.
    پیکسل‌های سخت (ضایعه) رو بیشتر وزن می‌ده.
    gamma=2 مقدار استاندارد از مقاله Lin et al. 2017
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # می‌تونی وزن کلاس ضایعه رو بیشتر کنی

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets.long(),
                                  weight=self.weight, reduction='none')
        pt      = torch.exp(-ce_loss)
        focal   = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


class CombinedLoss(nn.Module):
    """
    CE + Dice ترکیبی — بهترین عملکرد در اکثر مقالات MS segmentation.
    alpha کنترل می‌کنه چقدر از هر کدوم استفاده بشه.
    """
    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.alpha    = alpha
        self.ce       = nn.CrossEntropyLoss()
        self.dice     = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        ce_loss   = self.ce(logits, targets.long())
        dice_loss = self.dice(logits, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


def get_loss(loss_type: str, device='cuda'):
    """
    در trainer.py به جای nn.CrossEntropyLoss() از این استفاده کن:
        criterion = get_loss(opt.loss_type)
    """
    if loss_type == 'CE':
        return nn.CrossEntropyLoss()

    elif loss_type == 'Dice':
        return DiceLoss()

    elif loss_type == 'Focal':
        return FocalLoss(gamma=2.0)

    elif loss_type == 'CE+Dice':
        return CombinedLoss(alpha=0.5)

    else:
        raise ValueError(f"loss_type نامعتبر: {loss_type}. "
                         f"مقادیر معتبر: CE, Dice, Focal, CE+Dice")
