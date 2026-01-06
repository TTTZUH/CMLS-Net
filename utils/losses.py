"""
Loss Functions for CMLS-Net
Includes Cross Entropy Loss and Dice Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    Optimizes region-level overlap
    """
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
        
        Returns:
            Dice loss value
        """
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        B, C, H, W = pred.shape
        target_one_hot = torch.zeros_like(pred)
        
        # Create mask for valid pixels
        valid_mask = (target != self.ignore_index)
        target_masked = target.clone()
        target_masked[~valid_mask] = 0
        
        target_one_hot.scatter_(1, target_masked.unsqueeze(1), 1.0)
        
        # Apply mask to predictions and targets
        valid_mask = valid_mask.unsqueeze(1).expand_as(pred)
        pred = pred * valid_mask.float()
        target_one_hot = target_one_hot * valid_mask.float()
        
        # Calculate Dice score for each class
        dice_scores = []
        for c in range(C):
            pred_c = pred[:, c, :, :].contiguous().view(B, -1)
            target_c = target_one_hot[:, c, :, :].contiguous().view(B, -1)
            
            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average across classes and batch
        dice_scores = torch.stack(dice_scores, dim=0)
        dice_loss = 1.0 - dice_scores.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Cross Entropy and Dice Loss
    As described in the paper: L_total = L_CE + λ * L_Dice
    """
    def __init__(self, num_classes, ce_weight=None, dice_weight=1.0, 
                 ignore_index=255, smooth=1.0):
        super(CombinedLoss, self).__init__()
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=ce_weight,
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
        
        Returns:
            Combined loss value
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total_loss = ce + self.dice_weight * dice
        
        return total_loss, ce, dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
        """
        # Calculate cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none', 
                                  ignore_index=self.ignore_index)
        
        # Calculate focal term
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma) * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


def get_loss_function(loss_type='combined', num_classes=6, class_weights=None):
    """
    Factory function to get loss function
    
    Args:
        loss_type: Type of loss ('ce', 'dice', 'combined', 'focal')
        num_classes: Number of segmentation classes
        class_weights: Optional class weights for CE loss
    
    Returns:
        Loss function
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    elif loss_type == 'dice':
        return DiceLoss(smooth=1.0, ignore_index=255)
    
    elif loss_type == 'combined':
        return CombinedLoss(
            num_classes=num_classes,
            ce_weight=class_weights,
            dice_weight=1.0,
            ignore_index=255
        )
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=255)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Dummy data
    B, C, H, W = 2, 6, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    
    # Test different losses
    print("\n1. Cross Entropy Loss:")
    ce_loss = get_loss_function('ce', num_classes=C)
    loss_val = ce_loss(pred, target)
    print(f"Loss value: {loss_val.item():.4f}")
    
    print("\n2. Dice Loss:")
    dice_loss = get_loss_function('dice')
    loss_val = dice_loss(pred, target)
    print(f"Loss value: {loss_val.item():.4f}")
    
    print("\n3. Combined Loss:")
    combined_loss = get_loss_function('combined', num_classes=C)
    loss_val, ce, dice = combined_loss(pred, target)
    print(f"Total loss: {loss_val.item():.4f}, CE: {ce.item():.4f}, Dice: {dice.item():.4f}")
    
    print("\n4. Focal Loss:")
    focal_loss = get_loss_function('focal', num_classes=C)
    loss_val = focal_loss(pred, target)
    print(f"Loss value: {loss_val.item():.4f}")
    
    print("\nAll loss functions tested successfully!")
