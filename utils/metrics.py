"""
Evaluation Metrics for Semantic Segmentation
Includes mIoU, mPA, mF1 as described in the paper
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """
    Calculate segmentation metrics:
    - mIoU (mean Intersection over Union)
    - mPA (mean Pixel Accuracy)
    - mF1 (mean F1 Score)
    """
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """
        Update metrics with new predictions
        
        Args:
            pred: Predicted labels [B, H, W] or [B, C, H, W] (logits)
            target: Ground truth labels [B, H, W]
        """
        # Convert to numpy
        if torch.is_tensor(pred):
            if pred.ndim == 4:  # If logits [B, C, H, W]
                pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
        
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        # Flatten
        pred = pred.flatten()
        target = target.flatten()
        
        # Remove ignored indices
        valid_mask = (target != self.ignore_index) & (target < self.num_classes)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Update confusion matrix
        cm = confusion_matrix(
            target, pred, 
            labels=np.arange(self.num_classes)
        )
        self.confusion_matrix += cm
    
    def get_iou_per_class(self):
        """Calculate IoU for each class"""
        iou_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator == 0:
                iou = 0.0
            else:
                iou = tp / denominator
            
            iou_per_class.append(iou)
        
        return np.array(iou_per_class)
    
    def get_miou(self):
        """Calculate mean IoU"""
        iou_per_class = self.get_iou_per_class()
        return np.mean(iou_per_class)
    
    def get_pixel_accuracy_per_class(self):
        """Calculate pixel accuracy for each class"""
        pa_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            total = self.confusion_matrix[i, :].sum()
            
            if total == 0:
                pa = 0.0
            else:
                pa = tp / total
            
            pa_per_class.append(pa)
        
        return np.array(pa_per_class)
    
    def get_mpa(self):
        """Calculate mean Pixel Accuracy"""
        pa_per_class = self.get_pixel_accuracy_per_class()
        return np.mean(pa_per_class)
    
    def get_f1_per_class(self):
        """Calculate F1 score for each class"""
        f1_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            f1_per_class.append(f1)
        
        return np.array(f1_per_class)
    
    def get_mf1(self):
        """Calculate mean F1 Score"""
        f1_per_class = self.get_f1_per_class()
        return np.mean(f1_per_class)
    
    def get_all_metrics(self):
        """Get all metrics at once"""
        return {
            'mIoU': self.get_miou(),
            'mPA': self.get_mpa(),
            'mF1': self.get_mf1(),
            'IoU_per_class': self.get_iou_per_class(),
            'PA_per_class': self.get_pixel_accuracy_per_class(),
            'F1_per_class': self.get_f1_per_class()
        }
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        return self.confusion_matrix
    
    def __str__(self):
        """String representation of metrics"""
        metrics = self.get_all_metrics()
        
        s = f"Segmentation Metrics:\n"
        s += f"  mIoU: {metrics['mIoU']:.4f}\n"
        s += f"  mPA:  {metrics['mPA']:.4f}\n"
        s += f"  mF1:  {metrics['mF1']:.4f}\n"
        s += f"\nPer-class IoU:\n"
        
        for i, iou in enumerate(metrics['IoU_per_class']):
            s += f"  Class {i}: {iou:.4f}\n"
        
        return s


def calculate_metrics(pred, target, num_classes, ignore_index=255):
    """
    Convenience function to calculate metrics
    
    Args:
        pred: Predicted labels or logits
        target: Ground truth labels
        num_classes: Number of classes
        ignore_index: Index to ignore in evaluation
    
    Returns:
        Dictionary with mIoU, mPA, mF1
    """
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(pred, target)
    return metrics.get_all_metrics()


if __name__ == "__main__":
    # Test metrics
    print("Testing segmentation metrics...")
    
    num_classes = 6
    B, H, W = 2, 64, 64
    
    # Generate dummy predictions and targets
    pred_logits = torch.randn(B, num_classes, H, W)
    pred_labels = torch.argmax(pred_logits, dim=1)
    target = torch.randint(0, num_classes, (B, H, W))
    
    # Test with logits
    print("\n1. Testing with logits:")
    metrics1 = SegmentationMetrics(num_classes)
    metrics1.update(pred_logits, target)
    print(metrics1)
    
    # Test with labels
    print("\n2. Testing with labels:")
    metrics2 = SegmentationMetrics(num_classes)
    metrics2.update(pred_labels, target)
    print(metrics2)
    
    # Test convenience function
    print("\n3. Testing convenience function:")
    results = calculate_metrics(pred_logits, target, num_classes)
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mPA: {results['mPA']:.4f}")
    print(f"mF1: {results['mF1']:.4f}")
    
    print("\nAll metrics tested successfully!")
