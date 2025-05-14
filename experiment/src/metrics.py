import gc
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from sklearn.metrics import recall_score
import numpy as np

def mIoU(pred_mask, mask, smooth=1e-8, n_classes=2, output_eachcls=False):
    """
    Compute background IoU, mean IoU, and mean Dice score.
    
    Args:
        pred_mask (torch.Tensor): Predicted logits [batch, n_classes, H, W].
        mask (torch.Tensor): Ground truth mask, either one-hot or class indices.
        smooth (float): Smoothing constant.
        n_classes (int): Number of classes.
        output_eachcls (bool): If True, return per-class IoU for non-background.
        
    Returns:
        If output_eachcls is False:
            (bg_iou, mean_iou, mean_dice)
        Otherwise:
            (bg_iou, per_class_iou_array, None)
    """
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        if len(mask.shape) == 4:
            mask = torch.argmax(mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        
        iou_per_class = []
        dice_per_class = []
        for cls in range(n_classes):
            pred_cls = pred_mask == cls
            true_cls = mask == cls
            intersect = torch.logical_and(pred_cls, true_cls).sum().float().item()
            union = torch.logical_or(pred_cls, true_cls).sum().float().item()
            cls_iou = (intersect + smooth) / (union + smooth)
            cls_dice = (2 * intersect + smooth) / (pred_cls.sum().float() + true_cls.sum().float() + smooth)
            if cls == 0:
                bg_iou = cls_iou
            else:
                iou_per_class.append(cls_iou)
                dice_per_class.append(cls_dice)
        if output_eachcls:
            return bg_iou, np.array(iou_per_class), None
        else:
            return bg_iou, np.nanmean(iou_per_class), np.nanmean(dice_per_class)

def pixel_accuracy(predictions, targets):
    """
    Calculate pixel-wise accuracy.
    
    Args:
        predictions (torch.Tensor): Model predictions [batch, n_classes, H, W].
        targets (torch.Tensor): Ground truth mask [batch, H, W] or one-hot encoded [batch, n_classes, H, W].
    
    Returns:
        float: Pixel accuracy.
    """
    predicted = torch.argmax(predictions, dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.numel()
    return correct / total

def calculate_recall_precision(predictions, targets, num_classes=2, smooth=1e-6):
    """
    Calculate mean recall and precision for non-background classes.
    
    Args:
        predictions (torch.Tensor): Model predictions [batch, n_classes, H, W].
        targets (torch.Tensor): Ground truth mask.
        num_classes (int): Number of classes.
        smooth (float): Smoothing constant.
        
    Returns:
        tuple: (mean recall, mean precision, background recall, background precision)
    """
    pred_max = torch.argmax(predictions, dim=1)
    if len(targets.shape) == 4:
        target_max = torch.argmax(targets, dim=1)
    else:
        target_max = targets

    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    for i in range(num_classes):
        pred_mask = (pred_max == i)
        target_mask = (target_max == i)
        true_positives[i] = (pred_mask & target_mask).sum()
        false_positives[i] = (pred_mask & ~target_mask).sum()
        false_negatives[i] = ((~pred_mask) & target_mask).sum()

    recalls_bg = true_positives[0] / (true_positives[0] + false_negatives[0] + smooth)
    precisions_bg = true_positives[0] / (true_positives[0] + false_positives[0] + smooth)

    recalls = true_positives[1:] / (true_positives[1:] + false_negatives[1:] + smooth)
    precisions = true_positives[1:] / (true_positives[1:] + false_positives[1:] + smooth)

    mean_recall = recalls.mean().item()
    mean_precision = precisions.mean().item()

    return mean_recall, mean_precision, recalls_bg, precisions_bg

def calc_metrics(predictions, targets, num_classes=2, smooth=1e-6):
    """
    Calculate metrics using segmentation_models_pytorch functions.
    
    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth mask.
        num_classes (int): Number of classes.
        smooth (float): Smoothing constant.
        
    Returns:
        tuple: (iou_score, recall, precision)
    """
    tp, fp, fn, tn = smp.metrics.get_stats(predictions, targets.to(torch.long), mode='multilabel', threshold=0.5)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro", class_weights=[0, 1])
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro", class_weights=[0, 1])
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise", class_weights=[0, 1])
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise", class_weights=[0, 1])
    return iou_score, recall, precision

def calc_batch_noise_recall(cfg, output, noisy_mask, clean_mask):
    """
    Calculate batch-level noise recall.
    
    Args:
        cfg: Configuration object containing cfg.data.num_classes.
        output (torch.Tensor): Model output logits.
        noisy_mask (torch.Tensor): Noisy segmentation mask.
        clean_mask (torch.Tensor): Clean segmentation mask.
        
    Returns:
        float: Batch-level noise recall.
    """
    with torch.no_grad():
        noise_target = (noisy_mask != clean_mask).any(dim=1).long()
        pred_mask_argmax = output.argmax(dim=1)
        
        noise_pred = torch.zeros_like(pred_mask_argmax)
        noise_pred[noise_target == 1] = pred_mask_argmax[noise_target == 1]
        
        class_recalls = []
        for i in range(1, cfg.data.num_classes):
            class_preds = (noise_pred == i).cpu().numpy().flatten()
            class_targets = ((noise_target == 1) & (clean_mask.argmax(dim=1) == i)).cpu().numpy().flatten()
            if class_targets.sum() > 0:
                r = recall_score(class_targets, class_preds)
                class_recalls.append(r)
        batch_recall = sum(class_recalls) / (cfg.data.num_classes - 1)
    return batch_recall
