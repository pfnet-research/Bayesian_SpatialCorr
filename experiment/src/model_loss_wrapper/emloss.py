import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import segmentation_models_pytorch as smp

epsilon = 1e-8


def onehot4binarymask(targets):
    """
    Convert a binary mask (shape: [B, H, W]) with values {0, 1}
    to a two-channel one-hot encoding (background and foreground).
    
    Args:
        targets (torch.Tensor): Binary mask of shape (B, H, W).
    
    Returns:
        torch.Tensor: One-hot encoded mask of shape (B, 2, H, W).
    """
    true_targets = targets.unsqueeze(1)
    bg_targets = 1.0 - true_targets
    return torch.cat([bg_targets, true_targets], dim=1)


def dice_loss(predictions, targets, image_id, smooth=1.0, label_smoothing=0, img_wise=False):
    """
    Compute the Dice loss.
    
    Args:
        predictions (torch.Tensor): Predicted probabilities/logits of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth mask. If shape is (B, H, W), it is converted using onehot4binarymask.
        image_id: Unused (for compatibility).
        smooth (float): Smoothing constant.
        label_smoothing (float): Label smoothing factor (unused here).
        img_wise (bool): If True, compute loss per image.
    
    Returns:
        torch.Tensor: Dice loss.
    """
    if len(targets.shape) == 3:
        targets = onehot4binarymask(targets)
    
    intersection = (predictions * targets).sum(dim=(-2, -1))
    sum_preds = predictions.sum(dim=(-2, -1))
    sum_targets = targets.sum(dim=(-2, -1))
    
    dice = (2 * intersection + smooth) / (sum_preds + sum_targets + smooth)
    
    if not img_wise:
        return 1 - dice.mean()
    else:
        return 1 - dice.mean(1)


def cross_entropy(predictions, targets, image_id, label_smoothing=0, img_wise=False):
    """
    Compute Cross-Entropy Loss.
    
    Args:
        predictions (torch.Tensor): Predictions of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth mask. If targets has more than 3 dimensions, the argmax is taken.
        image_id: Unused (for compatibility).
        label_smoothing (float): Label smoothing factor.
        img_wise (bool): If True, compute loss per image.
    
    Returns:
        torch.Tensor: Cross-Entropy loss.
    """
    if len(targets.shape) > 3:
        _, targets = targets.max(dim=1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss = criterion(predictions, targets)
    return loss


def focal_loss(predictions, targets, image_id, label_smoothing=0, img_wise=False):
    """
    Compute the Focal Loss using segmentation_models_pytorch.
    
    Args:
        predictions (torch.Tensor): Predictions of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth mask.
        image_id: Unused (for compatibility).
        label_smoothing (float): Label smoothing factor.
        img_wise (bool): If True, compute loss per image.
    
    Returns:
        torch.Tensor: Focal loss.
    """
    criterion = smp.losses.FocalLoss('binary', gamma=2.0)
    loss = criterion(predictions, targets)
    return loss


def bce_with_logits_loss(predictions, targets, image_id, img_wise=False, label_smoothing=0):
    """
    Compute BCEWithLogitsLoss.
    
    Args:
        predictions (torch.Tensor): Predictions of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth mask. If shape is (B, H, W), convert to one-hot encoding.
        image_id: Unused (for compatibility).
        img_wise (bool): If True, compute loss per image.
        label_smoothing (float): Label smoothing factor (unused here).
    
    Returns:
        torch.Tensor: BCE with logits loss.
    """
    if len(targets.shape) == 3:
        targets = F.one_hot(targets, num_classes=predictions.shape[1]).permute(0, 3, 1, 2).float()
    loss_function = torch.nn.BCEWithLogitsLoss()
    loss = loss_function(predictions, targets)
    return loss


def dice_ce_loss(predictions, targets, image_id, smooth=1.0, label_smoothing=0):
    """
    Compute the sum of Dice loss and Cross-Entropy loss.
    
    Args:
        predictions (torch.Tensor): Predictions.
        targets (torch.Tensor): Ground truth.
        image_id: Unused (for compatibility).
        smooth (float): Smoothing constant.
        label_smoothing (float): Label smoothing factor.
    
    Returns:
        torch.Tensor: Combined loss.
    """
    return (dice_loss(predictions, targets, image_id, smooth, label_smoothing) + 
            cross_entropy(predictions, targets, image_id, label_smoothing))


class EMloss(nn.Module):
    def __init__(self, model, label_smooth=0.05, class_num=2, freeze_confusion=False, e0_T=1.0, efinal_T=1.0, gamma=1.0, enfepoch=100):
        """
        EM Loss (version 5) for training with synthetic noisy labels.
        
        Args:
            model (nn.Module): The segmentation model.
            label_smooth (float): Label smoothing factor.
            class_num (int): Number of classes.
            freeze_confusion (bool): If True, the confusion matrix is not updated.
            e0_T (float): Initial temperature.
            efinal_T (float): Final temperature.
            gamma (float): Gamma factor for temperature scaling.
            enfepoch (int): Epoch at which the temperature scaling reaches efinal_T.
        """
        super(EMloss, self).__init__()
        self.model = model
        self.class_num = class_num
        self.freeze_confusion = freeze_confusion
        self.temperature = e0_T
        self.e0_T = e0_T
        self.efinal_T = efinal_T
        self.gamma = gamma
        self.enfepoch = enfepoch

        # Initialize confusion matrix with label smoothing
        confusion_matrix_init = torch.full((class_num, class_num),
                                             np.log((label_smooth + epsilon) / (class_num - 1)))
        for i in range(class_num):
            confusion_matrix_init[i, i] = np.log(1 - label_smooth + epsilon)
        self.confusion_matrix = nn.Parameter(confusion_matrix_init, requires_grad=not freeze_confusion)

    def _temperature_softmax(self, x, dim, T=1.0):
        """
        Apply softmax with temperature scaling.
        
        Args:
            x (torch.Tensor): Input tensor.
            dim (int): Dimension to apply softmax.
            T (float): Temperature.
        
        Returns:
            torch.Tensor: Softmax output.
        """
        return torch.exp(x / T) / torch.sum(torch.exp(x / T), dim=dim, keepdim=True)

    def temperature_scaling(self, epoch):
        """
        Adjust the temperature based on the current epoch using a cosine schedule.
        
        Args:
            epoch (int): Current epoch.
        """
        if epoch < self.enfepoch:
            t = epoch / self.enfepoch
            cos_term = np.cos(np.pi * t)
            self.temperature = self.efinal_T + (self.e0_T - self.efinal_T) * ((1 + cos_term) / 2) ** self.gamma
        else:
            self.temperature = self.efinal_T
        print('Temperature:', self.temperature)

    def forward(self, images, targets, image_id, label_smoothing=0, visualize_q=False):
        """
        Forward pass for EM loss.
        
        Args:
            images (torch.Tensor): Input images.
            targets (torch.Tensor): Ground truth segmentation masks (one-hot or class indices).
            image_id: Unused (for compatibility).
            label_smoothing (float): Label smoothing factor (unused here).
            visualize_q (bool): If True, return the computed q.
        
        Returns:
            If visualize_q is False:
                (predictions, batch_loss)
            Otherwise:
                (predictions, batch_loss, q)
        """
        predictions = self.model(images).float()
        predictions = self._temperature_softmax(predictions, dim=1, T=self.temperature)
        
        if len(targets.shape) > 3:
            _, targets = targets.max(dim=1)
        
        batch_loss = []
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            img_loss = []
            q_clswise = []
            pre_p_l = []
            for cls_id in range(self.class_num):
                normalized_conf_matrix = F.softmax(self.confusion_matrix, dim=1)
                pre_p = torch.gather(normalized_conf_matrix[cls_id].expand(target.size(0), -1), 1, target)
                numel_p = pre_p * pred[cls_id]
                pre_p_l.append(pre_p)
                q_clswise.append(numel_p.detach().clone())
            q = torch.stack(q_clswise)
            q = q / q.sum(0, keepdim=True)
            for cls_id in range(self.class_num):
                if not self.freeze_confusion:
                    img_loss.append(
                        q[cls_id] * (torch.log(pre_p_l[cls_id] + epsilon) + torch.log(pred[cls_id] + epsilon))
                    )
                else:
                    img_loss.append(q[cls_id] * torch.log(pred[cls_id] + epsilon))
            batch_loss.append(torch.stack(img_loss).sum(0).mean())
        batch_loss = -torch.stack(batch_loss).mean()
        
        if not visualize_q:
            return predictions, batch_loss
        else:
            return predictions, batch_loss, q
