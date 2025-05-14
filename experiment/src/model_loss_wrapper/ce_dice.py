
import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, model):
        super(CELoss, self).__init__()
        self.model = model
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def names(self):
        return "loss"

    def forward(self, image, targets, img_names=None, device=None):
        output = self.model(image).float() 
        loss = self.bce_loss(output, targets.float())
        return output, loss
