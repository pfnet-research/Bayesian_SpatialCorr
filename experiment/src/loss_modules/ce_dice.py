
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .dice import DiceLoss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
    
        self.ce = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss"

    def forward(self, inputs, targets, img_names):
        targets = targets.argmax(dim=1)
        
        celoss = self.ce(inputs, targets)

        return celoss


# class CEDiceLoss(nn.Module):
#     def __init__(self):
#         # super(CEDiceLoss, self).__init__()
#         super().__init__()
    
#         self.ce = nn.CrossEntropyLoss()
#         self.dice = DiceLoss()

#     @property
#     def names(self):
#         return "loss" #, "loss_ce", "loss_dice"

#     def forward(self, inputs, targets, img_names):
#         targets = targets.argmax(dim=1)
        
#         celoss = self.ce(inputs, targets)

#         diceloss = self.dice(inputs, targets, img_names)

#         loss = celoss + diceloss

#         # return loss, celoss, diceloss
#         return loss