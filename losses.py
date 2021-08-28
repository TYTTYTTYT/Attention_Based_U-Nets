import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        
        return

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
    
# Credits: "https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Focal-Loss"

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        return
        
    def forward(self, inputs, targets, smooth=1): 
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss

class Dice_Focal(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, size_average=True):
        super(Dice_Focal, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha, gamma)
        
    def forward(self, y_pred, y_true):
        dice_loss = self.dice.forward(y_pred, y_true)
        focal_loss = self.focal.forward(y_pred, y_true)
        
        total_loss = dice_loss + focal_loss
        return total_loss