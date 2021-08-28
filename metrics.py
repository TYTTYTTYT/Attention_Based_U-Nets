import torch
import torch.nn as nn

class IoU_Score(nn.Module):

    def __init__(self):
        super(IoU_Score, self).__init__()
        self.EPS = 1e-6

    def forward(self, outputs, labels):
        outputs = outputs.int()
        labels = labels.int()
        # Taken from: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
        intersection = (outputs & labels).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((2, 3))  # Will be zero if both are 0

        iou = (intersection + self.EPS) / (union + self.EPS)  # We smooth our devision to avoid 0/0
        return iou.mean()
        