import torch
from torch import nn
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import earthnet as en

def get_loss_from_name(loss_name):
    if loss_name == "l2":
        return Cube_loss(nn.MSELoss())
    elif loss_name == "l1":
        return Cube_loss(nn.L1Loss())
    elif loss_name == "BCE":
        return Cube_loss_binary(nn.BCELoss(reduction='sum'))
    elif loss_name == "Huber":
        return Cube_loss(nn.HuberLoss())
    elif loss_name == "SSIM":
        return Cube_acc(SSIM())
    elif loss_name == "Ensamble":
        return Ensamble_loss()

# simple L2 loss on the RGBI channels, mostly used for training
class Cube_loss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.l = loss
    
    def forward(self, labels: torch.Tensor, prediction: torch.Tensor, mask_channel):
        # only compute loss on non-cloudy pixels
        mask = 1 - labels[:, mask_channel:mask_channel+1] # [b, 1, h, w, t]
        mask = mask.repeat(1, mask_channel, 1, 1, 1)
        masked_prediction = prediction * mask
        masked_labels = labels[:, :mask_channel] * mask
        loss = self.l(masked_prediction, masked_labels)
        return loss
    
class Cube_loss_binary(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.l = loss
    
    def forward(self, labels: torch.Tensor, prediction: torch.Tensor, mask_channel):
        # only compute loss on non-cloudy pixels
        mask = 1 - labels[:, mask_channel:mask_channel+1] # [b, 1, h, w, t]
        mask = mask.repeat(1, mask_channel, 1, 1, 1)
        masked_prediction = prediction * mask
        masked_labels = labels[:, :mask_channel] * mask
        loss = self.l(nn.Sigmoid()(masked_prediction), masked_labels)
        return loss
    
# simple L2 loss on the RGBI channels, mostly used for training
class Cube_acc(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.l = loss
    
    def forward(self, labels: torch.Tensor, prediction: torch.Tensor, mask_channel):
        # only compute loss on non-cloudy pixels
        mask = 1 - labels[:, mask_channel:mask_channel+1] # [b, 1, h, w, t]
        mask = mask.repeat(1, mask_channel, 1, 1, 1)
        masked_prediction = prediction * mask
        masked_labels = labels[:, :mask_channel] * mask
        
        masked_prediction = masked_prediction.permute(0, 4, 1, 2, 3).squeeze(2)
        masked_labels = masked_labels.permute(0, 4, 1, 2, 3).squeeze(2)
        
        loss = (1 - self.l(masked_prediction, masked_labels))
        return loss

# loss using the Ensemble score
class Ensamble_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels: torch.Tensor, prediction: torch.Tensor, mask_channel):
        '''
            size of labels (b, w, h, c, t)
                b = batch_size (>0)
                c = channels (=2)
                w = width (=128)
                h = height (=128)
                t = time (=10)
            size of prediction (b, w, h, c, t)
                b = batch_size (>0)
                c = channels (=1) no mask
                w = width (=128)
                h = height (=128)
                t = time (=10)
        '''
        ssim = Cube_acc(SSIM())(labels, prediction, mask_channel)
        mse = Cube_loss(nn.MSELoss())(labels, prediction, mask_channel)
        huber = Cube_loss(nn.HuberLoss())(labels, prediction, mask_channel)
        
        loss = 3 / (1 / ssim + 1 / mse + 1 / huber)
        
        return loss, ssim, mse, huber
      

