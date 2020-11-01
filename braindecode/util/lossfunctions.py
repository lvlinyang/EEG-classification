from torch import nn
import torch.nn.functional as F



def loss_function (pred_lable,ture_lable,feature_1,feature_2,raw,guide):
    loss_1 = F.nll_loss(pred_lable, ture_lable)

    loss_2 = nn.MSELoss(feature_1, feature_2)

    loss_3 = nn.MSELoss(raw, guide)

    loss = loss_1 + loss_2 + loss_3

    return loss

def loss_function2 (feature_1,feature_2,raw,guide):

    loss_2 = nn.MSELoss(feature_1, feature_2)

    loss_3 = nn.MSELoss(raw, guide)

    loss = loss_2 + loss_3

    return loss



