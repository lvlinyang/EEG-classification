import numpy as np
from torch import nn
from torch.nn import init,Module,Sequential
import torch
from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var
from torch.nn.functional import interpolate


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

def _transpose_time_to_spat(x):
    return x.permute(0, 3,2,1)




class SelfShallowBaseLine (Module):
    def __init__(self,
                 conv_nonlin=square,
                 pool_nonlin=safe_log):
        self.__dict__.update(locals())
        # del self.self
        super(SelfShallowBaseLine,self).__init__()

        # self.conv = Sequential()
        self.transpose = Expression(_transpose_time_to_spat)
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 40, kernel_size=(25, 1), stride=(1, 1)))


        self.conv1_2 = nn.Sequential(nn.Conv2d(40, 40, kernel_size=(1, 22), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     Expression(conv_nonlin))

        self.pool1_1 = nn.Sequential(nn.AvgPool2d(kernel_size=(75, 1), stride=(15, 1), padding=0),
                                     Expression(pool_nonlin),
                                     nn.Dropout(p = 0.5))

        self.classfier = nn.Sequential(nn.Conv2d(40, 4, kernel_size=(69, 1), stride=(1, 1)),
                                       nn.LogSoftmax(dim = 1),
                                       Expression(_squeeze_final_output))


    def forward(self,x):
        x_raw = self.transpose(x)


        batch, C, Height, Width = x_raw.size()

        x1_1 = self.conv1_1(x_raw)
        x1_2 = self.conv1_2(x1_1)
        xpool1_1 = self.pool1_1(x1_2)
        output = self.classfier(xpool1_1)

        return output


if __name__ == '__main__':

    input = torch.rand(32,22,1125)
    model = SelfShallowBaseLine()
    out = model(input)


    print("hello lly")



















